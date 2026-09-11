<script>
    import { onMount } from 'svelte';
    import ModelWorkbench from './ModelWorkbench.svelte';
    import ExportPanel from './ExportPanel.svelte';
    import ProjectOpen from './ProjectOpen.svelte';
    import ExplorePanel from './ExplorePanel.svelte';
    import VariableScreening from './VariableScreening.svelte';
    import SplitSettings from './SplitSettings.svelte';
    let comparison = '',
        modelContext = { fitted: [], selected: '', champion: null };
    let view = 'variables',
        resultsReady = false;
    function modelState(snapshot) {
        const keepDraft = dirty;
        state = snapshot;
        preview = null;
        if (!keepDraft) {
            draft = structuredClone(snapshot.setup);
            jsonText = roleJson();
        }
    }
    const pageTitles = {
        project: 'Project & data',
        variables: 'Variables',
        explore: 'Explore',
        model: 'Model',
        diagnostics: 'Diagnostics',
        compare: 'Compare',
        tables: 'Rate tables',
        export: 'Export',
    };
    let projectInfo = null,
        projectError = '';
    async function navigate(next) {
        view = next;
        if (next === 'project' || next === 'export') {
            try {
                projectInfo = await api('project');
                projectError = '';
            } catch (e) {
                projectError = e.message;
            }
        }
    }
    let state = null,
        draft = null,
        tab = 'table',
        query = '',
        roleFilter = 'all';
    let jsonText = '',
        error = '',
        notice = '',
        busy = false,
        preview = null,
        scrollTop = 0;
    let exploreRequest = { name: '', id: 0 };
    let previewScroll = 0;
    $: previewStart = Math.max(0, Math.floor(previewScroll / 34) - 3);
    const single = ['target', 'weight', 'exposure', 'offset', 'current_premium', 'split', 'time'];
    const groups = ['predictor', 'id', 'unassigned', 'ignore'];
    const roles = [...single, ...groups];
    const types = ['auto', 'numeric', 'categorical'];
    $: roleMap = draft
        ? new Map([
              ...single.filter((r) => draft.assignments[r]).map((r) => [draft.assignments[r], r]),
              ...groups.flatMap((r) => (draft.roles[r] || []).map((n) => [n, r])),
          ])
        : new Map();
    $: typeMap = draft
        ? new Map(Object.entries(draft.types).flatMap(([t, ns]) => ns.map((n) => [n, t])))
        : new Map();
    $: dirty =
        state &&
        draft &&
        (JSON.stringify(draft) !== JSON.stringify(state.setup) ||
            (tab === 'json' && jsonText !== roleJson()));
    $: screeningContext = JSON.stringify([
        state?.session_id,
        state?.project_id,
        state?.revision,
        draft,
        jsonText,
    ]);
    $: filtered = state
        ? state.columns.filter(
              (c) =>
                  c.name.toLowerCase().includes(query.toLowerCase()) &&
                  (roleFilter === 'all' || getRole(c.name, roleMap) === roleFilter),
          )
        : [];
    $: rowOffsets = filtered.reduce(
        (offsets, column) => {
            offsets.push(offsets.at(-1) + (getRole(column.name, roleMap) === 'split' ? 170 : 38));
            return offsets;
        },
        [0],
    );
    $: start = Math.max(0, rowOffsets.findIndex((offset) => offset > scrollTop) - 6);
    $: visible = filtered.slice(start, start + 28);
    $: counts =
        state && draft
            ? roles.map((role) => ({
                  role,
                  count: state.columns.filter((c) => getRole(c.name, roleMap) === role).length,
              }))
            : [];

    let token = '',
        serverSession = '',
        bootstrapPending = null,
        reconcilePending = null;
    let differentProject = false;

    function responseError(data, fallback) {
        return new Error(
            typeof data.detail === 'string'
                ? data.detail
                : data.detail
                  ? JSON.stringify(data.detail)
                  : fallback,
        );
    }
    async function bootstrap() {
        if (!bootstrapPending) {
            bootstrapPending = (async () => {
                const response = await fetch('/api/session', {
                    cache: 'no-store',
                    credentials: 'same-origin',
                });
                const data = await response.json();
                if (!response.ok)
                    throw responseError(data, 'Cannot reconnect to the local server.');
                if (!data.token || !data.session_id)
                    throw new Error(
                        'The server needs the current workbench build. Your draft is kept.',
                    );
                token = data.token;
                serverSession = data.session_id;
            })().finally(() => {
                bootstrapPending = null;
            });
        }
        return bootstrapPending;
    }
    async function send(path, body) {
        return fetch('/api/' + path, {
            method: body ? 'POST' : 'GET',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/json', 'X-EasyGLM-Token': token },
            ...(body ? { body: JSON.stringify(body) } : {}),
        });
    }
    async function reconcile(force = false) {
        if (!state || (!force && state.session_id === serverSession)) return;
        if (!reconcilePending) {
            reconcilePending = (async () => {
                const response = await send('variables');
                const fresh = await response.json();
                if (!response.ok)
                    throw responseError(fresh, 'Cannot read current settings. Your draft is kept.');
                if (fresh.project_id !== state.project_id) {
                    differentProject = true;
                    preview = null;
                    error =
                        'This address now serves a different project. Your draft is kept. Download it before discarding and loading the new project.';
                    throw new Error(error);
                }
                // Update only the applied baseline. Never replace the browser draft or raw JSON text.
                state = fresh;
                preview = null;
                error = '';
                notice =
                    'Reconnected to the local server. Your draft is kept; preview changes before applying.';
            })().finally(() => {
                reconcilePending = null;
            });
        }
        return reconcilePending;
    }
    async function api(path, body, asFile = false) {
        if (differentProject)
            throw new Error(
                'This address serves a different project. Download your kept draft before discarding and reloading.',
            );
        if (!token) await bootstrap();
        const requestToken = token;
        let response;
        try {
            response = await send(path, body);
        } catch {
            throw new Error(
                'Cannot reach the local server. Your draft is kept. Start the server, then reconnect.',
            );
        }
        let data = !asFile || !response.ok ? await response.json() : null;
        if (response.status === 401 && data.code === 'session_expired') {
            // Concurrent failed reads share one bootstrap. Never retry origin/host denials.
            if (requestToken === token) await bootstrap();
            await reconcile();
            if (body && body.session_id !== serverSession) {
                throw new Error(
                    asFile
                        ? 'The server reconnected. Please download again.'
                        : 'The server reconnected. Your draft is kept. Preview changes again before applying.',
                );
            }
            response = await send(path, body);
            data = !asFile || !response.ok ? await response.json() : null;
        }
        if (!response.ok) {
            const failure = responseError(data, 'The request failed. Your draft is kept.');
            failure.status = response.status;
            throw failure;
        }
        if (asFile) {
            const disposition = response.headers.get('content-disposition') || '';
            const encoded = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
            const plain = disposition.match(/filename="([^"]+)"/i)?.[1];
            return {
                blob: await response.blob(),
                filename: encoded ? decodeURIComponent(encoded) : plain || 'easyglm-export',
            };
        }
        return data;
    }
    async function uploadProject(file, body) {
        if (differentProject)
            throw new Error(
                'This address serves a different project. Download your kept draft before discarding and reloading.',
            );
        if (!token) await bootstrap();
        const requestToken = token;
        const query = new URLSearchParams({ ...body, filename: file.name });
        const sendUpload = () =>
            fetch('/api/project/upload?' + query, {
                method: 'POST',
                cache: 'no-store',
                credentials: 'same-origin',
                headers: { 'Content-Type': 'application/octet-stream', 'X-EasyGLM-Token': token },
                body: file,
            });
        let response;
        try {
            response = await sendUpload();
        } catch {
            throw new Error(
                'Cannot reach the local server. Reconnect before opening the file again.',
            );
        }
        let data = await response.json();
        if (response.status === 401 && data.code === 'session_expired') {
            if (requestToken === token) await bootstrap();
            await reconcile();
            if (body.session_id !== serverSession)
                throw new Error('The server reconnected. Click Open again to confirm this file.');
            response = await sendUpload();
            data = await response.json();
        }
        if (!response.ok) {
            const failure = responseError(data, 'Cannot open this file.');
            failure.status = response.status;
            throw failure;
        }
        return data;
    }
    function projectName(name) {
        return name === 'French motor · Svelte review'
            ? 'Motor insurance example'
            : name || 'Untitled project';
    }
    function sourceFilename(info) {
        return info?.data?.source?.path?.split(/[\\/]/).pop() || 'Data in this session';
    }
    function countLabel(value, singular) {
        return `${value.toLocaleString()} ${singular}${value === 1 ? '' : 's'}`;
    }
    async function reconnect() {
        busy = true;
        try {
            await bootstrap();
            await reconcile(true);
            if (!state) await reload();
            else {
                error = '';
            }
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function downloadDraft() {
        const blob = new Blob(
            [
                JSON.stringify(
                    {
                        project: state?.name,
                        columns: state?.columns,
                        setup: draft,
                        role_json: jsonText,
                    },
                    null,
                    2,
                ),
            ],
            { type: 'application/json' },
        );
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = 'easyglm-variable-draft.json';
        link.click();
        URL.revokeObjectURL(url);
    }
    function roleJson() {
        return JSON.stringify(
            {
                ...draft.assignments,
                ...draft.roles,
                split: draft.assignments.split
                    ? {
                          column: draft.assignments.split,
                          train_value: draft.split.train_value,
                      }
                    : null,
            },
            null,
            2,
        );
    }
    function getRole(name, lookup = roleMap) {
        return lookup.get(name) || 'ignore';
    }
    function getType(name, lookup = typeMap) {
        return lookup.get(name) || 'auto';
    }
    function touch() {
        draft = structuredClone(draft);
        preview = null;
        notice = '';
        error = '';
        jsonText = roleJson();
    }
    function randomSplitDraft() {
        const names = new Set(
            state.columns.map((column) => draft.renames[column.name] || column.name),
        );
        let column = 'traintest',
            suffix = 2;
        while (names.has(column)) column = `traintest_${suffix++}`;
        return { ...draft.split, mode: 'random', column, train_value: 1, holdout_value: null };
    }
    function setRole(name, role) {
        const previousSplit = draft.assignments.split;
        for (const key of single)
            if (draft.assignments[key] === name) draft.assignments[key] = null;
        for (const key of groups)
            draft.roles[key] = (draft.roles[key] || []).filter((c) => c !== name);
        if (single.includes(role)) {
            const old = draft.assignments[role];
            if (old && old !== name) draft.roles.unassigned.push(old);
            draft.assignments[role] = name;
        } else draft.roles[role].push(name);
        if (role === 'split' && (previousSplit !== name || draft.split.mode !== 'column')) {
            draft.split = {
                ...draft.split,
                mode: 'column',
                column: name,
                train_value: null,
                holdout_value: null,
            };
        }
        if (previousSplit === name && role !== 'split') {
            draft.split = randomSplitDraft();
        }
        touch();
    }
    function setType(name, type) {
        for (const key of types)
            if (draft.types[key]) {
                draft.types[key] = draft.types[key].filter((c) => c !== name);
                if (!draft.types[key].length) delete draft.types[key];
            }
        if (type !== 'auto') (draft.types[type] ||= []).push(name);
        touch();
    }
    function setName(name, value) {
        if (value.trim() && value.trim() !== name) draft.renames[name] = value.trim();
        else delete draft.renames[name];
        touch();
    }
    function parseRoles() {
        try {
            const obj = JSON.parse(jsonText);
            if (!obj || Array.isArray(obj) || typeof obj !== 'object')
                throw new Error('Use a JSON object with the ten roles.');
            if (Object.keys(obj).some((r) => !roles.includes(r)))
                throw new Error('Use the ten roles shown in the template.');
            const splitEntry = obj.split;
            if (splitEntry && typeof splitEntry === 'object') {
                if (
                    Array.isArray(splitEntry) ||
                    Object.keys(splitEntry).some(
                        (key) => !['column', 'train_value', 'holdout_value'].includes(key),
                    )
                )
                    throw new Error('split needs column and train_value.');
                obj.split = splitEntry.column;
            }
            const known = new Set(state.columns.map((c) => c.name)),
                used = new Set();
            const assignments = {},
                grouped = {};
            for (const role of roles) {
                const val = Object.hasOwn(obj, role)
                    ? obj[role]
                    : single.includes(role)
                      ? null
                      : [];
                const values = single.includes(role) ? (val === null ? [] : [val]) : val;
                if (!Array.isArray(values)) throw new Error(`${role} must be a list.`);
                for (const name of values) {
                    if (typeof name !== 'string' || !known.has(name))
                        throw new Error(`${role}: unknown source column ${JSON.stringify(name)}.`);
                    if (used.has(name)) throw new Error(`${name} appears more than once.`);
                    used.add(name);
                }
                if (single.includes(role)) assignments[role] = val;
                else grouped[role] = values;
            }
            // Match the established bulk format: omitted columns are ignored.
            grouped.ignore.push(...state.columns.map((c) => c.name).filter((n) => !used.has(n)));
            let split = structuredClone(draft.split);
            if (assignments.split) {
                if (split.column !== assignments.split) {
                    split = {
                        ...split,
                        column: assignments.split,
                        train_value: null,
                        holdout_value: null,
                    };
                }
                split.mode = 'column';
                if (splitEntry && typeof splitEntry === 'object') {
                    split.train_value = splitEntry.train_value ?? null;
                    split.holdout_value = splitEntry.holdout_value ?? null;
                }
            } else if (draft.assignments.split && split.mode === 'column') {
                split = randomSplitDraft();
            }
            draft = { ...draft, assignments, roles: grouped, split };
            preview = null;
            error = '';
            return true;
        } catch (e) {
            error = e.message;
            preview = null;
            return false;
        }
    }
    function switchTab(next) {
        if (tab === 'json' && !parseRoles()) return;
        if (next === 'json') jsonText = roleJson();
        tab = next;
        scrollTop = 0;
    }
    function reset() {
        draft = structuredClone(state.setup);
        jsonText = roleJson();
        error = '';
        notice = 'Draft reset to the last applied settings.';
        preview = null;
    }
    async function reload() {
        busy = true;
        try {
            await bootstrap();
            const response = await send('variables');
            const fresh = await response.json();
            if (!response.ok) throw responseError(fresh, 'Cannot load the local project.');
            state = fresh;
            differentProject = false;
            reset();
            notice = '';
            if (!state.columns.length) await navigate('project');
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function reconnectForOpen() {
        await bootstrap();
        await reconcile(true);
    }
    async function projectOpened(opened, kind = 'data') {
        // The POST already replaced the server project. Adopt its confirmed snapshot
        // before any fallible refresh, so old drafts can never masquerade as new data.
        state = opened;
        differentProject = false;
        reset();
        notice = '';
        tab = 'table';
        query = '';
        roleFilter = 'all';
        scrollTop = 0;
        previewScroll = 0;
        comparison = '';
        modelContext = { fitted: [], selected: '', champion: null };
        resultsReady = false;
        projectInfo = null;
        projectError = '';
        exploreRequest = { name: '', id: 0 };
        token = '';
        serverSession = opened.session_id;
        view = kind === 'example' ? 'project' : 'variables';
        busy = true;
        try {
            await bootstrap();
            const response = await send('variables');
            const fresh = await response.json();
            if (!response.ok) throw responseError(fresh, 'Cannot refresh the connection.');
            if (fresh.project_id !== opened.project_id) {
                differentProject = true;
                throw new Error('Another project has since been opened at this address.');
            }
            state = fresh;
            reset();
            notice = '';
            await navigate(kind === 'example' ? 'project' : 'variables');
        } catch (e) {
            error = 'Data loaded. Reconnect to continue. ' + e.message;
            throw new Error(error);
        } finally {
            busy = false;
        }
    }
    async function review() {
        if (tab === 'json' && !parseRoles()) return;
        busy = true;
        error = '';
        try {
            preview = await api('variables/preview', {
                session_id: state.session_id,
                revision: state.revision,
                setup: draft,
            });
            previewScroll = 0;
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function prepareScreening() {
        return !busy && !differentProject && (tab !== 'json' || parseRoles());
    }
    async function removeScreenedPredictors(names, expectedContext) {
        const current = JSON.stringify([
            state?.session_id,
            state?.project_id,
            state?.revision,
            draft,
            jsonText,
        ]);
        if (busy || differentProject || current !== expectedContext)
            throw new Error('Settings changed. Check predictors again before removing them.');
        const removed = new Set(names);
        if (!removed.size || [...removed].some((name) => !draft.roles.predictor.includes(name)))
            throw new Error('Only predictors in the checked draft can be removed.');
        draft.roles.predictor = draft.roles.predictor.filter((name) => !removed.has(name));
        draft.roles.ignore = [...new Set([...draft.roles.ignore, ...removed])];
        touch();
        await review();
    }
    async function apply() {
        busy = true;
        error = '';
        try {
            state = await api('variables/apply', {
                session_id: state.session_id,
                revision: state.revision,
                setup: draft,
            });
            draft = structuredClone(state.setup);
            jsonText = roleJson();
            preview = null;
            notice = 'Settings applied to this session.';
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function saveDownload(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    async function downloadProject() {
        const data = await api('project');
        const filename = (data.name || 'project').replace(/[\\/:*?"<>|\u0000-\u001f]/g, '_');
        saveDownload(
            new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }),
            filename + '.easyglm-project.json',
        );
    }
    async function download() {
        try {
            await downloadProject();
        } catch (e) {
            error = e.message;
        }
    }
    onMount(async () => {
        try {
            await reload();
            const initialView = new URLSearchParams(window.location.search).get('view');
            if (Object.hasOwn(pageTitles, initialView)) await navigate(initialView);
        } catch (e) {
            error = e.message;
        }
    });
</script>

<svelte:head><title>EasyGLM · {pageTitles[view]}</title></svelte:head>
<div class="shell">
    <aside class="rail">
        <div class="brand">
            <span class="brand-icon">e</span><span>easy<span class="light">glm</span></span>
        </div>
        <div class="workspace-label">LOCAL WORKSPACE</div>
        <div class="portfolio">{state ? projectName(state.name) : 'Opening portfolio…'}</div>
        <nav aria-label="Workbench">
            <div class="nav-label">WORKFLOW</div>
            {#each Object.entries(pageTitles) as [key, title]}
                <button
                    class="nav-link"
                    class:active={view === key}
                    disabled={!state ||
                        (['diagnostics', 'compare', 'tables'].includes(key) && !resultsReady)}
                    onclick={() => navigate(key)}
                    >{title}{#if key === 'variables'}
                        <span class="nav-count">{state?.columns.length || '—'}</span>{/if}</button
                >
            {/each}
        </nav>
        <div class="setup-progress" aria-label="Setup progress">
            <strong>Setup progress</strong>
            <span>{state?.columns.length ? '✓' : '○'} Data loaded</span>
            <span
                >{state?.setup.assignments.target && state?.setup.roles.predictor.length
                    ? '✓'
                    : '○'} Target and predictors</span
            >
            <span>{state?.models.length ? '✓' : '○'} Model defined</span>
            <span>{resultsReady ? '✓' : '○'} Model fitted</span>
        </div>
        <div class="rail-bottom">
            <span class="status-dot"></span> Local session
            <div class="version">EasyGLM 0.463</div>
        </div>
    </aside>
    <div class="workspace">
        <header>
            <div class="breadcrumb">
                Workbench <span>/</span>
                <strong>{pageTitles[view]}</strong>
            </div>
            <div class="header-actions">
                <button class="quiet" onclick={download} disabled={!state}>Export project</button>
            </div>
        </header>
        <main>
            <div hidden={view !== 'variables'}>
                <div class="heading">
                    <div>
                        <div class="eyebrow">PORTFOLIO SETUP</div>
                        <h1>Variables</h1>
                        <p>
                            Choose the target and predictors. Numeric values measure an amount;
                            categorical values identify a group.
                        </p>
                    </div>
                    <div class="dataset-meta">
                        <strong>{state?.row_count.toLocaleString() || '—'}</strong> rows<span
                        ></span><strong>{state?.columns.length || '—'}</strong> variables
                    </div>
                </div>
                <div class="session-note">
                    Edits are held in this session. Export applied settings before stopping the
                    server. Source files are unchanged.
                </div>
                {#if error}<div class="message error" role="alert">
                        <span>{error}</span><button onclick={reconnect} disabled={busy}
                            >Reconnect</button
                        >
                        {#if draft}<button onclick={downloadDraft}>Download draft</button><button
                                onclick={reload}
                                disabled={busy}>Discard draft and reload</button
                            >{/if}
                    </div>{/if}
                {#if notice}<div class="message" role="status">{notice}</div>{/if}
                {#if state && draft}
                    <div class="role-summary">
                        {#each counts as item}<button
                                class:chosen={roleFilter === item.role}
                                onclick={() => {
                                    roleFilter = roleFilter === item.role ? 'all' : item.role;
                                    scrollTop = 0;
                                }}
                                ><span class="dot {item.role}"></span>{item.role.replaceAll(
                                    '_',
                                    ' ',
                                )}<b>{item.count}</b></button
                            >{/each}
                    </div>
                    <fieldset disabled={busy} class="edit-fieldset">
                        <section class="editor-card">
                            <div class="toolbar">
                                <div class="segmented">
                                    <button
                                        class:active={tab === 'table'}
                                        onclick={() => switchTab('table')}>Table</button
                                    ><button
                                        class:active={tab === 'json'}
                                        onclick={() => switchTab('json')}>Role JSON</button
                                    >
                                </div>
                                <span class="edit-status"
                                    >{dirty ? 'Unapplied changes' : 'All changes applied'}</span
                                >
                                <div class="spacer"></div>
                                <button onclick={reset} disabled={busy}>Reset</button><button
                                    class="primary"
                                    onclick={review}
                                    disabled={busy || differentProject}>Preview changes</button
                                >
                            </div>
                            {#if tab === 'table'}
                                <div class="filterbar">
                                    <input
                                        class="search"
                                        aria-label="Search variables"
                                        placeholder="Search source columns…"
                                        bind:value={query}
                                        oninput={() => (scrollTop = 0)}
                                    /><select
                                        aria-label="Filter roles"
                                        bind:value={roleFilter}
                                        onchange={() => (scrollTop = 0)}
                                        ><option value="all">All roles</option
                                        >{#each roles as role}<option value={role}
                                                >{role.replaceAll('_', ' ')}</option
                                            >{/each}</select
                                    ><span>{filtered.length} variables</span>
                                </div>
                                <div class="table-head grid-row">
                                    <span>SOURCE COLUMN</span><span>DISPLAY NAME</span><span
                                        >ROLE</span
                                    ><span>VARIABLE TYPE</span><span>SOURCE TYPE</span>
                                </div>
                                <div
                                    class="virtual-table"
                                    role="region"
                                    aria-label="Variable settings"
                                    onscroll={(e) => (scrollTop = e.currentTarget.scrollTop)}
                                >
                                    <div
                                        style:height={rowOffsets.at(-1) + 'px'}
                                        class="virtual-space"
                                    >
                                        <div
                                            class="virtual-rows"
                                            style:transform={'translateY(' +
                                                rowOffsets[start] +
                                                'px)'}
                                        >
                                            {#each visible as column (column.name)}<div>
                                                    <div
                                                        class="grid-row data-row"
                                                        data-column={column.name}
                                                    >
                                                        <button
                                                            class="column-name"
                                                            title="Explore this variable"
                                                            onclick={() => {
                                                                exploreRequest = {
                                                                    name:
                                                                        state.setup.renames[
                                                                            column.name
                                                                        ] || column.name,
                                                                    id: exploreRequest.id + 1,
                                                                };
                                                                void navigate('explore');
                                                            }}>{column.name}</button
                                                        ><input
                                                            aria-label={'Name for ' + column.name}
                                                            value={draft.renames[column.name] ||
                                                                column.name}
                                                            onchange={(e) =>
                                                                setName(
                                                                    column.name,
                                                                    e.currentTarget.value,
                                                                )}
                                                        /><select
                                                            aria-label={'Role for ' + column.name}
                                                            value={getRole(column.name, roleMap)}
                                                            onchange={(e) =>
                                                                setRole(
                                                                    column.name,
                                                                    e.currentTarget.value,
                                                                )}
                                                            >{#each roles as role}<option
                                                                    value={role}
                                                                    >{role.replaceAll(
                                                                        '_',
                                                                        ' ',
                                                                    )}</option
                                                                >{/each}</select
                                                        ><select
                                                            aria-label={'Type for ' + column.name}
                                                            value={getType(column.name, typeMap)}
                                                            onchange={(e) =>
                                                                setType(
                                                                    column.name,
                                                                    e.currentTarget.value,
                                                                )}
                                                            >{#each types as type}<option
                                                                    value={type}
                                                                    >{type === 'auto'
                                                                        ? 'Infer from data'
                                                                        : type === 'numeric'
                                                                          ? 'Numeric'
                                                                          : 'Categorical'}</option
                                                                >{/each}</select
                                                        ><span class="dtype">{column.dtype}</span>
                                                    </div>
                                                    {#if getRole(column.name, roleMap) === 'split'}
                                                        <SplitSettings
                                                            {api}
                                                            {state}
                                                            setup={draft}
                                                            disabled={busy || differentProject}
                                                            onchange={(next) => {
                                                                draft = next;
                                                                touch();
                                                            }}
                                                        />
                                                    {/if}
                                                </div>{/each}
                                        </div>
                                    </div>
                                </div>
                            {:else}
                                <div class="json-hint">
                                    Roles use source column names. Missing columns become ignored.
                                    The split entry includes its column and training value; the
                                    other value is holdout. Names and types stay as set in the
                                    table.
                                </div>
                                <textarea
                                    class="json-editor"
                                    aria-label="Role JSON"
                                    bind:value={jsonText}
                                    oninput={() => {
                                        preview = null;
                                        notice = '';
                                    }}
                                    spellcheck="false"></textarea>
                            {/if}
                            <div class="table-footer">
                                <span
                                    >Roles determine eligibility. Model selections and interactions
                                    are configured separately.</span
                                ><span>Revision {state.revision}</span>
                            </div>
                        </section>
                        {#if !draft.assignments.split}
                            <section
                                class="model-card variable-random-split"
                                aria-label="Train and holdout split"
                            >
                                <h2>Train / holdout split</h2>
                                <p class="help-text">
                                    Random split. To use an existing column, assign it the split
                                    role above.
                                </p>
                                <div class="random-split-fields">
                                    <label
                                        >Training fraction<input
                                            aria-label="Training fraction"
                                            type="number"
                                            min="0.01"
                                            max="0.99"
                                            step="0.05"
                                            bind:value={draft.split.fraction}
                                            oninput={() => {
                                                draft.split.mode = 'random';
                                                touch();
                                            }}
                                        /></label
                                    >
                                    <label
                                        >Seed<input
                                            aria-label="Split seed"
                                            type="number"
                                            min="0"
                                            step="1"
                                            bind:value={draft.split.seed}
                                            oninput={() => {
                                                draft.split.mode = 'random';
                                                touch();
                                            }}
                                        /></label
                                    >
                                    <label
                                        >Generated column<input
                                            aria-label="Split column name"
                                            bind:value={draft.split.column}
                                            oninput={() => {
                                                draft.split.mode = 'random';
                                                touch();
                                            }}
                                        /></label
                                    >
                                </div>
                                <p class="help-text">
                                    Used by Explore and every model. Apply with your variable
                                    changes.
                                </p>
                            </section>
                        {/if}
                        {#if view === 'variables'}<VariableScreening
                                {api}
                                {state}
                                setup={draft}
                                context={screeningContext}
                                disabled={busy || differentProject}
                                prepare={prepareScreening}
                                onRemove={removeScreenedPredictors}
                            />{/if}
                        {#if preview}<section class="preview-card">
                                <div class="preview-title">
                                    <strong>{preview.changes.length} variable changes</strong
                                    ><button
                                        class="primary"
                                        disabled={busy ||
                                            differentProject ||
                                            !preview.changes.length}
                                        onclick={apply}>Apply changes</button
                                    >
                                </div>
                                <div
                                    class="preview-list"
                                    onscroll={(e) => (previewScroll = e.currentTarget.scrollTop)}
                                >
                                    <div
                                        style:height={preview.changes.length * 34 + 'px'}
                                        style:position="relative"
                                    >
                                        <div
                                            style:position="absolute"
                                            style:width="100%"
                                            style:transform={'translateY(' +
                                                previewStart * 34 +
                                                'px)'}
                                        >
                                            {#each preview.changes.slice(previewStart, previewStart + 18) as change}
                                                <div class="preview-change">
                                                    <b>{change['raw column']}</b><span
                                                        >{change.name}</span
                                                    ><span>{change.role}</span><span
                                                        >{change.type}</span
                                                    >
                                                </div>
                                            {/each}
                                        </div>
                                    </div>
                                </div>
                                {#each preview.notices as item}<p>{item[1]}</p>{/each}
                            </section>{/if}
                    </fieldset>
                {:else}<div class="loading">Opening local data…</div>{/if}
            </div>
            <section hidden={view !== 'project'} class="workflow-page">
                <div class="heading">
                    <div>
                        <h1>Project & data</h1>
                        <p>Open your own data, continue a saved project or try an example.</p>
                    </div>
                </div>
                {#if projectError}<p role="alert">{projectError}</p>{/if}
                {#if state?.columns.length}
                    <section class="loaded-dataset" aria-label="Current dataset">
                        <strong>Loaded:</strong>
                        <span class="source-filename" title={projectInfo?.data?.source?.path || ''}>
                            {sourceFilename(projectInfo)}
                        </span>
                        <span>· {countLabel(state.row_count, 'row')}</span>
                        <span>· {countLabel(state.columns.length, 'column')}</span>
                    </section>
                {/if}
                {#if state}<ProjectOpen
                        {api}
                        upload={uploadProject}
                        {state}
                        onOpened={projectOpened}
                        onReconnect={reconnectForOpen}
                    />{/if}
            </section>
            <section hidden={view !== 'explore'} class="workflow-page">
                {#if state && draft}{#key state.project_id}<ExplorePanel
                            {api}
                            {state}
                            active={view === 'explore'}
                            requestColumn={exploreRequest}
                            onNavigate={navigate}
                            onReconnect={reconnectForOpen}
                        />{/key}{/if}
            </section>
            {#if view === 'export'}<ExportPanel
                    {api}
                    {state}
                    context={modelContext}
                    {comparison}
                    {downloadProject}
                    {saveDownload}
                />{/if}
            {#if state && draft}{#key state.project_id}<ModelWorkbench
                        {api}
                        {state}
                        {view}
                        bind:comparison
                        onContext={(context) => (modelContext = context)}
                        onState={modelState}
                        onReady={(ready) => (resultsReady = ready)}
                        onNavigate={navigate}
                    />{/key}{/if}
        </main>
    </div>
</div>

<style>
    .loaded-dataset {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 4px 8px;
        margin: -8px 0 22px;
        color: var(--muted);
        font-size: 14px;
    }
    .loaded-dataset strong {
        color: var(--text);
    }
    .source-filename {
        overflow-wrap: anywhere;
        min-width: 0;
    }

    .variable-random-split {
        margin: 20px 0;
        padding: 20px;
    }
    .random-split-fields {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 16px;
    }
    .random-split-fields label {
        display: grid;
        gap: 6px;
        font-size: 13px;
    }
    .random-split-fields input {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    @media (max-width: 900px) {
        .random-split-fields {
            grid-template-columns: 1fr;
        }
    }
</style>
