<script>
    import { onMount } from 'svelte';
    import ModelWorkbench from './ModelWorkbench.svelte';
    import ExportPanel from './ExportPanel.svelte';
    import ProjectOpen from './ProjectOpen.svelte';
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
    let selected = '',
        plot = null,
        plotError = '',
        hover = null,
        zoom = 100;
    let plotSequence = 0;
    let previewScroll = 0;
    $: previewStart = Math.max(0, Math.floor(previewScroll / 34) - 3);
    const single = ['target', 'weight', 'exposure', 'offset', 'current_premium', 'split'];
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
            (tab === 'json' &&
                jsonText !== JSON.stringify({ ...draft.assignments, ...draft.roles }, null, 2)));
    $: filtered = state
        ? state.columns.filter(
              (c) =>
                  c.name.toLowerCase().includes(query.toLowerCase()) &&
                  (roleFilter === 'all' || getRole(c.name, roleMap) === roleFilter),
          )
        : [];
    $: start = Math.max(0, Math.floor(scrollTop / 38) - 5);
    $: visible = filtered.slice(start, start + 28);
    $: maxCount = plot ? Math.max(1, ...plot.table.map((r) => r.exposure)) : 1;
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
        if (!response.ok) throw responseError(data, 'The request failed. Your draft is kept.');
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
    async function reconnect() {
        busy = true;
        try {
            await bootstrap();
            await reconcile(true);
            if (!state) await reload();
            else {
                error = '';
                void loadPlot();
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
        return JSON.stringify({ ...draft.assignments, ...draft.roles }, null, 2);
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
    function setRole(name, role) {
        for (const key of single)
            if (draft.assignments[key] === name) draft.assignments[key] = null;
        for (const key of groups)
            draft.roles[key] = (draft.roles[key] || []).filter((c) => c !== name);
        if (single.includes(role)) {
            const old = draft.assignments[role];
            if (old && old !== name) draft.roles.unassigned.push(old);
            draft.assignments[role] = name;
        } else draft.roles[role].push(name);
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
                throw new Error('Unknown role. Use the ten roles shown in the template.');
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
            draft = { ...draft, assignments, roles: grouped };
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
            if (!state.columns.some((c) => c.name === selected))
                selected = state.setup.roles.predictor[0] || state.columns[0]?.name;
            void loadPlot();
            if (!state.columns.length) await navigate('project');
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function projectOpened() {
        comparison = '';
        modelContext = { fitted: [], selected: '', champion: null };
        resultsReady = false;
        await reload();
        await navigate('variables');
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
            void loadPlot();
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function loadPlot() {
        if (!selected) return;
        const sequence = ++plotSequence;
        plotError = '';
        hover = null;
        try {
            const result = await api('plot?column=' + encodeURIComponent(selected));
            if (sequence === plotSequence) plot = result;
        } catch (e) {
            if (sequence === plotSequence) {
                plotError = e.message;
                plot = null;
            }
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
        <div class="portfolio">{state?.name || 'Opening portfolio…'}</div>
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
        {#if modelContext.fitted.length > 1}<label class="sidebar-comparison"
                >Default comparison model<select
                    aria-label="Default comparison model"
                    bind:value={comparison}
                    ><option value="">None</option
                    >{#each modelContext.fitted.filter((n) => n !== modelContext.selected) as name}<option
                            value={name}>{name}</option
                        >{/each}</select
                ></label
            >{/if}
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
            <div class="version">EasyGLM 0.460</div>
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
                        <p>Roles, names and types</p>
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
                                    ><span>MODELLING TYPE</span><span>SOURCE TYPE</span>
                                </div>
                                <div
                                    class="virtual-table"
                                    role="region"
                                    aria-label="Variable settings"
                                    onscroll={(e) => (scrollTop = e.currentTarget.scrollTop)}
                                >
                                    <div
                                        style:height={filtered.length * 38 + 'px'}
                                        class="virtual-space"
                                    >
                                        <div
                                            class="virtual-rows"
                                            style:transform={'translateY(' + start * 38 + 'px)'}
                                        >
                                            {#each visible as column (column.name)}<div
                                                    class="grid-row data-row"
                                                    data-column={column.name}
                                                >
                                                    <button
                                                        class="column-name"
                                                        title="Show distribution"
                                                        onclick={() => {
                                                            selected = column.name;
                                                            loadPlot();
                                                            view = 'explore';
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
                                                        >{#each roles as role}<option value={role}
                                                                >{role.replaceAll('_', ' ')}</option
                                                            >{/each}</select
                                                    ><select
                                                        aria-label={'Type for ' + column.name}
                                                        value={getType(column.name, typeMap)}
                                                        onchange={(e) =>
                                                            setType(
                                                                column.name,
                                                                e.currentTarget.value,
                                                            )}
                                                        >{#each types as type}<option>{type}</option
                                                            >{/each}</select
                                                    ><span class="dtype">{column.dtype}</span>
                                                </div>{/each}
                                        </div>
                                    </div>
                                </div>
                            {:else}
                                <div class="json-hint">
                                    Roles use source column names. Missing columns become ignored.
                                    Names and types stay as set in the table.
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
                        <p>Open your data or continue a saved project.</p>
                    </div>
                </div>
                {#if projectError}<p role="alert">{projectError}</p>{/if}
                {#if state}<ProjectOpen {api} {state} onOpened={projectOpened} />{/if}
                {#if state?.columns.length}<div class="model-card">
                        <h2>{state?.name || 'Current project'}</h2>
                        <div class="result-totals">
                            <span><b>{state?.row_count.toLocaleString()}</b> rows</span><span
                                ><b>{state?.columns.length}</b> source variables</span
                            ><span><b>{state?.models.length}</b> models</span>
                        </div>
                        <p>
                            Applied settings are held in this local session. Export the project to
                            keep them.
                        </p>
                        <div class="model-actions">
                            <button class="primary" onclick={() => navigate('variables')}
                                >Set up variables</button
                            ><button onclick={() => navigate('model')}>Review model</button>
                        </div>
                    </div>
                    {#if projectInfo}<div class="model-card">
                            <h2>Applied data setup</h2>
                            <dl class="project-facts">
                                <dt>Source</dt>
                                <dd>
                                    {projectInfo.data?.source?.path ||
                                        'Data loaded into the local session'}
                                </dd>
                                <dt>Target</dt>
                                <dd>{state?.setup.assignments.target || 'Not assigned'}</dd>
                                <dt>Weight</dt>
                                <dd>{state?.setup.assignments.weight || 'None'}</dd>
                                <dt>Split</dt>
                                <dd>
                                    {projectInfo.data?.split?.mode || 'Not configured'} · {projectInfo
                                        .data?.split?.column || 'No column'}
                                </dd>
                            </dl>
                            <details>
                                <summary>Applied project settings</summary>
                                <pre>{JSON.stringify(projectInfo, null, 2)}</pre>
                            </details>
                        </div>{/if}
                {/if}
            </section>
            <section hidden={view !== 'explore'} class="workflow-page">
                <div class="heading">
                    <div>
                        <h1>Explore</h1>
                        <p>Inspect variable distributions before designing the model.</p>
                    </div>
                </div>
                <p class="help-text">
                    Distributions use applied settings. Apply any Variables draft before reviewing
                    its effect.
                </p>
                {#if state && draft}
                    <section class="plot-card">
                        <div class="plot-heading">
                            <div>
                                <div class="eyebrow">DATA PREVIEW</div>
                                <h2>Variable distribution</h2>
                            </div>
                            <select
                                aria-label="Plot variable"
                                bind:value={selected}
                                onchange={loadPlot}
                                >{#each state.columns as column}<option value={column.name}
                                        >{draft.renames[column.name] || column.name}</option
                                    >{/each}</select
                            >
                            <div class="spacer"></div>
                            <label class="zoom"
                                >Zoom <input
                                    aria-label="Chart zoom"
                                    type="range"
                                    min="100"
                                    max="300"
                                    step="25"
                                    bind:value={zoom}
                                /></label
                            >
                        </div>
                        {#if plotError}<div class="message error">
                                <span>{plotError}</span><button onclick={reconnect} disabled={busy}
                                    >Reconnect</button
                                >
                            </div>{:else if plot}<div class="chart-scroll">
                                <div class="chart" style:width={zoom + '%'}>
                                    <svg
                                        viewBox="0 0 1000 180"
                                        role="img"
                                        aria-label={'Distribution of ' + plot.column}
                                        preserveAspectRatio="none"
                                        ><title>Distribution of {plot.column}</title
                                        >{#each [0, 1, 2, 3] as line}<line
                                                x1="0"
                                                x2="1000"
                                                y1={15 + line * 45}
                                                y2={15 + line * 45}
                                                stroke="#e8eced"
                                            />{/each}{#each plot.table as row, i}<rect
                                                role="img"
                                                aria-label={row.label +
                                                    ': ' +
                                                    row.exposure +
                                                    ' rows'}
                                                x={(i * 1000) / plot.table.length + 4}
                                                y={160 - (row.exposure / maxCount) * 140}
                                                width={Math.max(1, 1000 / plot.table.length - 8)}
                                                height={(row.exposure / maxCount) * 140}
                                                rx="3"
                                                fill={hover === i ? '#194e4b' : '#55a69d'}
                                                onmouseenter={() => (hover = i)}
                                                onmouseleave={() => (hover = null)}
                                                ><title
                                                    >{row.label}: {row.exposure.toLocaleString()} rows</title
                                                ></rect
                                            >{/each}</svg
                                    >
                                </div>
                            </div>
                            <div class="chart-caption">
                                <strong
                                    >{hover !== null
                                        ? plot.table[hover].label +
                                          ' · ' +
                                          plot.table[hover].exposure.toLocaleString() +
                                          ' rows'
                                        : plot.column}</strong
                                ><span
                                    >{plot.rows.toLocaleString()} rows {plot.sampled
                                        ? '· first 50,000 source rows before filters'
                                        : 'after filters'} · applied settings · hover for values</span
                                >
                            </div>{:else}<p>Loading distribution…</p>{/if}
                    </section>
                {/if}
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
