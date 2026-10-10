<script>
    import { onMount } from 'svelte';
    import ModelWorkbench from './ModelWorkbench.svelte';
    import StackWorkbench from './StackWorkbench.svelte';
    import ProjectModelPlan from './ProjectModelPlan.svelte';
    import PlanVariables from './PlanVariables.svelte';
    import PlanFitting from './PlanFitting.svelte';
    import ExportPanel from './ExportPanel.svelte';
    import ProjectOpen from './ProjectOpen.svelte';
    import ExplorePanel from './ExplorePanel.svelte';
    import VariableScreening from './VariableScreening.svelte';
    import FeatureSelection from './FeatureSelection.svelte';
    import SplitSettings from './SplitSettings.svelte';
    import BinningSettings from './BinningSettings.svelte';
    let comparison = '',
        modelContext = { fitted: [], selected: '', champion: null };
    let legacySelected = '';
    let view = 'project',
        resultsReady = false;
    let stackMode = false;
    let workflowModeChosen = false;
    let stackDraftDirty = false,
        stackNavigationAttempt = 0,
        stackNavigationTarget = '';
    function requestStackNavigation(target) {
        stackNavigationTarget = target;
        stackNavigationAttempt += 1;
    }
    function resolveStackNavigation(target) {
        stackDraftDirty = false;
        if (target === '__single_model__') {
            stackMode = false;
            workflowModeChosen = true;
            return;
        }
        navigate(target);
    }
    function chooseModelWorkflow(useStack) {
        if (useStack === stackMode) return;
        if (stackMode && stackDraftDirty) {
            requestStackNavigation('__single_model__');
            return;
        }
        stackMode = useStack;
        workflowModeChosen = true;
    }
    function modelState(snapshot) {
        const keepDraft = dirty;
        const variablesChanged =
            snapshot.session_id !== state?.session_id ||
            snapshot.project_id !== state?.project_id ||
            snapshot.revision !== state?.revision ||
            JSON.stringify(snapshot.setup) !== JSON.stringify(state?.setup);
        state = snapshot;
        if (variablesChanged) preview = null;
        if (!keepDraft) {
            draft = structuredClone(snapshot.setup);
            jsonText = roleJson();
        }
    }
    const pageTitles = {
        project: 'Project & data',
        explore: 'Explore',
        model: 'Model',
        diagnostics: 'Diagnostics',
        compare: 'Compare',
        tables: 'Rate tables',
        export: 'Export',
    };
    let projectInfo = null,
        projectError = '';
    let planState = null,
        planStageId = '',
        planVariablesDirty = false,
        planRevisionIntent = false,
        showSharedSetup = false,
        requestedPlanModel = '',
        focusModelSetup = false,
        planRefreshEpoch = 0,
        choosingNewPlan = false,
        setupStep = 'data',
        singleVariableStep = 'selection',
        planFittingDirty = false,
        planVariablesReset = 0;
    $: fittedStageCount = planState?.fit_refs?.length || 0;
    $: plannedStageCount = planState?.plan?.stages?.length || 0;
    $: planFullyFitted = Boolean(
        planState?.plan && plannedStageCount && fittedStageCount >= plannedStageCount,
    );
    $: fittedProgressLabel = planState?.plan
        ? planFullyFitted
            ? 'Fitted'
            : `${fittedStageCount} of ${plannedStageCount} fitted`
        : resultsReady
          ? 'Fitted'
          : 'Not fitted';
    $: sharedSetupLocked = Boolean(planState?.fit_refs?.length);
    $: planHasSelectedFactors = Boolean(
        planState?.plan?.stages?.some((stage) => stage.factors?.length),
    );
    $: modelIsDefined = Boolean(state?.models?.length || planHasSelectedFactors);
    $: setupSteps = !planState?.plan
        ? [
              ['data', 'Data'],
              ['structure', 'Model structure'],
              ...(state?.models?.length ? [['shared', 'Shared columns']] : []),
          ]
        : planState.plan.mode === 'single'
          ? [
                ['data', 'Data'],
                ['structure', 'Model structure'],
                ['shared', 'Shared columns'],
                ['variables', 'Variables'],
                ['details', 'Model details'],
                ['fit', 'Fit'],
            ]
          : [
                ['data', 'Data'],
                ['structure', 'Model structure'],
                ['shared', 'Shared columns'],
                ['models', 'Models'],
                ['variables', 'Variables'],
                ['fit', 'Fit'],
            ];
    $: contextTarget = state?.setup?.assignments?.target
        ? state.setup.renames?.[state.setup.assignments.target] || state.setup.assignments.target
        : '';
    $: contextWeight = state?.setup?.assignments?.weight
        ? state.setup.renames?.[state.setup.assignments.weight] || state.setup.assignments.weight
        : '';
    $: responseContext = !contextTarget
        ? ''
        : !planState?.plan
          ? contextTarget
          : planState.plan.divide_target_by_weight && contextWeight
            ? `${contextTarget} / ${contextWeight}`
            : contextWeight
              ? `${contextTarget} · weight: ${contextWeight}`
              : contextTarget;
    $: splitContext = state?.setup?.assignments?.split
        ? `${state.setup.renames?.[state.setup.assignments.split] || state.setup.assignments.split}: ${state.setup.split?.train_value} is training`
        : state?.setup?.split?.mode === 'random'
          ? `${Math.round(100 * (state.setup.split.fraction ?? 0.7))}% training`
          : '';
    function setupStepStatusesFor(planSnapshot, stateSnapshot, hasUnsavedRoles, ready) {
        const plan = planSnapshot?.plan;
        const stages = plan?.stages || [];
        const fitted = planSnapshot?.fit_refs?.length || 0;
        const selected = stages.filter((stage) => stage.factors?.length).length;
        const fullyFitted = Boolean(plan && stages.length && fitted >= stages.length);
        const fitStatus = plan
            ? fullyFitted
                ? 'Fitted'
                : fitted
                  ? `${fitted} of ${stages.length} fitted`
                  : selected
                    ? 'Ready to fit'
                    : 'Needs attention'
            : ready
              ? 'Fitted'
              : selected
                ? 'Ready to fit'
                : 'Needs attention';
        return {
            data: stateSnapshot?.columns?.length ? 'Complete' : 'Needs attention',
            structure: plan ? 'Saved' : 'Needs attention',
            shared: hasUnsavedRoles
                ? 'Unsaved changes'
                : stateSnapshot?.setup?.assignments?.target
                  ? fitted
                    ? 'Locked after fit'
                    : 'Ready'
                  : 'Needs attention',
            models: stages.length ? 'Saved' : 'Needs attention',
            variables: !selected
                ? 'Needs attention'
                : selected === stages.length
                  ? 'Ready'
                  : `Partly configured (${selected} of ${stages.length})`,
            details: stages[0]?.kind ? 'Saved' : 'Needs attention',
            fit: fitStatus,
        };
    }
    $: setupStepStatuses = setupStepStatusesFor(planState, state, dirty, resultsReady);
    function firstIncompleteSetupStep() {
        const currentPlan = planState?.plan;
        if (!currentPlan) return state?.columns?.length ? 'structure' : 'data';
        if (!state?.setup?.assignments?.target) return 'shared';
        const stages = currentPlan.stages || [];
        if (currentPlan.mode === 'sequential' && !stages.length) return 'models';
        const firstBlank = stages.find((stage) => !stage.factors?.length);
        if (firstBlank) {
            planStageId = firstBlank.component_id;
            return 'variables';
        }
        if (currentPlan.mode === 'single') return 'details';
        return 'fit';
    }
    function continueFromSharedColumns() {
        if (dirty) {
            notice = 'Preview and apply shared column changes before continuing.';
            return;
        }
        selectSetupStep(planState?.plan?.mode === 'sequential' ? 'models' : 'variables');
    }
    async function refreshPlan() {
        if (!state?.columns?.length) {
            planState = null;
            return;
        }
        const generation = `${state.session_id}:${state.revision}`;
        const requestEpoch = ++planRefreshEpoch;
        try {
            const fresh = await api('model-plan');
            if (
                requestEpoch === planRefreshEpoch &&
                `${state?.session_id}:${state?.revision}` === generation
            )
                planChanged(fresh);
        } catch (e) {
            if (
                requestEpoch === planRefreshEpoch &&
                `${state?.session_id}:${state?.revision}` === generation
            ) {
                error = `Cannot read the current model plan. ${e.message}`;
            }
        }
    }
    function planChanged(next) {
        // A successful save supersedes any older initial/background plan read.
        planRefreshEpoch += 1;
        planState = next;
        const wasChoosingNewPlan = choosingNewPlan;
        choosingNewPlan = false;
        if (next?.revision != null && state) state = { ...state, revision: next.revision };
        if (next?.plan && (wasChoosingNewPlan || setupStep === 'structure')) {
            setupStep = 'shared';
            writeSetupLocation('shared', '', true);
        }
        const stages = next?.plan?.stages || [];
        if (!stages.some((stage) => stage.component_id === planStageId))
            planStageId = stages[0]?.component_id || '';
        const latest = next?.fit_refs?.at(-1)?.model_name;
        if (latest) {
            requestedPlanModel = latest;
            resultsReady = true;
        } else resultsReady = Boolean(!next?.plan && next?.history?.length);
        if (!(next?.fit_refs || []).length) planRevisionIntent = false;
    }
    async function reviseSharedAssignments() {
        if (!planState?.plan || busy) return;
        if (
            !window.confirm(
                'Create a revised plan with these shared assignments? The original model recipes remain in this project, but their fitted results no longer match these shared assignments. Every stage in the revised plan must be fitted again.',
            )
        )
            return;
        busy = true;
        error = '';
        try {
            planChanged(
                await api('model-plan', {
                    session_id: state.session_id,
                    revision: state.revision,
                    operation: 'start',
                    plan: structuredClone(planState.plan),
                }),
            );
            planRevisionIntent = false;
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function restoreSavedSelection() {
        if (planState?.plan) return;
        const latest = planState?.history?.at(-1);
        legacySelected =
            (latest && !latest.components?.length ? latest.name : '') ||
            state?.models?.at(-1) ||
            '';
        requestedPlanModel = latest?.components?.length ? latest.name : '';
    }
    function writeSetupLocation(next, stageId = planStageId, replace = false) {
        const url = new URL(window.location.href);
        url.searchParams.set('view', 'project');
        url.searchParams.set('setup', next);
        if (next === 'variables' && stageId) url.searchParams.set('stage', stageId);
        else url.searchParams.delete('stage');
        const method = replace ? 'replaceState' : 'pushState';
        window.history[method]({}, '', `${url.pathname}?${url.searchParams}${url.hash}`);
    }
    function selectSetupStep(next, stageId = '', fromHistory = false, skipVariableGuard = false) {
        const staysInSingleVariableWorkflow =
            planState?.plan?.mode === 'single' &&
            ['variables', 'details'].includes(setupStep) &&
            ['variables', 'details'].includes(next);
        const changesVariableStage =
            next === 'variables' && Boolean(stageId) && stageId !== planStageId;
        if (
            !skipVariableGuard &&
            planVariablesDirty &&
            (next !== setupStep || changesVariableStage) &&
            !staysInSingleVariableWorkflow
        ) {
            if (!window.confirm('Discard unsaved variable changes?')) {
                if (fromHistory) {
                    writeSetupLocation(setupStep, planStageId, true);
                }
                return false;
            }
            planVariablesDirty = false;
            planVariablesReset += 1;
        }
        if (next === 'fit') {
            void navigate('model');
            return true;
        }
        if (stageId) planStageId = stageId;
        if (next === 'details') singleVariableStep = 'details';
        if (next === 'variables' && planState?.plan?.mode === 'single')
            singleVariableStep = 'selection';
        setupStep = next;
        if (!fromHistory) writeSetupLocation(next, planStageId);
        requestAnimationFrame(() =>
            document.getElementById(`setup-${next}`)?.scrollIntoView({ block: 'start' }),
        );
    }
    function restoreSetupStepFromLocation() {
        const search = new URLSearchParams(window.location.search);
        const requested = search.get('setup');
        const requestedStageId = search.get('stage') || '';
        const stageId = planState?.plan?.stages?.some(
            (stage) => stage.component_id === requestedStageId,
        )
            ? requestedStageId
            : '';
        if (requested && setupSteps.some(([key]) => key === requested))
            selectSetupStep(requested, stageId, true);
    }
    async function navigate(next) {
        // Variable definitions now belong to Project & data. Keep old bookmarks and
        // links from Explore and Model working while taking the user to the setup.
        if (next === 'variables') {
            next = 'project';
            setupStep = planState?.plan ? 'variables' : 'structure';
            if (planState?.plan?.mode === 'single') singleVariableStep = 'selection';
            writeSetupLocation(setupStep, planState?.plan ? planStageId : '', true);
            focusModelSetup = true;
        }
        if (planFittingDirty && view === 'model' && next !== view) {
            if (!window.confirm('Leave with unsaved fit settings?')) return;
            planFittingDirty = false;
        }
        if (next === 'model' && dirty) {
            notice = 'Preview and apply shared column changes before opening Fit.';
            if (view === 'project') {
                setupStep = 'shared';
                writeSetupLocation('shared', '', true);
            }
            return false;
        }
        if (planVariablesDirty && next !== view && next !== 'project') {
            if (!window.confirm('Discard unsaved stage-variable changes?')) return;
            planVariablesDirty = false;
            planVariablesReset += 1;
        }
        if (stackMode && stackDraftDirty && next !== view) {
            requestStackNavigation(next);
            return;
        }
        if (
            !workflowModeChosen &&
            ['model', 'diagnostics', 'compare', 'tables', 'export'].includes(next) &&
            state?.columns.length
        ) {
            try {
                const workbench = await api('workbench');
                const models = Object.values(workbench.models || {});
                if (models.length && models.every((model) => model.components?.length))
                    stackMode = true;
                workflowModeChosen = true;
            } catch (e) {
                error = e.message;
            }
        }
        view = next;
        if (next === 'project' && focusModelSetup) {
            requestAnimationFrame(() => {
                document.getElementById(`setup-${setupStep}`)?.scrollIntoView({ block: 'start' });
                focusModelSetup = false;
            });
        }
        if (['project', 'variables', 'model'].includes(next)) await refreshPlan();
        if (next === 'project' || next === 'export') {
            try {
                projectInfo = await api('project');
                projectError = '';
            } catch (e) {
                projectError = e.message;
            }
        }
        return true;
    }
    function confirmDiscardStageDraft() {
        if (!planVariablesDirty) return true;
        if (!window.confirm('Discard unsaved model-variable changes?')) return false;
        planVariablesDirty = false;
        planVariablesReset += 1;
        return true;
    }
    function beginPlanRevision() {
        if (!planState?.fit_refs?.length) return;
        if (!confirmDiscardStageDraft()) return;
        planRevisionIntent = true;
    }
    function requestMobilePage(event) {
        const next = event.currentTarget.value;
        // Native selects change their displayed value before a guarded navigation
        // can be accepted. Put it back until `navigate` changes the actual view.
        event.currentTarget.value = view;
        navigate(next);
    }
    let state = null,
        draft = null,
        tab = 'table',
        query = '',
        roleFilter = 'all';
    let jsonText = '',
        error = '',
        binningError = '',
        binningReset = 0,
        notice = '',
        busy = false,
        reviewing = false,
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
        (Boolean(binningError) ||
            JSON.stringify(draft) !== JSON.stringify(state.setup) ||
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
    async function uploadCatalogue(file, body) {
        if (differentProject) throw new Error('Reconnect to the current project before uploading.');
        if (!token) await bootstrap();
        const query = new URLSearchParams({ ...body, filename: file.name });
        const response = await fetch('/api/catalogues/upload?' + query, {
            method: 'POST',
            cache: 'no-store',
            credentials: 'same-origin',
            headers: { 'Content-Type': 'application/octet-stream', 'X-EasyGLM-Token': token },
            body: file,
        });
        const data = await response.json();
        if (!response.ok) throw responseError(data, 'Could not read catalogue.');
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
                ...(!planState?.plan ? { binning: draft.binning } : {}),
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
            const unchangedJson = jsonText === roleJson();
            const obj = JSON.parse(jsonText);
            if (!obj || Array.isArray(obj) || typeof obj !== 'object')
                throw new Error('Use a JSON object with the ten roles.');
            const acceptedJsonKeys = planState?.plan ? roles : [...roles, 'binning'];
            if (Object.keys(obj).some((r) => !acceptedJsonKeys.includes(r)))
                throw new Error('Use the shared roles shown in the template.');
            let binning = draft.binning;
            if (Object.hasOwn(obj, 'binning')) {
                const entry = obj.binning;
                if (
                    !entry ||
                    Array.isArray(entry) ||
                    typeof entry !== 'object' ||
                    Object.keys(entry).some((key) => !['default_bins', 'overrides'].includes(key))
                )
                    throw new Error('binning needs default_bins and overrides.');
                const validCount = (value) => Number.isInteger(value) && value >= 2 && value <= 200;
                if (!validCount(entry.default_bins))
                    throw new Error('binning.default_bins must be a whole number from 2 to 200.');
                if (
                    !entry.overrides ||
                    Array.isArray(entry.overrides) ||
                    typeof entry.overrides !== 'object'
                )
                    throw new Error('binning.overrides must be an object.');
                const knownNames = new Set(state.columns.map((column) => column.name));
                for (const [name, setting] of Object.entries(entry.overrides)) {
                    if (!knownNames.has(name))
                        throw new Error(`Unknown binning source column ${JSON.stringify(name)}.`);
                    if (setting == null) continue;
                    if (!setting || Array.isArray(setting) || typeof setting !== 'object')
                        throw new Error(`${name}: binning override must be an object or null.`);
                    const keys = Object.keys(setting);
                    if (setting.method === 'quantile') {
                        if (
                            keys.some((key) => !['method', 'bins'].includes(key)) ||
                            !validCount(setting.bins)
                        )
                            throw new Error(
                                `${name}: quantile bins must be a whole number from 2 to 200.`,
                            );
                    } else if (setting.method === 'cuts') {
                        if (
                            keys.some((key) => !['method', 'cuts'].includes(key)) ||
                            !Array.isArray(setting.cuts) ||
                            setting.cuts.some(
                                (value, index) =>
                                    typeof value !== 'number' ||
                                    !Number.isFinite(value) ||
                                    (index > 0 && value <= setting.cuts[index - 1]),
                            )
                        )
                            throw new Error(
                                `${name}: cuts must be finite, strictly increasing numbers.`,
                            );
                    } else if (setting.method === 'integer') {
                        if (
                            keys.some((key) => !['method', 'fallback_bins'].includes(key)) ||
                            (setting.fallback_bins != null && !validCount(setting.fallback_bins))
                        )
                            throw new Error(
                                `${name}: integer fallback must be a whole number from 2 to 200.`,
                            );
                    } else throw new Error(`${name}: unknown binning method.`);
                }
                binning = {
                    default_bins: entry.default_bins,
                    overrides: Object.fromEntries(
                        Object.entries(entry.overrides).filter(([, value]) => value != null),
                    ),
                };
            }
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
            const nextDraft = { ...draft, assignments, roles: grouped, split, binning };
            if (!unchangedJson) preview = null;
            draft = nextDraft;
            error = '';
            binningError = '';
            return true;
        } catch (e) {
            error = e.message;
            preview = null;
            return false;
        }
    }
    function switchTab(next) {
        if (next === tab) return;
        if (tab === 'table' && binningError) {
            error = binningError;
            return;
        }
        if (tab === 'json' && !parseRoles()) return;
        if (next === 'json') jsonText = roleJson();
        tab = next;
        scrollTop = 0;
    }
    function reset() {
        draft = structuredClone(state.setup);
        binningReset++;
        binningError = '';
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
            workflowModeChosen = false;
            stackMode = false;
            planState = null;
            requestedPlanModel = '';
            resultsReady = false;
            showSharedSetup = false;
            planRevisionIntent = false;
            differentProject = false;
            reset();
            notice = '';
            await refreshPlan();
            setupStep = firstIncompleteSetupStep();
            projectInfo = null;
            projectError = '';
            try {
                projectInfo = await api('project');
            } catch (e) {
                projectError = e.message;
            }
            restoreSavedSelection();
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
        workflowModeChosen = false;
        stackMode = false;
        planState = null;
        differentProject = false;
        reset();
        notice = '';
        tab = 'table';
        query = '';
        roleFilter = 'all';
        scrollTop = 0;
        previewScroll = 0;
        comparison = '';
        legacySelected = '';
        requestedPlanModel = '';
        modelContext = { fitted: [], selected: '', champion: null };
        resultsReady = false;
        projectInfo = null;
        projectError = '';
        exploreRequest = { name: '', id: 0 };
        token = '';
        serverSession = opened.session_id;
        view = 'project';
        setupStep = opened.columns?.length ? 'structure' : 'data';
        writeSetupLocation(setupStep, '', true);
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
            await navigate('project');
            restoreSavedSelection();
        } catch (e) {
            error = 'Data loaded. Reconnect to continue. ' + e.message;
            throw new Error(error);
        } finally {
            busy = false;
        }
    }
    async function review() {
        if (tab === 'json' && !parseRoles()) return;
        if (binningError) {
            error = binningError;
            return;
        }
        busy = true;
        reviewing = true;
        error = '';
        const reviewed = JSON.stringify([
            state.session_id,
            state.project_id,
            state.revision,
            draft,
        ]);
        try {
            const result = await api('variables/preview', {
                session_id: state.session_id,
                revision: state.revision,
                setup: draft,
            });
            if (
                reviewed !==
                JSON.stringify([state.session_id, state.project_id, state.revision, draft])
            ) {
                preview = null;
                error = 'Settings changed while previewing. Preview changes again before applying.';
                return;
            }
            preview = result;
            previewScroll = 0;
        } catch (e) {
            error = e.message;
        } finally {
            reviewing = false;
            busy = false;
        }
    }
    function prepareScreening() {
        return !busy && !differentProject && !binningError && (tab !== 'json' || parseRoles());
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
    async function stageFeatureSelection(action, names, expectedContext) {
        const current = JSON.stringify([
            state?.session_id,
            state?.project_id,
            state?.revision,
            draft,
            jsonText,
        ]);
        if (busy || differentProject || binningError || current !== expectedContext)
            throw new Error('Settings changed. Run feature selection again before changing roles.');
        if (!['ignore', 'predictor'].includes(action)) throw new Error('Unknown role change.');
        const chosen = new Set(names);
        const eligible = new Set([
            ...(draft.roles.predictor || []),
            ...(draft.roles.unassigned || []),
        ]);
        if (!chosen.size || [...chosen].some((name) => !eligible.has(name)))
            throw new Error('Only checked predictor or unassigned candidates can change roles.');
        draft.roles.predictor = (draft.roles.predictor || []).filter((name) => !chosen.has(name));
        draft.roles.unassigned = (draft.roles.unassigned || []).filter((name) => !chosen.has(name));
        draft.roles[action] = [...new Set([...(draft.roles[action] || []), ...chosen])];
        touch();
        await review();
    }
    async function apply() {
        if (binningError) {
            error = binningError;
            return;
        }
        busy = true;
        error = '';
        try {
            state = await api('variables/apply', {
                session_id: state.session_id,
                revision: state.revision,
                setup: draft,
            });
            draft = structuredClone(state.setup);
            binningReset++;
            binningError = '';
            jsonText = roleJson();
            preview = null;
            notice = 'Settings applied to this session.';
            await refreshPlan();
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
            if (initialView === 'variables') {
                await navigate('variables');
                const setup = planState?.plan ? 'variables' : 'structure';
                setupStep = setup;
                window.history.replaceState(
                    {},
                    '',
                    `${window.location.pathname}?view=project&setup=${setup}`,
                );
            } else if (Object.hasOwn(pageTitles, initialView)) await navigate(initialView);
            restoreSetupStepFromLocation();
        } catch (e) {
            error = e.message;
        }
        const onPopState = () => {
            if (new URLSearchParams(window.location.search).get('view') === 'project')
                restoreSetupStepFromLocation();
        };
        window.addEventListener('popstate', onPopState);
        return () => window.removeEventListener('popstate', onPopState);
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
                >{state?.setup.assignments.target &&
                (state?.setup.roles.predictor.length ||
                    planHasSelectedFactors ||
                    (stackMode && (modelContext?.fitted?.length || modelContext?.selected)))
                    ? '✓'
                    : '○'} Target and predictors</span
            >
            <span>{modelIsDefined ? '✓' : '○'} Model defined</span>
            <span>{planFullyFitted || (!planState?.plan && resultsReady) ? '✓' : '○'} Model fitted{planState?.plan ? ` · ${fittedProgressLabel}` : ''}</span>
        </div>
        <div class="rail-bottom">
            <span class="status-dot"></span> Local session
            <div class="version">EasyGLM 0.464</div>
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
        <label class="mobile-page-selector"
            >Page
            <select aria-label="Page" value={view} onchange={requestMobilePage}>
                {#each Object.entries(pageTitles) as [key, title]}
                    <option
                        value={key}
                        disabled={!state ||
                            (['diagnostics', 'compare', 'tables'].includes(key) && !resultsReady)}
                        >{title}</option
                    >
                {/each}
            </select>
        </label>
        <main>
            {#if error && view === 'project'}<div class="message error" role="alert">
                    <span>{error}</span><button onclick={reconnect} disabled={busy}
                        >Reconnect</button
                    >
                    {#if draft}<button onclick={downloadDraft}>Download draft</button><button
                            onclick={reload}
                            disabled={busy}>Discard draft and reload</button
                        >{/if}
                </div>{/if}
            {#snippet sharedDataSetup()}
                <section
                    class="model-card shared-data-setup"
                    aria-label="Shared column assignments"
                >
                    <div class="heading">
                        <div>
                            <h2>Shared columns</h2>
                            <p>
                                Set the response, exposure or weight, ID and train/test split once.
                                Ignore removes a column from every model picker and search.
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
                    {#if notice}<div class="message" role="status">{notice}</div>{/if}
                    {#if sharedSetupLocked}
                        <div class="message">
                            Shared assignments are locked because this plan has fitted stages.
                            Create a revised plan before changing them. The original fitted versions
                            are kept as history, and every stage in the revised plan must be
                            refitted.
                            <button onclick={reviseSharedAssignments} disabled={busy}
                                >Revise shared assignments</button
                            >
                        </div>
                    {/if}
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
                        <fieldset disabled={busy || sharedSetupLocked} class="edit-fieldset">
                            <section class="editor-card">
                                <div class="toolbar">
                                    <div class="segmented">
                                        <button
                                            class:active={tab === 'table'}
                                            onclick={() => switchTab('table')}>Table</button
                                        ><button
                                            class:active={tab === 'json'}
                                            onclick={() => switchTab('json')}>Variables JSON</button
                                        >
                                    </div>
                                    <span class="edit-status"
                                        >{dirty ? 'Unapplied changes' : 'All changes applied'}</span
                                    >
                                    <div class="spacer"></div>
                                    <button onclick={reset} disabled={busy}>Reset</button><button
                                        class="primary"
                                        onclick={review}
                                        disabled={busy || differentProject || !!binningError}
                                        >Preview changes</button
                                    >
                                </div>
                                {#if reviewing}<p class="preview-feedback" role="status">
                                        Checking variable changes…
                                    </p>
                                {:else if preview?.changes.length === 0}<div
                                        class="preview-feedback"
                                        role="status"
                                    >
                                        No variable changes to apply. Your current settings are
                                        ready for a model.
                                        <button type="button" onclick={() => navigate('model')}
                                            >Continue to Model</button
                                        >
                                    </div>
                                {:else if preview?.changes.length}<p
                                        class="preview-feedback"
                                        role="status"
                                    >
                                        {preview.changes.length} changes ready to review.
                                    </p>{/if}
                                {#if preview?.changes.length}<section
                                        class="preview-card"
                                        id="variable-preview"
                                    >
                                        <div class="preview-title">
                                            <strong
                                                >{preview.changes.length} variable changes</strong
                                            ><button
                                                class="primary"
                                                disabled={busy ||
                                                    differentProject ||
                                                    !!binningError}
                                                onclick={apply}>Apply changes</button
                                            >
                                        </div>
                                        <div
                                            class="preview-list"
                                            onscroll={(e) =>
                                                (previewScroll = e.currentTarget.scrollTop)}
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
                                    <div
                                        class:legacy-variable-design={!planState?.plan}
                                        class="table-head grid-row shared-role-row"
                                    >
                                        <span>SOURCE COLUMN</span><span>DISPLAY NAME</span><span
                                            >ROLE</span
                                        >{#if !planState?.plan}<span>VARIABLE TYPE</span>{/if}<span
                                            >SOURCE TYPE</span
                                        >
                                    </div>
                                    <div
                                        class="virtual-table"
                                        role="region"
                                        aria-label="Variable settings"
                                        onscroll={(e) => (scrollTop = e.currentTarget.scrollTop)}
                                    >
                                        <div
                                            style:height={rowOffsets.at(-1) + 'px'}
                                            class:legacy-variable-design={!planState?.plan}
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
                                                            class:legacy-variable-design={!planState?.plan}
                                                            class="grid-row shared-role-row data-row"
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
                                                                aria-label={'Name for ' +
                                                                    column.name}
                                                                value={draft.renames[column.name] ||
                                                                    column.name}
                                                                onchange={(e) =>
                                                                    setName(
                                                                        column.name,
                                                                        e.currentTarget.value,
                                                                    )}
                                                            /><select
                                                                aria-label={'Role for ' +
                                                                    column.name}
                                                                value={getRole(
                                                                    column.name,
                                                                    roleMap,
                                                                )}
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
                                                            >{#if !planState?.plan}<select
                                                                    aria-label={'Type for ' +
                                                                        column.name}
                                                                    value={getType(
                                                                        column.name,
                                                                        typeMap,
                                                                    )}
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
                                                                >{/if}<span class="dtype"
                                                                >{column.dtype}</span
                                                            >
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
                                        Roles use source column names. Missing role columns become
                                        ignored. The split entry includes its column and training
                                        value; the other value is holdout. Variable types, bands and
                                        cuts are chosen separately for each model below.
                                    </div>
                                    <textarea
                                        class="json-editor"
                                        aria-label="Variables JSON"
                                        bind:value={jsonText}
                                        oninput={() => {
                                            preview = null;
                                            notice = '';
                                        }}
                                        spellcheck="false"></textarea>
                                {/if}
                                <div class="table-footer">
                                    <span
                                        >Roles determine eligibility. Model selections and
                                        interactions are configured separately.</span
                                    ><span>Revision {state.revision}</span>
                                </div>
                            </section>
                            {#if !planState?.plan && tab === 'table'}
                                <BinningSettings
                                    {api}
                                    {state}
                                    active={view === 'project'}
                                    setup={draft}
                                    disabled={busy || differentProject}
                                    resetKey={binningReset}
                                    onvalidity={(message) => {
                                        binningError = message;
                                        if (message) preview = null;
                                    }}
                                    onchange={(next) => {
                                        draft = next;
                                        touch();
                                    }}
                                />
                            {/if}
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
                        </fieldset>
                    {:else}<div class="loading">Opening local data…</div>{/if}
                </section>
            {/snippet}
            <section hidden={view !== 'project'} class="workflow-page project-page">
                <div class="heading">
                    <div>
                        <h1>Project & data</h1>
                        <p>Set up the data and model before fitting.</p>
                        {#if state?.columns.length}
                            <section class="loaded-dataset" aria-label="Current dataset">
                                <span>Loaded:</span>
                                <span class="source-filename" title={projectInfo?.data?.source?.path || ''}>
                                    {sourceFilename(projectInfo)}
                                </span>
                                <span>{countLabel(state.row_count, 'row')}</span>
                                <span>{countLabel(state.columns.length, 'column')}</span>
                                {#if responseContext}<span>{responseContext}</span>{/if}
                                {#if splitContext}<span>{splitContext}</span>{/if}
                            </section>
                        {/if}
                    </div>
                </div>
                {#if projectError}<p role="alert">{projectError}</p>{/if}
                {#if state?.columns.length}
                    <nav class="setup-step-nav" aria-label="Project setup steps">
                        {#each setupSteps as [key, label], index}
                            <button
                                class:active={setupStep === key}
                                aria-current={setupStep === key ? 'step' : undefined}
                                onclick={() => selectSetupStep(key)}
                            ><span>{index + 1}</span><span class="setup-step-label">{label}<small>{setupStepStatuses[key]}</small></span></button>
                        {/each}
                    </nav>
                {/if}
                {#if state}
                    <section id="setup-data" hidden={setupStep !== 'data'}>
                        <ProjectOpen
                            {api}
                            upload={uploadProject}
                            {state}
                            onOpened={projectOpened}
                            onReconnect={reconnectForOpen}
                        />
                    </section>
                {/if}
                {#if state?.columns.length}
                    <section id="setup-structure" hidden={setupStep !== 'structure'} aria-label="Model structure">
                        <ProjectModelPlan
                            {api}
                            {state}
                            {planState}
                            onChanged={planChanged}
                            onRevisionIntent={() => (planRevisionIntent = true)}
                            {confirmDiscardStageDraft}
                            showHistory={false}
                            routeChoiceOnly={Boolean(planState?.plan)}
                            onCancelNewPlan={() => {
                                choosingNewPlan = false;
                                setupStep = 'shared';
                                writeSetupLocation('shared', '', true);
                            }}
                        />
                    </section>
                    {#if !planState?.plan && state.models?.length}
                        <section class="model-card legacy-model-compat" aria-label="Existing saved models">
                            <div>
                                <h2>Existing saved model{state.models.length === 1 ? '' : 's'}</h2>
                                <p>These models predate model plans. You can still open and work with them.</p>
                            </div>
                            <button onclick={() => navigate('model')}>Open saved model</button>
                        </section>
                    {/if}
                    <section id="setup-shared" hidden={setupStep !== 'shared'}>
                        {@render sharedDataSetup()}
                        {#if !sharedSetupLocked}
                            <div class="setup-continue">
                                <button
                                    class="primary"
                                    disabled={busy || dirty || differentProject || !!binningError}
                                    onclick={continueFromSharedColumns}
                                >Continue to {planState?.plan?.mode === 'sequential' ? 'models' : 'variables'}</button>
                            </div>
                        {/if}
                    </section>
                    {#if planState?.plan}
                        {#if planState.plan.mode === 'sequential'}
                            <section id="setup-models" hidden={setupStep !== 'models'} aria-label="Models">
                                <ProjectModelPlan
                                    {api}
                                    {state}
                                    {planState}
                                    onChanged={planChanged}
                                    onRevisionIntent={() => (planRevisionIntent = true)}
                                    {confirmDiscardStageDraft}
                                    showHistory={false}
                                    modelsOnly={true}
                                    onNavigate={(_, stageId) => selectSetupStep('variables', stageId)}
                                />
                                <div class="setup-continue">
                                    <button class="primary" onclick={() => selectSetupStep('variables')}>Choose variables</button>
                                </div>
                            </section>
                        {/if}
                        <section id="setup-variables" hidden={setupStep !== 'variables' && setupStep !== 'details'} aria-label="Variables">
                            <PlanVariables
                                {api}
                                {state}
                                {planState}
                                {uploadCatalogue}
                                bind:selectedStageId={planStageId}
                                bind:singleStep={singleVariableStep}
                                revisionIntent={planRevisionIntent}
                                embedded={true}
                                onChanged={planChanged}
                                onDirty={(value) => (planVariablesDirty = value)}
                                onCreateRevision={beginPlanRevision}
                                onStageChange={(stageId) => selectSetupStep('variables', stageId, false, true)}
                                resetSignal={planVariablesReset}
                                onSingleStepChange={(step) => {
                                    if (planState?.plan?.mode === 'single') {
                                        setupStep = step === 'selection' ? 'variables' : 'details';
                                        writeSetupLocation(setupStep);
                                    }
                                }}
                                onSaved={(_, detail) => {
                                    if (planState?.plan?.mode === 'sequential') {
                                        const next = planState?.plan?.stages?.[detail.stageIndex + 1];
                                        if (next) selectSetupStep('variables', next.component_id);
                                        else void navigate('model');
                                    } else void navigate('model');
                                }}
                            />
                        </section>
                        <details
                            class="model-card saved-versions-history"
                            hidden={setupStep !== 'variables' && setupStep !== 'details'}
                        >
                            <summary>Saved fitted versions ({(planState?.history || []).length})</summary>
                            {#if (planState?.history || []).length}
                                <ul>{#each planState.history as fitted}<li>{fitted.name}</li>{/each}</ul>
                            {:else}<p>No fitted versions yet.</p>{/if}
                        </details>
                    {/if}
                {/if}
            </section>
            <section hidden={view !== 'explore'} class="workflow-page">
                {#if state && draft}{#key state.project_id}<ExplorePanel
                            {api}
                            {state}
                            active={view === 'explore'}
                            requestColumn={exploreRequest}
                            onNavigate={navigate}
                            onReconnect={reconnectForOpen}
                        />{/key}
                    <VariableScreening
                        {api}
                        {state}
                        setup={draft}
                        context={screeningContext}
                        active={view === 'explore'}
                        disabled={busy || differentProject}
                        prepare={prepareScreening}
                        onRemove={removeScreenedPredictors}
                    />
                    <FeatureSelection
                        {api}
                        {state}
                        setup={draft}
                        context={screeningContext}
                        disabled={busy || differentProject || !!binningError}
                        prepare={prepareScreening}
                        onChange={stageFeatureSelection}
                    />
                {/if}
            </section>
            {#if view === 'export' && ((!stackMode && !planState?.plan) || (planState?.plan?.mode === 'single' && planState.plan.stages[0]?.kind === 'glm'))}<ExportPanel
                    {api}
                    {state}
                    context={modelContext}
                    {comparison}
                    {downloadProject}
                    {saveDownload}
                />{/if}
            {#if state && !planState?.plan && view === 'model' && !state.models?.length}<section
                    class="model-card workflow-page"
                >
                    <h1>Define a model plan</h1>
                    <p>
                        Choose Single model or Build in stages in Project &amp; data before fitting.
                    </p>
                    <button class="primary" onclick={() => navigate('project')}
                        >Go to Project &amp; data</button
                    >
                </section>{/if}
            {#if state && planState?.plan && view === 'model'}<PlanFitting
                    {api}
                    {state}
                    {planState}
                    onChanged={planChanged}
                    onDirty={(value) => (planFittingDirty = value)}
                    onNavigate={(target, stageId) => {
                        if (target === 'variables' || target === 'models') {
                            void (async () => {
                                await navigate('project');
                                if (view !== 'project') return;
                                selectSetupStep(
                                    target === 'variables'
                                        ? 'variables'
                                        : planState?.plan?.mode === 'single'
                                          ? 'details'
                                          : 'models',
                                    stageId,
                                );
                            })();
                        } else if (target === 'project') void navigate('project');
                        else if (target === 'diagnostics') void navigate('diagnostics');
                    }}
                    onFitted={(name) => {
                        requestedPlanModel = name;
                        resultsReady = true;
                    }}
                />{/if}
            {#if state && draft && ((!planState?.plan && !stackMode && (view !== 'model' || state.models?.length)) || (planState?.plan?.mode === 'single' && planState.plan.stages[0]?.kind === 'glm' && ['diagnostics', 'compare', 'tables', 'export'].includes(view)))}{#key state.project_id}<ModelWorkbench
                        {api}
                        {state}
                        {view}
                        bind:comparison
                        onContext={(context) => (modelContext = context)}
                        onState={modelState}
                        onReady={(ready) =>
                            (resultsReady = ready || Boolean(planState?.fit_refs?.length))}
                        onNavigate={navigate}
                        requestedModel={planState?.plan ? requestedPlanModel : legacySelected}
                        initialSubset={planState?.plan ? 'train' : 'holdout'}
                        showBothDiagnosticSubsets={planState?.plan?.mode !== 'single'}
                        onRequestedModelConsumed={() => {
                            if (planState?.plan) requestedPlanModel = '';
                            else legacySelected = '';
                        }}
                    />{/key}{/if}
            {#if state && draft && ((!planState?.plan && stackMode) || (planState?.plan && !(planState.plan.mode === 'single' && planState.plan.stages[0]?.kind === 'glm') && ['diagnostics', 'compare', 'tables', 'export'].includes(view)))}{#key state.project_id}<StackWorkbench
                        {api}
                        {state}
                        {view}
                        {uploadCatalogue}
                        {saveDownload}
                        onState={modelState}
                        onReady={(ready) =>
                            (resultsReady = ready || Boolean(planState?.fit_refs?.length))}
                        onContext={(context) => (modelContext = context)}
                        onNavigate={navigate}
                        onDirty={(isDirty) => (stackDraftDirty = isDirty)}
                        navigationAttempt={stackNavigationAttempt}
                        navigationTarget={stackNavigationTarget}
                        onNavigationResolved={resolveStackNavigation}
                        requestedModel={requestedPlanModel}
                        onRequestedModelConsumed={() => (requestedPlanModel = '')}
                    />{/key}{/if}
        </main>
    </div>
</div>

<style>
    .preview-feedback {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 15px 4px;
        color: var(--text);
        font-size: 13px;
    }
    .preview-feedback button {
        margin-left: 4px;
    }
    .editor-card > .preview-card {
        margin: 12px 15px;
    }
    .loaded-dataset {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 4px 8px;
        margin: 12px 0 0;
        color: var(--muted);
        font-size: 14px;
    }
    .loaded-dataset span:first-child {
        color: var(--text);
        font-weight: 600;
    }
    .loaded-dataset span:nth-child(n + 3)::before {
        content: '·';
        color: #a5aea9;
        margin-right: 8px;
    }
    .source-filename {
        overflow-wrap: anywhere;
        min-width: 0;
    }
    .plan-route-summary,
    .legacy-model-compat {
        display: flex;
        align-items: start;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
    }
    .legacy-model-compat h2,
    .legacy-model-compat p {
        margin-top: 0;
    }
    .mobile-page-selector {
        display: none;
    }
    .setup-step-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 24px 0;
    }
    .setup-step-nav button {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        min-height: 46px;
        padding: 7px 12px;
        text-align: left;
    }
    .setup-step-label {
        display: grid;
        gap: 1px;
    }
    .setup-step-label small {
        color: var(--muted);
        font-size: 11px;
        font-weight: 400;
    }
    .setup-step-nav button > span:first-child {
        display: grid;
        width: 20px;
        height: 20px;
        place-items: center;
        border-radius: 999px;
        background: #e3ebe6;
        font-size: 12px;
        font-weight: 700;
    }
    .setup-step-nav button.active {
        border-color: #156347;
        background: #e6f2eb;
        color: #123e2d;
    }
    .setup-step-nav button.active > span:first-child {
        background: #156347;
        color: white;
    }
    .setup-continue {
        display: flex;
        justify-content: flex-end;
        margin: 16px 0 24px;
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
    .shared-data-setup .shared-role-row {
        grid-template-columns: minmax(135px, 1.2fr) minmax(140px, 1.3fr) minmax(125px, 1fr) 90px;
        min-width: 560px;
    }
    .shared-data-setup .shared-role-row.legacy-variable-design {
        grid-template-columns:
            minmax(135px, 1.1fr) minmax(140px, 1.2fr) minmax(125px, 1fr) minmax(140px, 1fr)
            90px;
        min-width: 700px;
    }
    .shared-data-setup .virtual-space {
        min-width: 560px;
    }
    .shared-data-setup .virtual-space.legacy-variable-design {
        min-width: 700px;
    }
    @media (max-width: 900px) {
        .random-split-fields {
            grid-template-columns: 1fr;
        }
    }
    @media (max-width: 760px) {
        .legacy-model-compat {
            display: block;
        }
        .legacy-model-compat button {
            margin-top: 16px;
        }
        .mobile-page-selector {
            display: grid;
            gap: 4px;
            padding: 10px 12px 0;
            color: var(--muted);
            font-size: 12px;
        }
        .mobile-page-selector select {
            width: 100%;
            min-width: 0;
        }
    }
</style>
