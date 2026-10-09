<script>
    import { onDestroy, tick } from 'svelte';
    import DiagnosticPlot from './DiagnosticPlot.svelte';
    import ModelFlow from './ModelFlow.svelte';
    import { nodeState } from './modelFlow.js';
    import { formatNumber as num } from './format.js';
    export let api, state, view, uploadCatalogue, saveDownload;
    export let onState = () => {}, onReady = () => {}, onContext = () => {}, onNavigate = () => {};
    export let onDirty = () => {}, navigationAttempt = 0, navigationTarget = '', onNavigationResolved = () => {};
    const labels = { glm: 'GLM', catboost: 'CatBoost', interaction: 'Pair interaction', catalogue: 'Catalogue lookup', legacy: 'Accepted legacy model' };
    function unseenSummary(levels) {
        return Object.entries(levels || {})
            .filter(([, detail]) => Number(detail?.count || 0) > 0)
            .map(([factor, detail]) => `${factor}: ${detail.count} (${(detail.examples || []).join(', ')})`)
            .join('; ');
    }
    const defaults = {
        glm: { alpha: 0.001, cv: null, n_alphas: 20, l1_ratio: 1, min_alpha_ratio: 0.01, base: 'modal' },
        catboost: { iterations: 500, depth: 6, learning_rate: 0.05, l2_leaf_reg: 3, seed: 42, thread_count: 4, time_limit_minutes: 15 },
        interaction: { candidates: [{ iterations: 300, depth: 4, learning_rate: 0.05, l2_leaf_reg: 3 }], min_weight_share: 0.0025, cv_folds: 3, seed: 42, time_limit_minutes: 15 },
        catalogue: { key: '', method: 'catboost', source_path: '', source_sha256: '', fallback: null, iterations: 500, depth: 6, learning_rate: 0.05, l2_leaf_reg: 3, seed: 42, thread_count: 4, time_limit_minutes: 15 },
    };
    let wb = null, jobs = {}, selected = '', source = '', result = null, kind = 'glm',
        stackFamily = 'poisson', tweediePower = 1.5, divideTargetByWeight = false,
        name = '', stageName = '', factors = [], query = '', showSelected = false, settings = structuredClone(defaults.glm),
        catalogue = null, error = '', notice = '', busy = '', timer = null, known = '',
        reading = false, subset = 'train', variable = '', diagnosticQuery = '', diagnostics = null, challenger = '',
        comparison = null, stage = '', table = null, preview = null,
        lower = '', upper = '', decimals = '', operation = 'cap', amendedName = '', refitLater = false, viewKey = '',
        editorOpen = false, editingExisting = '', editorDirty = false, pendingAction = null,
        selectedTab = 'results', requestEpoch = 0, modelError = '', modelNotice = '', draftBaseline = '', revisionOf = '', lastNavigationAttempt = 0,
        sourceFitId = '', refreshSuggested = false, fitDraftName = '', fitJobId = '';
    $: names = wb ? Object.keys(wb.models || {}).filter((n) => wb.models[n]?.components?.length) : [];
    $: fitted = names.filter((n) => jobs[n]?.applicable);
    $: modelFlow = wb?.model_flow || [];
    // Keep status rendering explicitly dependent on both server snapshots. Svelte
    // cannot see the reactive inputs hidden inside a template helper call.
    $: modelStatuses = Object.fromEntries(names.map((n) => [n, modelStatusLabel(n, jobs, modelFlow, fitDraftName, fitJobId)]));
    $: selectedFlow = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || []).find((entry) => entry.name === selected);
    $: selectedParent = selectedFlow?.parent_name || (!Array.isArray(modelFlow) ? (modelFlow?.edges || []).find((edge) => edge.to === selected)?.from : '');
    $: selectedStatus = modelStatusLabel(selected, jobs, modelFlow, fitDraftName, fitJobId);
    $: selectedNeedsRecovery = selectedStatus === 'Cannot score' || selectedStatus === 'Needs a new fit';
    $: currentDraftState = JSON.stringify({ source, name, stageName, kind, factors: [...factors].sort(), settings, stackFamily, tweediePower, divideTargetByWeight });
    $: if (editorOpen && draftBaseline) editorDirty = draftBaseline !== currentDraftState;
    $: onDirty(Boolean(editorOpen && editorDirty));
    $: if (navigationAttempt !== lastNavigationAttempt) {
        lastNavigationAttempt = navigationAttempt;
        if (editorOpen && editorDirty) pendingAction = () => onNavigationResolved(navigationTarget);
        else onNavigationResolved(navigationTarget);
    }
    $: fittedSources = wb ? Object.keys(wb.stack_sources || {}).filter((n) => jobs[n]?.applicable && wb.models[n]).sort((a, b) => a.localeCompare(b)) : [];
    $: sourceInfo = source ? wb?.stack_sources?.[source] : null;
    $: components = result?.components || wb?.models?.[selected]?.components || [];
    $: choices = wb ? [...new Set(
        (source && wb.stack_candidates?.[source]?.at(-1)) || wb.stack_base_candidates ||
        [...(wb.pair_candidates?.predictors || wb.predictors || []), ...(wb.pair_candidates?.unassigned || [])]
    )].sort((a, b) => a.localeCompare(b)) : [];
    $: reviewChoices = result?.review_variables || choices;
    $: metricDefinitions = diagnostics?.metric_definitions?.[subset] || result?.metric_definitions?.[subset] || [];
    $: filteredReviewChoices = reviewChoices.filter((n) => n === variable || n.toLowerCase().includes(diagnosticQuery.toLowerCase()));
    $: factorChoices = kind === 'catalogue' && catalogue
        ? catalogue.columns.map((c) => c.name || c).filter((n) => n !== settings.key).sort((a, b) => a.localeCompare(b))
        : choices;
    $: filtered = factorChoices.filter((n) => showSelected ? factors.includes(n) : n.toLowerCase().includes(query.toLowerCase()));
    $: onReady(Boolean(jobs[selected]?.applicable));
    $: onContext({ fitted, selected, champion: wb?.champion });
    $: if (state && !reading && known !== `${state.session_id}:${state.revision}`) refresh();
    $: if (result && ['diagnostics', 'compare', 'tables'].includes(view) && viewKey !== `${view}:${selected}:${subset}:${variable}:${challenger}:${stage}`) {
        viewKey = `${view}:${selected}:${subset}:${variable}:${challenger}:${stage}`;
        loadView();
    }
    function rev() { return { session_id: state.session_id, revision: state.revision }; }
    function changeKind(value) { kind = value; settings = structuredClone(defaults[value]); catalogue = null; }
    function draftState() { return JSON.stringify({ source, name, stageName, kind, factors: [...factors].sort(), settings, stackFamily, tweediePower, divideTargetByWeight }); }
    function setDraftBaseline() { draftBaseline = draftState(); editorDirty = false; }
    function toggleFactor(value) { factors = factors.includes(value) ? factors.filter((n) => n !== value) : [...factors, value]; }
    function parentFor(model) { return (Array.isArray(modelFlow) ? [] : modelFlow?.edges || []).find((edge) => edge.to === model)?.from || ''; }
    function flowNodeFor(model, flow = modelFlow) { return (Array.isArray(flow) ? flow : flow?.nodes || []).find((entry) => entry.name === model); }
    function modelStatusLabel(model, currentJobs = jobs, flow = modelFlow, starting = '', acceptedJobId = '') {
        const node = flowNodeFor(model, flow);
        const job = currentJobs[model] || {};
        // A retry can still have the prior failed job in this snapshot when the
        // accepted fit request returns. Keep the saved draft locked until the
        // worker publishes the new job state; poll clears fitDraftName only for
        // an actual terminal result from that new attempt.
        const staleTerminal = ['failed', 'cancelled', 'stale'].includes(job.status);
        const waitingForAcceptedJob = acceptedJobId && job.id !== acceptedJobId;
        const localAccepted = starting === model && (!job.status || staleTerminal || waitingForAcceptedJob);
        const state = nodeState({ ...node, job: localAccepted ? { ...job, status: 'accepted' } : job });
        return {
            fitting: 'Fitting',
            fitted: 'Fitted',
            'cannot score': 'Cannot score',
            'needs a new fit': 'Needs a new fit',
            draft: 'Draft',
        }[state] || 'Draft';
    }
    function modelListContext(model) {
        const node = flowNodeFor(model);
        const parent = parentFor(model);
        if (!parent) return 'Starts on its own';
        if (node?.parent_fit_available === false || node?.edge_status === 'missing') return `Original starting fit: ${parent} (unavailable)`;
        return `Built on ${parent}`;
    }
    async function refresh() {
        reading = true;
        try {
            wb = await api('workbench'); jobs = wb.jobs || {}; known = `${wb.session_id}:${wb.revision}`;
            if (!wb.models[selected]?.components?.length) selected = Object.keys(wb.models).find((n) => wb.models[n]?.components?.length) || '';
            if (source && !wb.stack_sources?.[source]) {
                modelError = `The fitted model ${source} is no longer available. This draft still starts from it and cannot be saved as a standalone model.`;
            }
            if (selected && jobs[selected]?.applicable) await loadResult();
            if (!timer) timer = setInterval(poll, 1000);
        } catch (e) { error = e.message; } finally { reading = false; }
    }
    async function poll() {
        if (reading || busy || !wb) return;
        try {
            const next = await api('jobs');
            const changed = jobs[selected]?.id !== next[selected]?.id || jobs[selected]?.status !== next[selected]?.status;
            jobs = next;
            if (changed && next[selected]?.applicable) {
                const snapshot = await api('variables'); onState(snapshot);
                wb = await api('workbench'); jobs = wb.jobs; known = `${wb.session_id}:${wb.revision}`;
                editorOpen = false; editingExisting = ''; revisionOf = ''; draftBaseline = ''; editorDirty = false;
                fitDraftName = ''; fitJobId = '';
                modelNotice = 'Fit complete.';
                await loadResult();
            } else if (changed && ['failed', 'cancelled', 'stale'].includes(next[selected]?.status)) {
                if (fitDraftName === selected && fitJobId && next[selected]?.id !== fitJobId) return;
                const message = next[selected]?.message || 'Fit did not complete.';
                if (fitDraftName === selected) {
                    editSelectedDraft();
                    fitDraftName = ''; fitJobId = '';
                }
                modelNotice = '';
                modelError = message;
            }
        } catch (e) { error = e.message; }
    }
    async function loadResult() {
        const model = selected;
        const epoch = ++requestEpoch;
        if (!jobs[model]?.applicable) { result = null; return; }
        const data = await api('results/' + encodeURIComponent(model));
        if (model !== selected || epoch !== requestEpoch) return;
        if (!data.stack) { modelError = 'Open this model in Single model.'; result = null; return; }
        modelError = '';
        result = data;
        const currentComponents = data.components || wb?.models?.[model]?.components || [];
        if (!currentComponents.some((c) => (c.id || c.component_id) === stage)) stage = currentComponents.at(-1)?.id || currentComponents.at(-1)?.component_id || '';
        await loadView(epoch);
    }
    async function chooseSelected(model = selected) {
        requestEpoch += 1;
        if (editorOpen && !editorDirty) { clearEditor(); editorOpen = false; }
        selected = model;
        result = null;
        clearTransientModelState();
        error = '';
        selectedTab = 'results';
        const selectedNode = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || [])
            .find((entry) => entry.name === model);
        const fitting = ['running', 'queued'].includes(jobs[model]?.status);
        if (model && !jobs[model]?.applicable && !fitting && !selectedNode?.full_refit_required && wb?.models?.[model]?.components?.length) {
            editSelectedDraft();
            return;
        }
        await loadResult();
    }
    function clearEditor() {
        source = ''; name = ''; stageName = ''; factors = []; query = ''; showSelected = false;
        kind = 'glm'; settings = structuredClone(defaults.glm); catalogue = null;
        editingExisting = ''; revisionOf = ''; sourceFitId = ''; draftBaseline = ''; editorDirty = false; fitJobId = '';
    }
    function addAfter(model) {
        if (editorDirty) { pendingAction = () => addAfter(model); return; }
        source = model; sourceFitId = model ? wb?.stack_sources?.[model]?.source_fit_id || '' : '';
        name = ''; stageName = ''; factors = []; query = ''; showSelected = false; catalogue = null;
        kind = 'glm'; settings = structuredClone(defaults.glm);
        notice = model ? `This model will use ${model}'s complete fitted score as its offset.` : 'This model starts on its own.';
        modelNotice = ''; modelError = ''; refreshSuggested = false; editorOpen = true; editingExisting = ''; revisionOf = ''; setDraftBaseline();
    }
    function createAlternative(model) {
        if (editorDirty) { pendingAction = () => createAlternative(model); return; }
        const item = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || []).find((entry) => entry.name === model) || {};
        const edge = (Array.isArray(modelFlow) ? [] : modelFlow?.edges || []).find((entry) => entry.to === model);
        const parent = edge?.from || item.parent_name || '';
        const last = (item.components || wb?.models?.[model]?.components || []).at(-1);
        const original = modelFor(model);
        source = parent; sourceFitId = parent ? wb?.stack_sources?.[parent]?.source_fit_id || '' : '';
        name = uniqueModelName(`${model} revised`); stageName = last?.name ? `${last.name} revised` : '';
        kind = last?.kind || 'glm';
        factors = [...(last?.factors || last?.inputs || [])];
        settings = structuredClone(last?.settings || defaults[kind]);
        stackFamily = original?.family || 'poisson';
        tweediePower = original?.tweedie_power ?? 1.5;
        divideTargetByWeight = Boolean(original?.divide_target_by_weight);
        catalogue = null; restoreSavedCatalogue(last); query = ''; showSelected = false;
        notice = parent
            ? `New version starts from ${parent}. Its last addition was copied for editing.`
            : 'New standalone version copied its inputs for editing.';
        modelNotice = ''; modelError = ''; refreshSuggested = false; editorOpen = true; editingExisting = ''; revisionOf = model; setDraftBaseline();
    }
    function uniqueModelName(base) {
        let proposed = base, number = 2;
        while (wb?.models?.[proposed]) { proposed = `${base} ${number}`; number += 1; }
        return proposed;
    }
    function restoreSavedCatalogue(component) {
        if (component?.kind !== 'catalogue' || !component.settings?.source_path) return;
        const columns = [...new Set([component.settings.key, ...(component.factors || component.inputs || [])].filter(Boolean))]
            .map((name) => ({ name, type: 'Saved catalogue field' }));
        catalogue = {
            filename: component.settings.source_path.split('/').at(-1) || 'Saved catalogue',
            rows: 'saved', columns,
            source_path: component.settings.source_path,
            source_sha256: component.settings.source_sha256,
        };
    }
    function modelFor(name) { return wb?.models?.[name] || null; }
    function lastComponent(name) { return modelFor(name)?.components?.at(-1) || null; }
    function editSelectedDraft() {
        const model = modelFor(selected);
        const last = lastComponent(selected);
        if (!model || !last) return;
        source = model.lineage?.parent_name || ''; sourceFitId = model.lineage?.parent_fit_id || '';
        name = selected; stageName = last.name || ''; kind = last.kind || 'glm';
        factors = [...(last.factors || last.inputs || [])];
        settings = structuredClone(last.settings || defaults[kind]);
        catalogue = null; restoreSavedCatalogue(last); query = ''; showSelected = false;
        stackFamily = model.family || 'poisson'; tweediePower = model.tweedie_power ?? 1.5;
        divideTargetByWeight = Boolean(model.divide_target_by_weight);
        editingExisting = selected; revisionOf = ''; editorOpen = true;
        modelError = ['failed', 'cancelled', 'stale'].includes(jobs[selected]?.status)
            ? jobs[selected]?.message || 'Fit did not complete.'
            : '';
        refreshSuggested = false; setDraftBaseline();
    }
    function selectedAction(action) {
        if (editorDirty) { pendingAction = action; return; }
        action();
    }
    function selectModel(model) {
        selectedAction(() => chooseSelected(model));
    }
    function requestMobileModel(event) {
        const model = event.currentTarget.value;
        // Keep the native control in sync while the dirty-draft dialog is deciding.
        event.currentTarget.value = selected;
        selectModel(model);
    }
    function startStandalone() { selectedAction(() => addAfter('')); }
    function addAfterSelected() { selectedAction(() => addAfter(selected)); }
    function reviseSelected() { selectedAction(() => createAlternative(selected)); }
    function discardDraft() {
        editorDirty = false;
        if (editingExisting) { editorOpen = false; }
        else { clearEditor(); editorOpen = false; }
        clearTransientModelState();
    }
    function clearTransientModelState() {
        notice = ''; modelNotice = ''; modelError = ''; refreshSuggested = false;
        diagnostics = null; comparison = null; table = null; viewKey = '';
    }
    function resolvePending(choice) {
        const action = pendingAction; pendingAction = null;
        if (choice === 'keep') return;
        if (choice === 'save') { save().then(() => { if (!modelError && action) action(); }); return; }
        if (choice === 'discard') { discardDraft(); if (action) action(); }
    }
    async function loadView(epoch = requestEpoch) {
        if (!result) return;
        const model = selected;
        try {
            let data = null;
            if (view === 'diagnostics') data = await api(`stack/${encodeURIComponent(model)}/diagnostics?subset=${subset}${variable ? '&variable=' + encodeURIComponent(variable) : ''}`);
            if (view === 'compare' && challenger) data = await api(`stack/${encodeURIComponent(model)}/compare?subset=${subset}&challenger=${encodeURIComponent(challenger)}`);
            if (view === 'tables' && stage) data = await api(`stack/${encodeURIComponent(model)}/component/${encodeURIComponent(stage)}/table`);
            if (epoch !== requestEpoch || model !== selected) return;
            if (view === 'diagnostics') diagnostics = data;
            if (view === 'compare' && challenger) comparison = data;
            if (view === 'tables' && stage) table = data;
            modelError = '';
        } catch (e) { if (epoch === requestEpoch && model === selected) modelError = e.message; }
    }
    async function upload(event) {
        const file = event.currentTarget.files?.[0]; if (!file) return;
        busy = 'upload'; error = '';
        try { catalogue = await uploadCatalogue(file, rev()); settings = { ...settings, source_path: catalogue.source_path, source_sha256: catalogue.source_sha256 }; }
        catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function save(fitAfter = false) {
        busy = 'save'; modelError = ''; modelNotice = ''; refreshSuggested = false;
        const operationEpoch = ++requestEpoch;
        try {
            if (!name.trim() || (!editingExisting && wb.models[name])) throw new Error('Enter a new model name. Existing fitted models stay unchanged.');
            if (!stageName.trim()) throw new Error('Name this model component.');
            if (!factors.length || (kind === 'interaction' && factors.length !== 2)) throw new Error(kind === 'interaction' ? 'Choose exactly two factors.' : 'Choose at least one factor.');
            if (source && !jobs[source]?.applicable) throw new Error('Fit the model this one starts from first.');
            if (source && (!sourceFitId || !sourceInfo?.components?.length)) throw new Error('Refresh the fitted model before adding to it.');
            if (kind === 'catalogue') {
                if (!catalogue?.source_path || !settings.key) throw new Error('Upload a full catalogue and choose its unique key.');
                const available = new Set(catalogue.columns.map((c) => c.name || c));
                if (factors.includes(settings.key) || factors.some((f) => !available.has(f))) throw new Error('Choose catalogue attributes present in the file, excluding its key.');
            }
            const base = source ? wb.models[source] : { family: stackFamily, link: 'log', target: wb.target, weight: wb.weight, offset: wb.offset, divide_target_by_weight: divideTargetByWeight, tweedie_power: tweediePower, base: 'modal', predictors: [], interactions: [], pair_stages: [], penalty: { alpha: 0.001, cv: null, n_alphas: 20, l1_ratio: 1 } };
            const component = { component_id: crypto.randomUUID(), name: stageName.trim(), kind, factors: [...factors], settings: structuredClone(settings), amendments: [] };
            const fields = Object.fromEntries([
                'family', 'link', 'target', 'weight', 'offset', 'divide_target_by_weight',
                'predictors', 'penalty', 'tweedie_power', 'base', 'pair_method',
                'pair_time_limit_minutes',
            ].filter((key) => key in base).map((key) => [key, base[key]]));
            if (fields.penalty) fields.penalty = Object.fromEntries(
                ['alpha', 'cv', 'n_alphas', 'l1_ratio']
                    .filter((key) => key in fields.penalty)
                    .map((key) => [key, fields.penalty[key]])
            );
            const frozenPrefix = editingExisting
                ? (modelFor(editingExisting)?.components || []).slice(0, -1)
                : (source ? sourceInfo.components : []);
            const snapshot = await api('models/save', { ...rev(), name: name.trim(), create: !editingExisting,
                ...(source && !editingExisting ? { source_name: source, source_fit_id: sourceFitId } : {}),
                fields: { ...fields, components: [...frozenPrefix, component], interactions: [], pair_stages: [] } });
            onState(snapshot);
            if (operationEpoch !== requestEpoch) return;
            selected = name.trim(); result = null; editorDirty = false;
            if (fitAfter) { editingExisting = selected; editorOpen = true; setDraftBaseline(); }
            else { editorOpen = false; editingExisting = ''; draftBaseline = ''; }
            await refresh();
            if (selected !== name.trim()) return;
            modelNotice = fitAfter ? '' : 'Draft saved.';
            if (fitAfter) await fit(false, selected);
        } catch (e) { if (operationEpoch === requestEpoch) { modelError = e.message; refreshSuggested = e.status === 409 || /revision/i.test(e.message); } } finally { busy = ''; }
    }
    async function fit(fullRefit = false, target = selected) {
        busy = 'fit'; modelError = ''; modelNotice = '';
        const operationEpoch = ++requestEpoch;
        const jobBeforeRequest = structuredClone(jobs[target] || {});
        try {
            const started = await api(`models/${encodeURIComponent(target)}/fit`, { ...rev(), ...(fullRefit ? { full_refit: true } : {}) });
            if (operationEpoch === requestEpoch && target === selected) {
                // A successful request has saved this draft and started its fit. Lock it
                // before waiting for the worker to publish its first job status.
                fitDraftName = target;
                fitJobId = started?.id || '';
                editorDirty = false;
                setDraftBaseline();
                editorOpen = false;
                editingExisting = target;
                revisionOf = '';
                modelNotice = fullRefit ? 'Full-model rebuild started.' : 'Fit started.';
            }
            jobs = await api('jobs');
            if (operationEpoch === requestEpoch && target === selected) {
                const job = jobs[target] || {};
                const terminal = ['failed', 'cancelled', 'stale'].includes(job.status);
                const staleTerminalSnapshot = terminal && (
                    (fitJobId && job.id !== fitJobId)
                    || (!fitJobId
                        && ['failed', 'cancelled', 'stale'].includes(jobBeforeRequest.status)
                        && job.id === jobBeforeRequest.id)
                );
                if (terminal && !staleTerminalSnapshot) {
                    editSelectedDraft();
                    fitDraftName = ''; fitJobId = '';
                    modelNotice = '';
                    modelError = job.message || 'Fit did not complete.';
                    return;
                }
                if (job.applicable) {
                    editorOpen = false;
                    editingExisting = '';
                    fitDraftName = ''; fitJobId = '';
                    modelNotice = 'Fit complete.';
                    await loadResult();
                }
            }
        }
        catch (e) {
            if (operationEpoch === requestEpoch && target === selected) {
                modelNotice = '';
                modelError = e.message;
            }
        } finally { busy = ''; }
    }
    async function createFullCopy() {
        const original = modelFor(selected);
        if (!original) return;
        const copiedName = uniqueModelName(`${selected} rebuilt`);
        const operationEpoch = ++requestEpoch;
        busy = 'copy'; modelError = ''; modelNotice = '';
        try {
            const fields = Object.fromEntries([
                'family', 'link', 'target', 'weight', 'offset', 'divide_target_by_weight',
                'predictors', 'penalty', 'tweedie_power', 'base', 'pair_method',
                'pair_time_limit_minutes',
            ].filter((key) => key in original).map((key) => [key, structuredClone(original[key])]));
            if (fields.penalty) fields.penalty = Object.fromEntries(
                ['alpha', 'cv', 'n_alphas', 'l1_ratio'].filter((key) => key in fields.penalty).map((key) => [key, fields.penalty[key]])
            );
            const snapshot = await api('models/save', {
                ...rev(), name: copiedName, create: true,
                fields: { ...fields, components: structuredClone(original.components || []), interactions: [], pair_stages: [] },
            });
            onState(snapshot);
            if (operationEpoch !== requestEpoch) return;
            selected = copiedName; result = null; await refresh();
            if (selected === copiedName) await fit(false, copiedName);
        } catch (e) { if (operationEpoch === requestEpoch) modelError = e.message; } finally { busy = ''; }
    }
    async function refreshAppliedSettings() {
        busy = 'refresh';
        try {
            const snapshot = await api('variables');
            onState(snapshot); await tick();
            wb = await api('workbench'); jobs = wb.jobs || {}; known = `${wb.session_id}:${wb.revision}`;
            const currentSourceFit = source ? wb.stack_sources?.[source]?.source_fit_id : '';
            if (source && sourceFitId && currentSourceFit !== sourceFitId) {
                modelError = `The fitted model ${source} changed while this draft was open. Your draft is still here; choose Add model after this again to use its new fit.`;
                refreshSuggested = false;
            } else { modelError = ''; refreshSuggested = false; }
        } catch (e) { modelError = e.message; } finally { busy = ''; }
    }
    async function cancel() {
        try {
            jobs = await api(`models/${encodeURIComponent(selected)}/cancel`, rev());
            const job = jobs[selected] || {};
            if (fitDraftName === selected && ['failed', 'cancelled', 'stale'].includes(job.status)
                && (!fitJobId || job.id === fitJobId)) {
                editSelectedDraft();
                fitDraftName = ''; fitJobId = '';
                modelNotice = '';
                modelError = job.message || 'Fit was cancelled.';
            }
        } catch (e) { error = e.message; }
    }
    async function editPreview() {
        busy = 'preview'; error = ''; preview = null;
        try { preview = await api(`stack/${encodeURIComponent(selected)}/component/${encodeURIComponent(stage)}/preview`, { ...rev(), operation, lower: lower === '' ? null : Number(lower), upper: upper === '' ? null : Number(upper), decimals: decimals === '' ? null : Number(decimals), refit_later: refitLater }); }
        catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function editApply() {
        busy = 'apply'; error = '';
        try {
            const response = await api(`stack/${encodeURIComponent(selected)}/component/${encodeURIComponent(stage)}/apply`, { ...rev(), operation, lower: lower === '' ? null : Number(lower), upper: upper === '' ? null : Number(upper), decimals: decimals === '' ? null : Number(decimals), new_name: amendedName.trim(), refit_later: refitLater });
            onState(response.snapshot || response); selected = amendedName.trim(); preview = null; await refresh(); notice = 'Amended model saved as a new named version.';
        } catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function download(format, id = '') {
        busy = format; error = '';
        try {
            const path = id ? `stack/${encodeURIComponent(selected)}/export/${encodeURIComponent(id)}` : `exports/${encodeURIComponent(selected)}`;
            const file = await api(path, id ? undefined : { ...rev(), format }, true); saveDownload(file.blob, file.filename);
        } catch (e) { error = e.message; } finally { busy = ''; }
    }
    onDestroy(() => { if (timer) clearInterval(timer); });
</script>

<div hidden={!['model', 'diagnostics', 'compare', 'tables', 'export'].includes(view)} class="stack-workbench">
    <div class="heading"><div><div class="eyebrow">MODEL BUILDER</div><h1>{view === 'model' ? 'Build a model in stages' : view === 'tables' ? 'Component tables' : view === 'export' ? 'Export stacked model' : view === 'compare' ? 'Compare models' : 'Full-model diagnostics'}</h1><p>Start a model on its own, or add one after a fitted model. Each fitted model is kept as a named version.</p></div></div>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if notice}<div class="message" role="status">{notice}</div>{/if}
    {#if modelError}<div class="message error" role="alert">{modelError}{#if refreshSuggested}<button onclick={refreshAppliedSettings} disabled={!!busy}>Refresh applied settings</button>{/if}</div>{/if}
    {#if modelNotice}<div class="message" role="status">{modelNotice}</div>{/if}
    {#if !wb}<p>Loading model settings…</p>{:else}
        {#if view === 'model'}
            <div class="model-builder">
                <aside class="model-list" aria-label="Models">
                    <div class="model-list-heading"><div><div class="eyebrow">MODELS</div><h2>Models</h2></div><button onclick={startStandalone}>New standalone model</button></div>
                    <label class="mobile-model-select">Model<select aria-label="Model" value={selected} onchange={requestMobileModel}><option value="">Choose a model</option>{#each names as n}<option value={n}>{n}</option>{/each}</select></label>
                    <div class="model-list-items">{#each names as n}
                        {@const node = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || []).find((entry) => entry.name === n)}
                        {@const last = wb.models[n]?.components?.at(-1)}
                        <button class:active={selected === n} class="model-list-item" onclick={() => selectModel(n)}><strong>{n}</strong><span>{modelStatuses[n]}</span><small>{modelListContext(n)} · {last?.name || 'Model'}</small></button>
                    {/each}</div>
                    {#if !names.length}<p>No models yet. Start with a standalone model.</p>{/if}
                    <details class="model-map"><summary>Show model map</summary><ModelFlow {modelFlow} models={wb.models || {}} {jobs} {selected} onSelect={selectModel} onAdd={addAfter} onAlternative={createAlternative} /></details>
                </aside>
                <div class="selected-workspace">
                    {#if !selected && !editorOpen}<section class="model-card empty-workspace"><h2>Start a model</h2><p>Create a standalone model, or select an existing fitted model and add the next model after it.</p><button onclick={startStandalone}>New standalone model</button></section>{/if}
                    {#if selected && !editorOpen}<section class="model-card selected-model-header">
                        <div class="selected-model-identity"><div><div class="eyebrow">SELECTED MODEL</div><h2>{selected}</h2></div><span class:fit-state={selectedStatus === 'Fitted'} class="model-state">{selectedStatus}</span></div>
                        <div class:recovery={selectedNeedsRecovery} class="selected-model-context">
                            {#if selectedStatus === 'Fitted'}
                                <p>This model is fitted. Its score includes {components.map((c) => c.name).join(' and ')}.</p>
                                {#if selectedParent}<p class="help-text">{selectedFlow?.revise_available === false ? `It was originally built on an earlier ${selectedParent} fit. That exact fit is no longer available for revising this last addition.` : `Built on ${selectedParent}. Its complete fitted score is the offset for this model.`}</p>{:else}<p class="help-text">This model starts on its own.</p>{/if}
                            {:else if selectedStatus === 'Cannot score'}
                                <p>This saved model cannot be scored because its fitted files are unavailable.</p>
                                <p class="help-text">The saved recipe is still available below. It is read-only so the original model stays unchanged.</p>
                            {:else if selectedStatus === 'Needs a new fit'}
                                <p>This saved setup uses an earlier {selectedParent || 'starting'} fit as its offset. That exact fit is no longer available.</p>
                                <p class="help-text">Original starting fit: {selectedParent || 'not recorded'} (unavailable).</p>
                            {:else if selectedStatus === 'Fitting'}
                                <p>This model has been saved and is fitting now.</p>
                            {:else}
                                <p>This is a saved draft. It has not changed any fitted model.</p>
                                {#if selectedParent}<p class="help-text">It will use {selectedParent}'s complete fitted score as its offset.</p>{:else}<p class="help-text">This model starts on its own.</p>{/if}
                            {/if}
                        </div>
                        <div class="workspace-actions selected-model-actions">
                            {#if selectedStatus === 'Fitted'}<button onclick={addAfterSelected} disabled={selectedFlow?.append_available === false}>Add model after this</button><button onclick={reviseSelected} disabled={selectedFlow?.revise_available === false}>Revise model</button>{:else if selectedStatus === 'Fitting'}<span class="fit-progress" role="status">{['running', 'queued'].includes(jobs[selected]?.status) ? `Fitting: ${jobs[selected]?.message || jobs[selected]?.status}` : 'Starting fit…'}</span>{#if ['running', 'queued'].includes(jobs[selected]?.status)}<button onclick={cancel}>Cancel fit</button>{/if}{:else if selectedNeedsRecovery}<button class="primary" onclick={createFullCopy} disabled={!!busy}>Fit saved recipe as new model</button>{:else}<button class="primary" onclick={() => fit(false)} disabled={!!busy}>Fit model</button><button onclick={editSelectedDraft}>Edit draft</button>{/if}
                        </div>
                        {#if selectedNeedsRecovery}<p class="recovery-consequence">The new model fits {components.map((component) => component.name).join(' → ') || 'the saved components'} from the beginning. {selected} stays unchanged.</p>{/if}
                        {#if selectedStatus === 'Fitted' && selectedFlow?.revise_available === false}<p class="help-text">The model can still be used as the offset for a new model, but its original parent fit is unavailable, so this last addition cannot be revised.</p>{/if}
                    </section>
                    {#if selectedNeedsRecovery}<section class="model-card saved-recipe"><h2>Saved recipe</h2><p>The saved models and variables are shown in fitting order. This setup is read-only.</p><ol class="stack-components">{#each components as c, i}<li><strong>{c.name}</strong><span>{labels[c.kind]}</span><span>{(c.factors || c.inputs || []).join(', ') || 'No selected variables'}</span><span>{i ? `After ${components[i - 1].name}` : 'Starting model'}</span><span>Saved</span></li>{/each}</ol></section>{/if}
                    {#if jobs[selected]?.applicable}<section class="model-card selected-results"><div class="result-tabs"><button class:tab-active={selectedTab === 'results'} onclick={() => (selectedTab = 'results')}>Results</button><button class:tab-active={selectedTab === 'settings'} onclick={() => (selectedTab = 'settings')}>Settings</button><button onclick={() => onNavigate('compare')}>Compare</button><button onclick={() => onNavigate('export')}>Export</button></div>
                        {#if selectedTab === 'results'}<h2>Results</h2><p>{result?.summary?.components || components.length} model components · {result?.summary?.family || ''} · training A/E {result?.summary?.train_ae == null ? '—' : Number(result.summary.train_ae).toFixed(3)}</p><p class="help-text">Training results are shown here. Choose a holdout view from Diagnostics when you want an independent evaluation.</p><div class="stack-review-actions"><button onclick={() => onNavigate('diagnostics')}>Inspect actual versus expected</button><button onclick={() => onNavigate('tables')}>Inspect and amend tables</button></div>{:else}<h2>Settings</h2><p>These settings are fixed for this fitted model. Use Revise model to create a new version.</p><ol class="stack-components">{#each components as c, i}<li><strong>{c.name}</strong><span>{labels[c.kind]}</span><span>{(c.factors || c.inputs || []).join(', ')}</span><span>{i ? `After ${components[i - 1].name}` : 'Starting model'}</span><span>Fitted</span></li>{/each}</ol>{/if}
                    </section>{/if}
                    {/if}
        {#if editorOpen}<section class="model-card model-editor"><div class="stack-form-heading"><div><div class="eyebrow">DRAFT MODEL</div><h2>{revisionOf ? `Revise ${revisionOf}` : source ? 'Add model after this' : editingExisting ? 'Edit draft model' : 'New standalone model'}</h2><p>{source ? `This model is fitted after ${source}. ${source}'s complete fitted score is used as the offset.` : 'This model starts on its own.'}</p></div><button onclick={() => { discardDraft(); }}>Discard changes</button></div>
            <div class="form-grid"><label>Model name<input aria-label="Stacked model name" bind:value={name} /></label>
                <label>Component type<select aria-label="Model component type" value={kind} onchange={(e) => changeKind(e.currentTarget.value)}>{#each Object.entries(labels).filter(([key]) => key !== 'legacy') as [key,label]}<option value={key}>{label}</option>{/each}</select></label>
                <label>Component name<input aria-label="Model component name" bind:value={stageName} /></label></div>
            <div class="editor-actions editor-actions-top"><button onclick={() => save(false)} disabled={!!busy}>Save draft</button><button class="primary" onclick={() => save(true)} disabled={!!busy}>Fit model</button></div>
            {#if sourceInfo}<p class="help-text">Builds on <strong>{source}</strong> · complete model predictions as offset. Its complete fitted chain is kept: {sourceInfo.components.map((item) => item.name).join(' → ')}.</p>{/if}
            <div class="form-grid"><label>Model family{#if source}<input aria-label="Stack model family" value={wb.models[source]?.family || ''} readonly />{:else}<select aria-label="Stack model family" bind:value={stackFamily}><option value="poisson">Poisson</option><option value="tweedie">Tweedie</option><option value="gaussian">Gaussian</option></select>{/if}</label>
                <label>Link<input aria-label="Stack model link" value="Log" readonly /></label>
                <label>Target<input aria-label="Stack model target" value={source ? wb.models[source]?.target || '' : wb.target || ''} readonly /></label>
                <label>Weight or exposure<input aria-label="Stack model weight" value={source ? wb.models[source]?.weight || '' : wb.weight || ''} readonly /></label>
                <label>External offset<input aria-label="Stack model offset" value={source ? wb.models[source]?.offset || '' : wb.offset || ''} readonly /></label>
                {#if source && wb.models[source]?.family === 'tweedie'}<label>Tweedie power<input aria-label="Stack Tweedie power" value={wb.models[source]?.tweedie_power ?? ''} readonly /></label>{:else if !source && stackFamily === 'tweedie'}<label>Tweedie power<input aria-label="Stack Tweedie power" type="number" min="1.01" max="1.99" step=".05" bind:value={tweediePower} /></label>{/if}
            </div>
            <label class="stack-divide"><input aria-label="Divide target by weight" type="checkbox" checked={source ? Boolean(wb.models[source]?.divide_target_by_weight) : divideTargetByWeight} disabled={!!source} onchange={(e) => (divideTargetByWeight = e.currentTarget.checked)} />Divide target by weight when fitting unit outcomes</label>
            <h3>{source ? 'Variables for this addition' : 'Variables for this model'}</h3><p class="help-text">{kind === 'catalogue' ? 'Upload a catalogue below, then choose its required attributes.' : kind === 'catboost' ? 'CatBoost uses raw columns. GLM uses the shared bands from Variables.' : 'GLM uses these selected variables with the project’s shared bands.'}</p>
            {#if source}<p class="help-text">Already included in the offset: {sourceInfo?.components?.flatMap((c) => c.factors || c.inputs || []).join(', ') || 'the earlier model'}.</p>{/if}
            <div class="selected-factors"><strong>Selected ({factors.length})</strong>{#if factors.length}<div>{#each factors.slice().sort((a, b) => a.localeCompare(b)) as factor}<button class="selected-factor" onclick={() => toggleFactor(factor)} aria-label={`Remove ${factor}`}>{factor} ×</button>{/each}</div>{:else}<span>Choose variables below.</span>{/if}</div>
            <label>Available variables<input aria-label="Search model variables" placeholder="Search variables" bind:value={query} /></label>
            <div class="stack-factors" role="group" aria-label="Model variables">{#each filtered as f}<label><input type="checkbox" checked={factors.includes(f)} onchange={() => toggleFactor(f)} />{f}</label>{/each}</div>
            <p><button class="text-button" onclick={() => (showSelected = !showSelected)}>{showSelected ? 'Show matching variables' : 'Show selected only'}</button></p>
            {#if kind === 'catalogue'}<h3>Full catalogue</h3><p>Upload the complete CSV or Parquet file, including unseen keys.</p><input type="file" accept=".csv,.parquet" aria-label="Upload catalogue" onchange={upload} disabled={!!busy} />
                {#if catalogue}<p>{catalogue.filename} · {catalogue.rows} rows</p><label>Unique key<select aria-label="Catalogue key" bind:value={settings.key}><option value="">Choose key</option>{#each catalogue.columns as c}<option value={c.name || c}>{c.name || c}</option>{/each}</select></label><p>Select required catalogue attributes in Factors above.</p>
                    <label>Missing-key fallback value (optional)<input aria-label="Missing key fallback" type="number" min="0.000001" step="any" value={settings.fallback ?? ''} onchange={(e) => (settings = { ...settings, fallback: e.currentTarget.value ? Number(e.currentTarget.value) : null })} /></label>{/if}
            {/if}
            <div class="form-grid">
                {#if kind === 'glm'}<label>Alpha<input aria-label="GLM alpha" type="number" min="0" step="any" bind:value={settings.alpha} /></label><label>CV folds (optional)<input aria-label="GLM CV folds" type="number" min="2" value={settings.cv ?? ''} onchange={(e) => (settings = { ...settings, cv: e.currentTarget.value ? Number(e.currentTarget.value) : null, alpha: e.currentTarget.value ? null : 0.001 })} /></label><label>L1 ratio<input aria-label="GLM L1 ratio" type="number" min="0" max="1" step=".1" bind:value={settings.l1_ratio} /></label>
                {:else if kind === 'interaction'}<label>CV folds<input aria-label="Pair CV folds" type="number" min="2" bind:value={settings.cv_folds} /></label><label>Trial time limit (minutes)<input aria-label="Pair time limit" type="number" min="1" bind:value={settings.time_limit_minutes} /></label><label>Trial iterations<input aria-label="Pair iterations" type="number" min="1" value={settings.candidates[0].iterations} onchange={(e) => (settings = { ...settings, candidates: [{ ...settings.candidates[0], iterations: Number(e.currentTarget.value) }] })} /></label>
                {:else}<label>Iterations<input aria-label="CatBoost iterations" type="number" min="1" bind:value={settings.iterations} /></label><label>Depth<input aria-label="CatBoost depth" type="number" min="1" bind:value={settings.depth} /></label><label>Learning rate<input aria-label="CatBoost learning rate" type="number" min="0.0001" step="any" bind:value={settings.learning_rate} /></label><label>Trial time limit (minutes)<input aria-label="CatBoost time limit" type="number" min="1" bind:value={settings.time_limit_minutes} /></label>{/if}
            </div><div class="editor-actions"><button onclick={() => save(false)} disabled={!!busy}>Save draft</button><button class="primary" onclick={() => save(true)} disabled={!!busy}>Fit model</button></div></section>{/if}
                </div>
            </div>
        {/if}
        {#if result && view === 'diagnostics'}<section class="model-card"><h2>Actual versus expected</h2><p>Uses the complete accepted scorer. Training rows are shown first.</p><div class="stack-diagnostic-controls"><label>Rows<select aria-label="Diagnostic rows" bind:value={subset}><option value="train">Training (in-sample)</option><option value="holdout">Holdout (request evaluation)</option></select></label><label>Search group factors<input aria-label="Search A/E factors" placeholder="Search factors" bind:value={diagnosticQuery} /></label><label>Group by<select aria-label="A/E factor" bind:value={variable}><option value="">Overall</option>{#each filteredReviewChoices as n}<option value={n}>{n}</option>{/each}</select></label></div>
            {#if diagnostics}<dl class="stack-metrics">{#each metricDefinitions.filter((definition) => definition.section === 'primary' && diagnostics.metrics?.[definition.key] != null) as definition}<div><dt>{definition.label}</dt><dd>{num(diagnostics.metrics[definition.key])}</dd></div>{/each}</dl>
                <p class="stack-totals">{#each ['rows', 'exposure', 'actual', 'expected'] as key}<span>{key.charAt(0).toUpperCase() + key.slice(1)}: <strong>{num(diagnostics.metrics?.[key])}</strong></span>{/each}</p>
                <p class="stack-totals">{#each metricDefinitions.filter((definition) => definition.section === 'secondary' && diagnostics.metrics?.[definition.key] != null) as definition}<span>{definition.label}: <strong>{num(diagnostics.metrics[definition.key])}</strong></span>{/each}</p>
                {#if diagnostics.catalogue_coverage && Object.keys(diagnostics.catalogue_coverage).length}<h3>Catalogue coverage on these rows</h3><div class="stack-table"><table><thead><tr><th>Stage</th><th>Rows</th><th>Missing keys</th><th>Fallback rows</th></tr></thead><tbody>{#each Object.entries(diagnostics.catalogue_coverage) as [id, coverage]}<tr><td>{components.find((c) => (c.id || c.component_id) === id)?.name || id}</td><td>{coverage.rows}</td><td>{coverage.missing_key_rows}</td><td>{coverage.fallback_rows}</td></tr>{/each}</tbody></table></div>{/if}
                {#if diagnostics.ae?.length}<DiagnosticPlot rows={diagnostics.ae} title={`Actual and expected by ${variable}`} xKey="label" kind={wb.columns.find((column) => column.name === variable)?.numeric || diagnostics.ae.some((row) => /^[<≥\[]/.test(String(row.label))) ? 'numeric' : 'categorical'} showTable={false} ariaLabel={`Actual and expected by ${variable} on ${subset}`} /><div class="stack-table"><table><thead><tr>{#each Object.keys(diagnostics.ae[0]) as key}<th>{key.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each diagnostics.ae as row}<tr>{#each Object.values(row) as value}<td>{typeof value === 'number' ? Number(value.toPrecision(3)) : value ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{/if}
                {#if diagnostics.lift?.length}<DiagnosticPlot rows={diagnostics.lift} title="Lift by predicted risk" xKey="bin" showTable={false} ariaLabel={`Actual and expected lift on ${subset}`} /><div class="stack-table"><table><thead><tr>{#each Object.keys(diagnostics.lift[0]) as key}<th>{key.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each diagnostics.lift as row}<tr>{#each Object.values(row) as value}<td>{typeof value === 'number' ? Number(value.toPrecision(3)) : value ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{/if}
            {/if}</section>{/if}
        {#if result && view === 'compare'}<section class="model-card"><h2>Compare named models</h2><label>Compare with<select aria-label="Stack challenger" bind:value={challenger}><option value="">Choose a fitted model</option>{#each fitted.filter((n) => n !== selected) as n}<option value={n}>{n}</option>{/each}</select></label><label>Rows<select aria-label="Comparison rows" bind:value={subset}><option value="train">Training (in-sample)</option><option value="holdout">Holdout (request evaluation)</option></select></label>{#if comparison}<p>Expected change: {comparison.expected_change == null ? '—' : (100 * comparison.expected_change).toFixed(2) + '%'}</p><div class="stack-compare"><div><h3>{comparison.model}</h3><p>A/E {comparison.metrics?.ae == null ? '—' : Number(comparison.metrics.ae).toFixed(3)}</p></div><div><h3>{comparison.challenger}</h3><p>A/E {comparison.challenger_metrics?.ae == null ? '—' : Number(comparison.challenger_metrics.ae).toFixed(3)}</p></div></div>{/if}</section>{/if}
        {#if result && view === 'tables'}<section class="model-card"><h2>Component table or lookup</h2><label>Model component<select aria-label="Inspect model component" bind:value={stage}>{#each components as c}<option value={c.id || c.component_id}>{c.name}</option>{/each}</select></label>{#if table}{#if table.unit_label}<p class="help-text">{table.unit_label}</p>{/if}{#if table.coverage}<p class="help-text">{table.evidence?.catalogue_rows ?? '—'} catalogue entries · {table.coverage.rows} training rows · {table.coverage.missing_key_rows || 0} missing-key rows · {table.coverage.fallback_rows || 0} fallback rows</p>{/if}{#if unseenSummary(table.evidence?.unseen_attribute_levels)}<p class="help-text">Unseen attribute levels: {unseenSummary(table.evidence?.unseen_attribute_levels)}</p>{/if}{#each table.tables || [] as item}<h3>{item.name}</h3><div class="stack-table"><table><thead><tr>{#each item.columns as column}<th>{column.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each item.rows.slice(0, 300) as row}<tr>{#each item.columns as column}<td>{typeof row[column] === 'number' ? Number(row[column].toPrecision(7)) : row[column] ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{#if item.rows.length > 300}<p>Showing 300 of {item.rows.length} rows. Download the full rating file on Export.</p>{/if}{/each}{#if table.message}<p>{table.message}</p>{/if}{/if}
            {#if table?.editable}<h3>Adjust this lookup</h3><p>Preview the change to the lookup and full model. Apply creates a new named candidate.</p><label>Adjustment<select aria-label="Lookup adjustment" bind:value={operation} onchange={() => (preview = null)}><option value="cap">Cap lookup value</option><option value="round">Round for deployment</option></select></label><div class="form-grid">{#if operation === 'cap'}<label>Minimum lookup value<input aria-label="Minimum lookup value" type="number" step="any" bind:value={lower} oninput={() => (preview = null)} /></label><label>Maximum lookup value<input aria-label="Maximum lookup value" type="number" step="any" bind:value={upper} oninput={() => (preview = null)} /></label>{:else}<label>Decimal places<input aria-label="Rate decimal places" type="number" min="0" bind:value={decimals} oninput={() => (preview = null)} /></label>{/if}</div><label><input type="checkbox" bind:checked={refitLater} onchange={() => (preview = null)} />Refit later stages after applying (otherwise keep them fixed)</label><button onclick={editPreview} disabled={!!busy}>Preview adjustment</button>{#if preview}<p>{preview.changed_rows} rows changed · expected {preview.expected_before} → {preview.expected_after}</p><pre class="stack-json">{JSON.stringify(preview.sample_rows, null, 2)}</pre><label>New candidate name<input aria-label="Amended candidate name" bind:value={amendedName} /></label><button onclick={editApply} disabled={!!busy || !amendedName}>Apply to new candidate</button>{/if}{/if}
        </section>{/if}
        {#if result && view === 'export'}<section class="model-card"><h2>Combined scorer</h2><p>Save the complete fitted stack and inspect its component tables.</p><div class="stack-review-actions"><button onclick={() => download('easyglm')} disabled={!!busy}>Download JSON scorer</button><button onclick={() => download('python_score')} disabled={!!busy}>Download Python scorer</button><button onclick={() => download('xlsx')} disabled={!!busy}>Download Excel component workbook</button></div><h2>Stage rating files</h2>{#each components as c}<div class="stack-line"><span>{c.name} · {labels[c.kind]}</span>{#if c.kind === 'catalogue'}<button onclick={() => download('csv', c.id || c.component_id)} disabled={!!busy}>Download full lookup CSV</button>{:else}<span>{c.kind === 'catboost' ? 'Included in combined scorer' : 'Inspect on Rate tables'}</span>{/if}</div>{/each}</section>{/if}
        {#if !result && view !== 'model'}<section class="model-card"><p>Fit a named stacked candidate to inspect these results.</p><button onclick={() => onNavigate('model')}>Go to Model</button></section>{/if}
    {/if}
</div>
{#if pendingAction}<div class="draft-dialog-backdrop" role="presentation"><div class="draft-dialog" role="dialog" aria-modal="true" aria-label="Unsaved draft"><h2>Keep these draft changes?</h2><p>You have unsaved changes to this model.</p><div><button onclick={() => resolvePending('keep')}>Keep editing</button><button onclick={() => resolvePending('save')} disabled={!!busy}>Save draft</button><button onclick={() => resolvePending('discard')}>Discard changes</button></div></div></div>{/if}
<style>
    .stack-line { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:12px 0; }
    .stack-components { padding-left:22px; }
    .stack-components li { display:grid; grid-template-columns: 1fr 1fr 2fr 1fr auto; gap:12px; border-top:1px solid #dae3db; padding:10px 4px; }
    .stack-factors { max-height:220px; overflow:auto; border:1px solid #dae3db; padding:8px; margin:8px 0; display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:4px 12px; }
    .stack-factors label { display:flex; align-items:center; gap:6px; }
    .stack-json { max-height:440px; overflow:auto; padding:12px; background:#f5f8f5; font-size:12px; }
    .stack-metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px; }
    .stack-metrics div, .stack-compare div { padding:12px; background:#f5f8f5; }
    .stack-metrics dt { text-transform:capitalize; font-size:12px; }
    .stack-metrics dd { margin:4px 0 0; font-weight:bold; }
    .stack-totals { display:flex; flex-wrap:wrap; gap:8px 18px; }
    .stack-table { overflow:auto; max-height:420px; }
    .stack-table table { width:100%; border-collapse:collapse; }
    .stack-table th, .stack-table td { padding:7px; border-bottom:1px solid #dae3db; text-align:left; white-space:nowrap; }
    .stack-compare { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .stack-review-actions { display:flex; flex-wrap:wrap; gap:8px; }
    .stack-diagnostic-controls { display:flex; flex-wrap:wrap; gap:12px; margin:12px 0; }
    .stack-diagnostic-controls label { display:grid; gap:5px; min-width:160px; }
    .stack-divide { display:flex; align-items:center; gap:8px; margin:10px 0; }
    .stack-form-heading { display:flex; justify-content:space-between; gap:12px; align-items:start; }
    .text-button { border:0; background:transparent; color:#154b35; text-decoration:underline; padding:0 3px; }
    .model-builder { display:grid; grid-template-columns:minmax(230px, 300px) minmax(0, 1fr); gap:18px; align-items:start; }
    .model-list { position:sticky; top:12px; border:1px solid #c7d7ca; border-radius:12px; padding:14px; background:#f7faf7; min-width:0; }
    .model-list-heading { display:flex; justify-content:space-between; align-items:start; gap:12px; }.model-list-heading h2 { margin:2px 0; }
    .model-list-items { display:grid; gap:6px; margin-top:12px; }.model-list-item { display:grid; gap:3px; width:100%; text-align:left; color:inherit; background:white; border:1px solid #d6e0d7; padding:9px; white-space:normal; }.model-list-item.active { border-color:#287550; box-shadow:inset 3px 0 #287550; }.model-list-item span { color:#245f43; font-size:12px; }.model-list-item small { color:#526e5d; overflow-wrap:anywhere; }.mobile-model-select { display:none; }
    .model-map { margin-top:14px; }.model-map :global(.model-flow) { margin:10px 0 0; padding:10px; }.model-map :global(.flow-heading) { display:none; }
    .selected-workspace { min-width:0; }.selected-model-header { display:grid; gap:14px; }.selected-model-identity { display:grid; grid-template-columns:minmax(0, 1fr) auto; align-items:start; gap:12px; }.selected-model-identity h2 { margin:2px 0; overflow-wrap:anywhere; }.selected-model-header .model-state { border-radius:20px; padding:4px 9px; background:#eef2ee; white-space:nowrap; }.selected-model-header .fit-state { color:#1e6745; background:#e0f0e4; }.selected-model-context { width:100%; box-sizing:border-box; padding:12px 14px; border-left:3px solid #6e9478; background:#f5f8f5; }.selected-model-context.recovery { border-color:#a27944; background:#fff8ed; }.selected-model-context p { margin:0; }.selected-model-context p + p { margin-top:6px; }.fit-progress { color:#285c45; align-self:center; }.workspace-actions,.editor-actions,.result-tabs { display:flex; flex-wrap:wrap; gap:8px; margin-top:14px; }.selected-model-actions { margin-top:0; }.recovery-consequence { margin:0; color:#526e5d; }.result-tabs { border-bottom:1px solid #dae3db; padding-bottom:10px; }.result-tabs button { background:transparent; border:0; color:#285c45; }.result-tabs button.tab-active { font-weight:bold; border-bottom:3px solid #287550; border-radius:0; }.selected-factors { margin:12px 0; padding:10px; border:1px solid #d6e0d7; background:#f7faf7; display:grid; gap:7px; }.selected-factors > div { display:flex; gap:5px; flex-wrap:wrap; }.selected-factor { font-size:12px; padding:3px 6px; }.empty-workspace { min-height:180px; }.model-editor { border-top:4px solid #287550; }
    .draft-dialog-backdrop { position:fixed; inset:0; display:grid; place-items:center; background:rgb(20 35 25 / .35); z-index:20; padding:16px; }.draft-dialog { max-width:430px; background:white; padding:22px; border-radius:10px; box-shadow:0 6px 30px #2339; }.draft-dialog h2 { margin-top:0; }.draft-dialog div { display:flex; flex-wrap:wrap; gap:8px; }
    @media (max-width:900px) { .stack-components li { grid-template-columns:1fr 1fr; }.model-builder { grid-template-columns:1fr; }.model-list { position:static; }.model-list-items { display:none; }.mobile-model-select { display:grid; gap:5px; margin-top:12px; }.model-map { display:none; } }
    @media (max-width:430px) { .selected-model-identity { gap:8px; }.selected-model-header .model-state { font-size:12px; padding:4px 7px; }.selected-model-context { padding:10px; }.workspace-actions button { max-width:100%; white-space:normal; } }
</style>
