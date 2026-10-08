<script>
    import { onDestroy } from 'svelte';
    import DiagnosticPlot from './DiagnosticPlot.svelte';
    import ModelFlow from './ModelFlow.svelte';
    import { formatNumber as num } from './format.js';
    export let api, state, view, uploadCatalogue, saveDownload;
    export let onState = () => {}, onReady = () => {}, onContext = () => {}, onNavigate = () => {}, onSingle = () => {};
    const labels = { glm: 'GLM', catboost: 'CatBoost', interaction: 'Pair interaction', catalogue: 'Catalogue lookup', legacy: 'Accepted legacy model' };
    function statusLabel(status, fitted) {
        if (!status) return fitted ? 'Fitted' : 'Awaiting fit';
        if (status === 'current') return 'Fitted';
        if (status === 'amended_prefix_kept_fixed') return 'Kept after earlier edit · review fit evidence';
        return status.replaceAll('_', ' ');
    }
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
        lower = '', upper = '', decimals = '', operation = 'cap', amendedName = '', refitLater = false, viewKey = '';
    $: names = wb ? Object.keys(wb.models || {}).filter((n) => wb.models[n]?.components?.length) : [];
    $: fitted = names.filter((n) => jobs[n]?.applicable);
    $: modelFlow = wb?.model_flow || [];
    $: selectedFlow = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || []).find((entry) => entry.name === selected);
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
    function toggleFactor(value) { factors = factors.includes(value) ? factors.filter((n) => n !== value) : [...factors, value]; }
    async function refresh() {
        reading = true;
        try {
            wb = await api('workbench'); jobs = wb.jobs || {}; known = `${wb.session_id}:${wb.revision}`;
            if (!wb.models[selected]?.components?.length) selected = Object.keys(wb.models).find((n) => wb.models[n]?.components?.length) || '';
            if (source && !wb.stack_sources?.[source]) source = '';
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
                await loadResult();
            }
        } catch (e) { error = e.message; }
    }
    async function loadResult() {
        const model = selected;
        if (!jobs[model]?.applicable) { result = null; return; }
        const data = await api('results/' + encodeURIComponent(model));
        if (model !== selected) return;
        if (!data.stack) { error = 'Open this model in Single model.'; result = null; return; }
        result = data;
        const currentComponents = data.components || wb?.models?.[model]?.components || [];
        if (!currentComponents.some((c) => (c.id || c.component_id) === stage)) stage = currentComponents.at(-1)?.id || currentComponents.at(-1)?.component_id || '';
        await loadView();
    }
    async function chooseSelected() {
        await loadResult();
    }
    function clearEditor() {
        source = ''; name = ''; stageName = ''; factors = []; query = ''; showSelected = false;
        kind = 'glm'; settings = structuredClone(defaults.glm); catalogue = null;
    }
    function addAfter(model) {
        source = model;
        name = ''; stageName = ''; factors = []; query = ''; showSelected = false; catalogue = null;
        kind = 'glm'; settings = structuredClone(defaults.glm);
        notice = model ? `New candidate will build on ${model}. Choose its next stage.` : 'New standalone candidate. Choose its first stage.';
    }
    function createAlternative(model) {
        const item = (Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || []).find((entry) => entry.name === model) || {};
        const edge = (Array.isArray(modelFlow) ? [] : modelFlow?.edges || []).find((entry) => entry.to === model);
        const parent = edge?.from || item.parent_name || '';
        const last = (item.components || wb?.models?.[model]?.components || []).at(-1);
        source = parent;
        name = ''; stageName = last?.name ? `${last.name} alternative` : '';
        kind = last?.kind || 'glm';
        factors = [...(last?.factors || last?.inputs || [])];
        settings = structuredClone(last?.settings || defaults[kind]);
        catalogue = null; query = ''; showSelected = false;
        notice = parent
            ? `Alternative starts from ${parent}; its last stage was copied for editing.`
            : 'Standalone alternative copied its stage inputs for editing.';
    }
    async function loadView() {
        if (!result) return;
        try {
            if (view === 'diagnostics') diagnostics = await api(`stack/${encodeURIComponent(selected)}/diagnostics?subset=${subset}${variable ? '&variable=' + encodeURIComponent(variable) : ''}`);
            if (view === 'compare' && challenger) comparison = await api(`stack/${encodeURIComponent(selected)}/compare?subset=${subset}&challenger=${encodeURIComponent(challenger)}`);
            if (view === 'tables' && stage) table = await api(`stack/${encodeURIComponent(selected)}/component/${encodeURIComponent(stage)}/table`);
        } catch (e) { error = e.message; }
    }
    async function upload(event) {
        const file = event.currentTarget.files?.[0]; if (!file) return;
        busy = 'upload'; error = '';
        try { catalogue = await uploadCatalogue(file, rev()); settings = { ...settings, source_path: catalogue.source_path, source_sha256: catalogue.source_sha256 }; }
        catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function save() {
        busy = 'save'; error = ''; notice = '';
        try {
            if (!name.trim() || wb.models[name]) throw new Error('Enter a new candidate name. Existing checkpoints stay unchanged.');
            if (!stageName.trim()) throw new Error('Name the new stage.');
            if (!factors.length || (kind === 'interaction' && factors.length !== 2)) throw new Error(kind === 'interaction' ? 'Choose exactly two factors.' : 'Choose at least one factor.');
            if (source && !jobs[source]?.applicable) throw new Error('Fit the selected starting candidate first.');
            if (source && (!sourceInfo?.source_fit_id || !sourceInfo?.components?.length)) throw new Error('Refresh the fitted checkpoint before appending.');
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
            const snapshot = await api('models/save', { ...rev(), name: name.trim(), create: true,
                ...(source ? { source_name: source, source_fit_id: sourceInfo.source_fit_id } : {}),
                fields: { ...fields, components: [...(source ? sourceInfo.components : []), component], interactions: [], pair_stages: [] } });
            onState(snapshot); selected = name.trim(); result = null; await refresh(); notice = 'Candidate saved. Fit to inspect its complete score.';
        } catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function fit(fullRefit = false) {
        busy = 'fit'; error = '';
        try { await api(`models/${encodeURIComponent(selected)}/fit`, { ...rev(), ...(fullRefit ? { full_refit: true } : {}) }); jobs = await api('jobs'); notice = fullRefit ? 'Full recipe rebuild started.' : 'Fit started.'; }
        catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function cancel() { try { jobs = await api(`models/${encodeURIComponent(selected)}/cancel`, rev()); } catch (e) { error = e.message; } }
    async function editPreview() {
        busy = 'preview'; error = ''; preview = null;
        try { preview = await api(`stack/${encodeURIComponent(selected)}/component/${encodeURIComponent(stage)}/preview`, { ...rev(), operation, lower: lower === '' ? null : Number(lower), upper: upper === '' ? null : Number(upper), decimals: decimals === '' ? null : Number(decimals), refit_later: refitLater }); }
        catch (e) { error = e.message; } finally { busy = ''; }
    }
    async function editApply() {
        busy = 'apply'; error = '';
        try {
            const response = await api(`stack/${encodeURIComponent(selected)}/component/${encodeURIComponent(stage)}/apply`, { ...rev(), operation, lower: lower === '' ? null : Number(lower), upper: upper === '' ? null : Number(upper), decimals: decimals === '' ? null : Number(decimals), new_name: amendedName.trim(), refit_later: refitLater });
            onState(response.snapshot || response); selected = amendedName.trim(); preview = null; await refresh(); notice = 'Amended model saved as a new candidate.';
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
    <div class="heading"><div><div class="eyebrow">ORDERED MODEL</div><h1>{view === 'model' ? 'Build a model in stages' : view === 'tables' ? 'Component tables' : view === 'export' ? 'Export stacked model' : view === 'compare' ? 'Compare models' : 'Full-model diagnostics'}</h1><p>Each new candidate preserves its starting checkpoint and adds one fitted stage.</p></div></div>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if notice}<div class="message" role="status">{notice}</div>{/if}
    {#if !wb}<p>Loading model settings…</p>{:else}
        {#if view === 'model'}<ModelFlow {modelFlow} models={wb.models || {}} {jobs} {selected} onSelect={(model) => { if (wb.models[model]?.components?.length) { selected = model; chooseSelected(); } else onSingle(model); }} onAdd={addAfter} onAlternative={createAlternative} />{/if}
        <section class="model-card"><div class="stack-line"><label>Named candidate<select aria-label="Stacked model" bind:value={selected} onchange={chooseSelected}><option value="">Choose a candidate</option>{#each names as n}<option value={n}>{n}</option>{/each}</select></label>
            {#if selected}<span>{jobs[selected]?.status || 'Not fitted'}</span>{/if}
            {#if selected && !jobs[selected]?.applicable}<button onclick={() => fit(Boolean(selectedFlow?.full_refit_required))} disabled={!!busy || ['running','queued'].includes(jobs[selected]?.status)}>{selectedFlow?.full_refit_required ? 'Rebuild full model' : 'Fit candidate'}</button>{/if}
            {#if selectedFlow?.full_refit_required}<small class="stack-support">The prior fitted parent is no longer available. Rebuild fits this complete saved recipe afresh; it does not score the missing parent.</small>{/if}
            {#if ['running','queued'].includes(jobs[selected]?.status)}<button onclick={cancel}>Cancel fit</button>{/if}</div>
            {#if selected}<ol class="stack-components">{#each components as c, i}<li><strong>{c.name}</strong><span>{labels[c.kind]}</span><span>{(c.factors || c.inputs || []).join(', ')}</span><span>From {i ? components[i - 1].name : 'starting level'}</span><span>{statusLabel(c.status, jobs[selected]?.applicable)}</span>
                {#if c.kind === 'catalogue'}<small class="stack-support">{c.unit_label || 'Catalogue lookup'} · {c.evidence?.catalogue_rows ?? '—'} catalogue entries · {c.coverage?.rows ?? '—'} training rows · {c.coverage?.missing_key_rows ?? 0} missing-key rows · {c.coverage?.fallback_rows ?? 0} fallback rows{#if unseenSummary(c.evidence?.unseen_attribute_levels)} · Unseen attribute levels: {unseenSummary(c.evidence?.unseen_attribute_levels)}{/if}</small>{/if}
            </li>{/each}</ol>{/if}
        </section>
        {#if view === 'model'}<section class="model-card"><div class="stack-form-heading"><div><h2>{source ? 'Add model after this' : 'Add standalone model'}</h2><p>{source ? 'Each saved name is a complete model. Existing fitted predictions remain fixed as the offset for its child.' : kind === 'catboost' ? 'Includes its own fitted starting level; no predecessor model offset.' : 'This complete model starts without a predecessor-model offset.'}</p></div><button onclick={clearEditor}>Reset editor</button></div>
            <div class="form-grid"><label>Start from<select aria-label="Start from fitted candidate" bind:value={source}><option value="">New model</option>{#each fittedSources as n}<option value={n}>{n}{wb.models[n]?.components?.length ? '' : ' · fitted single model'}</option>{/each}</select></label>
                <label>New candidate name<input aria-label="New stacked model name" bind:value={name} /></label>
                <label>Stage type<select aria-label="New stage type" value={kind} onchange={(e) => changeKind(e.currentTarget.value)}>{#each Object.entries(labels).filter(([key]) => key !== 'legacy') as [key,label]}<option value={key}>{label}</option>{/each}</select></label>
                <label>Stage name<input aria-label="New stage name" bind:value={stageName} /></label></div>
            {#if sourceInfo}<p class="help-text">Builds on <strong>{source}</strong> · complete model predictions as offset. Its complete fitted chain is kept: {sourceInfo.components.map((item) => item.name).join(' → ')}.</p>{/if}
            <div class="form-grid"><label>Model family{#if source}<input aria-label="Stack model family" value={wb.models[source]?.family || ''} readonly />{:else}<select aria-label="Stack model family" bind:value={stackFamily}><option value="poisson">Poisson</option><option value="tweedie">Tweedie</option><option value="gaussian">Gaussian</option></select>{/if}</label>
                <label>Link<input aria-label="Stack model link" value="Log" readonly /></label>
                <label>Target<input aria-label="Stack model target" value={source ? wb.models[source]?.target || '' : wb.target || ''} readonly /></label>
                <label>Weight or exposure<input aria-label="Stack model weight" value={source ? wb.models[source]?.weight || '' : wb.weight || ''} readonly /></label>
                <label>External offset<input aria-label="Stack model offset" value={source ? wb.models[source]?.offset || '' : wb.offset || ''} readonly /></label>
                {#if source && wb.models[source]?.family === 'tweedie'}<label>Tweedie power<input aria-label="Stack Tweedie power" value={wb.models[source]?.tweedie_power ?? ''} readonly /></label>{:else if !source && stackFamily === 'tweedie'}<label>Tweedie power<input aria-label="Stack Tweedie power" type="number" min="1.01" max="1.99" step=".05" bind:value={tweediePower} /></label>{/if}
            </div>
            <label class="stack-divide"><input aria-label="Divide target by weight" type="checkbox" checked={source ? Boolean(wb.models[source]?.divide_target_by_weight) : divideTargetByWeight} disabled={!!source} onchange={(e) => (divideTargetByWeight = e.currentTarget.checked)} />Divide target by weight when fitting unit outcomes</label>
            <h3>Variables</h3><p class="help-text">{kind === 'catalogue' ? 'Upload a catalogue below, then choose its required attributes.' : kind === 'catboost' ? 'CatBoost receives these raw input variables.' : 'GLM uses these selected variables with the project’s shared settings.'}</p><input aria-label="Search stage factors" placeholder="Search variables" bind:value={query} />
            <div class="stack-factors" role="group" aria-label="Stage factors">{#each filtered as f}<label><input type="checkbox" checked={factors.includes(f)} onchange={() => toggleFactor(f)} />{f}</label>{/each}</div>
            <p>{factors.length ? `${factors.length} selected: ${factors.join(', ')}` : 'No factors selected'} <button class="text-button" onclick={() => (showSelected = !showSelected)}>{showSelected ? 'Show all' : 'Show selected'}</button></p>
            {#if kind === 'catalogue'}<h3>Full catalogue</h3><p>Upload the complete CSV or Parquet file, including unseen keys.</p><input type="file" accept=".csv,.parquet" aria-label="Upload catalogue" onchange={upload} disabled={!!busy} />
                {#if catalogue}<p>{catalogue.filename} · {catalogue.rows} rows</p><label>Unique key<select aria-label="Catalogue key" bind:value={settings.key}><option value="">Choose key</option>{#each catalogue.columns as c}<option value={c.name || c}>{c.name || c}</option>{/each}</select></label><p>Select required catalogue attributes in Factors above.</p>
                    <label>Missing-key fallback value (optional)<input aria-label="Missing key fallback" type="number" min="0.000001" step="any" value={settings.fallback ?? ''} onchange={(e) => (settings = { ...settings, fallback: e.currentTarget.value ? Number(e.currentTarget.value) : null })} /></label>{/if}
            {/if}
            <div class="form-grid">
                {#if kind === 'glm'}<label>Alpha<input aria-label="GLM alpha" type="number" min="0" step="any" bind:value={settings.alpha} /></label><label>CV folds (optional)<input aria-label="GLM CV folds" type="number" min="2" value={settings.cv ?? ''} onchange={(e) => (settings = { ...settings, cv: e.currentTarget.value ? Number(e.currentTarget.value) : null, alpha: e.currentTarget.value ? null : 0.001 })} /></label><label>L1 ratio<input aria-label="GLM L1 ratio" type="number" min="0" max="1" step=".1" bind:value={settings.l1_ratio} /></label>
                {:else if kind === 'interaction'}<label>CV folds<input aria-label="Pair CV folds" type="number" min="2" bind:value={settings.cv_folds} /></label><label>Trial time limit (minutes)<input aria-label="Pair time limit" type="number" min="1" bind:value={settings.time_limit_minutes} /></label><label>Trial iterations<input aria-label="Pair iterations" type="number" min="1" value={settings.candidates[0].iterations} onchange={(e) => (settings = { ...settings, candidates: [{ ...settings.candidates[0], iterations: Number(e.currentTarget.value) }] })} /></label>
                {:else}<label>Iterations<input aria-label="CatBoost iterations" type="number" min="1" bind:value={settings.iterations} /></label><label>Depth<input aria-label="CatBoost depth" type="number" min="1" bind:value={settings.depth} /></label><label>Learning rate<input aria-label="CatBoost learning rate" type="number" min="0.0001" step="any" bind:value={settings.learning_rate} /></label><label>Trial time limit (minutes)<input aria-label="CatBoost time limit" type="number" min="1" bind:value={settings.time_limit_minutes} /></label>{/if}
            </div><button onclick={save} disabled={!!busy}>Save new candidate</button></section>
            {#if result}<section class="model-card"><h2>Review complete model</h2><p>{result.summary?.components || components.length} stages · {result.summary?.family || ''} · training A/E {result.summary?.train_ae == null ? '—' : Number(result.summary.train_ae).toFixed(3)}</p><div class="stack-review-actions"><button onclick={() => onNavigate('diagnostics')}>Inspect A/E</button><button onclick={() => onNavigate('tables')}>Inspect tables</button><button onclick={() => onNavigate('export')}>Export</button></div></section>{/if}
        {/if}
        {#if result && view === 'diagnostics'}<section class="model-card"><h2>Actual versus expected</h2><p>Uses the complete accepted scorer. Training rows are shown first.</p><div class="stack-diagnostic-controls"><label>Rows<select aria-label="Diagnostic rows" bind:value={subset}><option value="train">Training (in-sample)</option><option value="holdout">Holdout (request evaluation)</option></select></label><label>Search group factors<input aria-label="Search A/E factors" placeholder="Search factors" bind:value={diagnosticQuery} /></label><label>Group by<select aria-label="A/E factor" bind:value={variable}><option value="">Overall</option>{#each filteredReviewChoices as n}<option value={n}>{n}</option>{/each}</select></label></div>
            {#if diagnostics}<dl class="stack-metrics">{#each metricDefinitions.filter((definition) => definition.section === 'primary' && diagnostics.metrics?.[definition.key] != null) as definition}<div><dt>{definition.label}</dt><dd>{num(diagnostics.metrics[definition.key])}</dd></div>{/each}</dl>
                <p class="stack-totals">{#each ['rows', 'exposure', 'actual', 'expected'] as key}<span>{key.charAt(0).toUpperCase() + key.slice(1)}: <strong>{num(diagnostics.metrics?.[key])}</strong></span>{/each}</p>
                <p class="stack-totals">{#each metricDefinitions.filter((definition) => definition.section === 'secondary' && diagnostics.metrics?.[definition.key] != null) as definition}<span>{definition.label}: <strong>{num(diagnostics.metrics[definition.key])}</strong></span>{/each}</p>
                {#if diagnostics.catalogue_coverage && Object.keys(diagnostics.catalogue_coverage).length}<h3>Catalogue coverage on these rows</h3><div class="stack-table"><table><thead><tr><th>Stage</th><th>Rows</th><th>Missing keys</th><th>Fallback rows</th></tr></thead><tbody>{#each Object.entries(diagnostics.catalogue_coverage) as [id, coverage]}<tr><td>{components.find((c) => (c.id || c.component_id) === id)?.name || id}</td><td>{coverage.rows}</td><td>{coverage.missing_key_rows}</td><td>{coverage.fallback_rows}</td></tr>{/each}</tbody></table></div>{/if}
                {#if diagnostics.ae?.length}<DiagnosticPlot rows={diagnostics.ae} title={`Actual and expected by ${variable}`} xKey="label" kind={wb.columns.find((column) => column.name === variable)?.numeric || diagnostics.ae.some((row) => /^[<≥\[]/.test(String(row.label))) ? 'numeric' : 'categorical'} showTable={false} ariaLabel={`Actual and expected by ${variable} on ${subset}`} /><div class="stack-table"><table><thead><tr>{#each Object.keys(diagnostics.ae[0]) as key}<th>{key.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each diagnostics.ae as row}<tr>{#each Object.values(row) as value}<td>{typeof value === 'number' ? Number(value.toPrecision(3)) : value ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{/if}
                {#if diagnostics.lift?.length}<DiagnosticPlot rows={diagnostics.lift} title="Lift by predicted risk" xKey="bin" showTable={false} ariaLabel={`Actual and expected lift on ${subset}`} /><div class="stack-table"><table><thead><tr>{#each Object.keys(diagnostics.lift[0]) as key}<th>{key.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each diagnostics.lift as row}<tr>{#each Object.values(row) as value}<td>{typeof value === 'number' ? Number(value.toPrecision(3)) : value ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{/if}
            {/if}</section>{/if}
        {#if result && view === 'compare'}<section class="model-card"><h2>Compare named candidates</h2><label>Challenger<select aria-label="Stack challenger" bind:value={challenger}><option value="">Choose fitted candidate</option>{#each fitted.filter((n) => n !== selected) as n}<option value={n}>{n}</option>{/each}</select></label><label>Rows<select aria-label="Comparison rows" bind:value={subset}><option value="train">Training (in-sample)</option><option value="holdout">Holdout (request evaluation)</option></select></label>{#if comparison}<p>Expected change: {comparison.expected_change == null ? '—' : (100 * comparison.expected_change).toFixed(2) + '%'}</p><div class="stack-compare"><div><h3>{comparison.model}</h3><p>A/E {comparison.metrics?.ae == null ? '—' : Number(comparison.metrics.ae).toFixed(3)}</p></div><div><h3>{comparison.challenger}</h3><p>A/E {comparison.challenger_metrics?.ae == null ? '—' : Number(comparison.challenger_metrics.ae).toFixed(3)}</p></div></div>{/if}</section>{/if}
        {#if result && view === 'tables'}<section class="model-card"><h2>Stage table or lookup</h2><label>Stage<select aria-label="Inspect stack stage" bind:value={stage}>{#each components as c}<option value={c.id || c.component_id}>{c.name}</option>{/each}</select></label>{#if table}{#if table.unit_label}<p class="help-text">{table.unit_label}</p>{/if}{#if table.coverage}<p class="help-text">{table.evidence?.catalogue_rows ?? '—'} catalogue entries · {table.coverage.rows} training rows · {table.coverage.missing_key_rows || 0} missing-key rows · {table.coverage.fallback_rows || 0} fallback rows</p>{/if}{#if unseenSummary(table.evidence?.unseen_attribute_levels)}<p class="help-text">Unseen attribute levels: {unseenSummary(table.evidence?.unseen_attribute_levels)}</p>{/if}{#each table.tables || [] as item}<h3>{item.name}</h3><div class="stack-table"><table><thead><tr>{#each item.columns as column}<th>{column.replaceAll('_', ' ')}</th>{/each}</tr></thead><tbody>{#each item.rows.slice(0, 300) as row}<tr>{#each item.columns as column}<td>{typeof row[column] === 'number' ? Number(row[column].toPrecision(7)) : row[column] ?? '—'}</td>{/each}</tr>{/each}</tbody></table></div>{#if item.rows.length > 300}<p>Showing 300 of {item.rows.length} rows. Download the full rating file on Export.</p>{/if}{/each}{#if table.message}<p>{table.message}</p>{/if}{/if}
            {#if table?.editable}<h3>Adjust this lookup</h3><p>Preview the change to the lookup and full model. Apply creates a new named candidate.</p><label>Adjustment<select aria-label="Lookup adjustment" bind:value={operation} onchange={() => (preview = null)}><option value="cap">Cap lookup value</option><option value="round">Round for deployment</option></select></label><div class="form-grid">{#if operation === 'cap'}<label>Minimum lookup value<input aria-label="Minimum lookup value" type="number" step="any" bind:value={lower} oninput={() => (preview = null)} /></label><label>Maximum lookup value<input aria-label="Maximum lookup value" type="number" step="any" bind:value={upper} oninput={() => (preview = null)} /></label>{:else}<label>Decimal places<input aria-label="Rate decimal places" type="number" min="0" bind:value={decimals} oninput={() => (preview = null)} /></label>{/if}</div><label><input type="checkbox" bind:checked={refitLater} onchange={() => (preview = null)} />Refit later stages after applying (otherwise keep them fixed)</label><button onclick={editPreview} disabled={!!busy}>Preview adjustment</button>{#if preview}<p>{preview.changed_rows} rows changed · expected {preview.expected_before} → {preview.expected_after}</p><pre class="stack-json">{JSON.stringify(preview.sample_rows, null, 2)}</pre><label>New candidate name<input aria-label="Amended candidate name" bind:value={amendedName} /></label><button onclick={editApply} disabled={!!busy || !amendedName}>Apply to new candidate</button>{/if}{/if}
        </section>{/if}
        {#if result && view === 'export'}<section class="model-card"><h2>Combined scorer</h2><p>Save the complete fitted stack and inspect its component tables.</p><div class="stack-review-actions"><button onclick={() => download('easyglm')} disabled={!!busy}>Download JSON scorer</button><button onclick={() => download('python_score')} disabled={!!busy}>Download Python scorer</button><button onclick={() => download('xlsx')} disabled={!!busy}>Download Excel component workbook</button></div><h2>Stage rating files</h2>{#each components as c}<div class="stack-line"><span>{c.name} · {labels[c.kind]}</span>{#if c.kind === 'catalogue'}<button onclick={() => download('csv', c.id || c.component_id)} disabled={!!busy}>Download full lookup CSV</button>{:else}<span>{c.kind === 'catboost' ? 'Included in combined scorer' : 'Inspect on Rate tables'}</span>{/if}</div>{/each}</section>{/if}
        {#if !result && view !== 'model'}<section class="model-card"><p>Fit a named stacked candidate to inspect these results.</p><button onclick={() => onNavigate('model')}>Go to Model</button></section>{/if}
    {/if}
</div>
<style>
    .stack-line { display:flex; flex-wrap:wrap; gap:12px; align-items:center; margin:12px 0; }
    .stack-line label { display:grid; gap:5px; }
    .stack-components { padding-left:22px; }
    .stack-components li { display:grid; grid-template-columns: 1fr 1fr 2fr 1fr auto; gap:12px; border-top:1px solid #dae3db; padding:10px 4px; }
    .stack-support { grid-column:1 / -1; color:#526e5d; }
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
    @media (max-width:900px) { .stack-components li { grid-template-columns:1fr 1fr; } }
</style>
