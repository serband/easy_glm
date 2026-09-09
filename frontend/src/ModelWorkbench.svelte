<script>
    import { onDestroy } from 'svelte';
    import ReviewPanel from './ReviewPanel.svelte';
    import RateChart from './RateChart.svelte';
    import DiagnosticTable from './DiagnosticTable.svelte';
    import { rateChartKind } from './rateChartData.js';
    export let comparison = '',
        onContext = () => {};
    let comparisonResult = null,
        comparisonKey = '',
        sentContext = '',
        readingComparison = false;
    $: effectiveChallenger =
        comparison !== selected && jobs[comparison]?.applicable ? comparison : '';
    $: context = JSON.stringify({
        fitted: Object.keys(jobs).filter((n) => jobs[n].applicable),
        selected,
        champion: wb?.champion,
    });
    $: if (context !== sentContext) {
        sentContext = context;
        onContext(JSON.parse(context));
    }
    $: compareKey = `${selected}:${effectiveChallenger}:${state?.revision}:${jobs[effectiveChallenger]?.id}`;
    $: if (loaded && compareKey !== comparisonKey && !readingComparison) loadComparison();
    async function loadComparison() {
        const key = compareKey;
        comparisonKey = key;
        comparisonResult = null;
        readingComparison = true;
        try {
            if (effectiveChallenger) {
                const result = await api('results/' + encodeURIComponent(effectiveChallenger));
                if (key === compareKey) comparisonResult = result;
            }
        } catch (e) {
            error = e.message;
        } finally {
            readingComparison = false;
        }
    }
    async function makeChampion(name) {
        try {
            const response = await api('review/' + encodeURIComponent(name), {
                ...rev(),
                action: 'champion',
            });
            onState(response.snapshot);
            await refresh(true);
            notice = name + ' is the project champion.';
        } catch (e) {
            error = e.message;
        }
    }
    $: metricRows = [result, comparisonResult].filter(Boolean).flatMap((r) =>
        Object.entries(r.metrics).map(([subset, m]) => ({
            model: r.summary.name,
            subset,
            ...m,
        })),
    );
    $: factRows = [result, comparisonResult]
        .filter(Boolean)
        .map((r) => ({ model: r.summary.name, ...r.diagnostic_info?.facts }));
    $: savedVersionRows = [result, comparisonResult]
        .filter(Boolean)
        .flatMap((r) => r.diagnostic_info?.saved_versions || []);
    let diagnosticTab = 'variable';
    let tableDetails = false,
        tableReview,
        reviewBusy = false;
    $: visibleColumns = table
        ? table.columns.filter(
              (c) =>
                  tableDetails ||
                  [
                      'label',
                      'label_a',
                      'label_b',
                      'fitted',
                      'relativity',
                      'exposure',
                      'slope',
                  ].includes(c),
          )
        : [];
    let rowEdits = {};
    function editRow(index, value) {
        rowEdits = { ...rowEdits, [index]: Number(value) };
    }
    function clearEdits() {
        rowEdits = {};
    }
    async function reviewed(snapshot) {
        onState(snapshot);
        resultId = '';
        clearEdits();
        await refresh(false);
    }
    export let api;
    export let state;
    export let view;
    export let onState;
    export let onReady;
    export let onNavigate;
    let wb = null,
        cfg = null,
        selected = '__new__',
        newName = 'Frequency',
        loading = false,
        saving = false,
        loaded = false;
    let error = '',
        notice = '',
        known = '',
        baseline = '',
        splitDraft = null,
        jobs = {},
        result = null,
        resultId = '';
    let mode = 'fixed',
        fixedAlpha = 0.001,
        cv = 5,
        pathLength = 20,
        l1 = 1,
        nBins = 20,
        levelShare = 0.0025,
        kinds = {};
    let termSearch = '',
        termScroll = 0,
        subset = 'holdout',
        tableName = '',
        table = null,
        tableOffset = 0,
        tableScroll = 0,
        tableBusy = false;
    let polling = false,
        timer = null;
    $: signature = state ? state.session_id + ':' + state.revision : '';
    $: if (
        (['model', 'diagnostics', 'compare', 'tables'].includes(view) || state?.models?.length) &&
        state &&
        !loading &&
        !saving &&
        (!loaded || known !== signature)
    )
        refresh(true);
    $: if (loaded && known !== signature && !loading && !saving) refresh(true);
    $: payload = cfg
        ? {
              fields: {
                  family: cfg.family,
                  link: cfg.link,
                  target: cfg.target,
                  weight: cfg.weight,
                  offset: cfg.offset,
                  divide_target_by_weight: cfg.divide_target_by_weight,
                  predictors: cfg.predictors,
                  tweedie_power: cfg.tweedie_power,
                  base: cfg.base,
                  penalty: {
                      alpha: mode === 'fixed' ? fixedAlpha : null,
                      cv: mode === 'cv' ? cv : null,
                      n_alphas: pathLength,
                      l1_ratio: l1,
                  },
              },
              designs: kinds,
              n_bins: nBins,
              min_level_share: levelShare,
          }
        : null;
    $: dirty = payload && JSON.stringify(payload) !== baseline;
    $: active = Object.values(jobs).some((j) => ['queued', 'running'].includes(j.status));
    $: job = jobs[selected];
    $: applicable = Boolean(job?.applicable);
    $: onReady(applicable);
    $: filteredTerms = wb
        ? wb.predictors.filter((n) => n.toLowerCase().includes(termSearch.toLowerCase()))
        : [];
    $: termStart = Math.max(0, Math.floor(termScroll / 34) - 3);
    $: tableStart = Math.max(0, Math.floor(tableScroll / 32) - 3);
    $: lift = result?.lift?.[subset] || [];
    $: maxRate = Math.max(
        0.000001,
        ...lift.flatMap((r) => [r.actual_rate || 0, r.expected_rate || 0]),
    );
    const empty = {
        family: 'poisson',
        link: null,
        target: null,
        weight: null,
        offset: null,
        divide_target_by_weight: false,
        predictors: [],
        tweedie_power: 1.5,
        base: 'modal',
        penalty: { alpha: 0.001, cv: null, n_alphas: 20, l1_ratio: 1 },
    };
    function rev() {
        return { session_id: state.session_id, revision: state.revision };
    }
    function pickModel() {
        cfg = structuredClone(
            wb.models[selected] || {
                ...empty,
                target: wb.target,
                weight: wb.weight,
                offset: wb.offset,
                divide_target_by_weight: Boolean(wb.weight),
                predictors: [...wb.predictors],
            },
        );
        mode = cfg.penalty.alpha === null ? 'cv' : 'fixed';
        fixedAlpha = cfg.penalty.alpha ?? 0.001;
        cv = cfg.penalty.cv ?? 5;
        pathLength = cfg.penalty.n_alphas;
        l1 = cfg.penalty.l1_ratio;
        nBins = wb.design.defaults.n_bins;
        levelShare = wb.design.defaults.min_level_share;
        kinds = Object.fromEntries(
            wb.predictors.map((n) => [n, wb.design.variables[n]?.kind || null]),
        );
        baseline = JSON.stringify({
            fields: {
                family: cfg.family,
                link: cfg.link,
                target: cfg.target,
                weight: cfg.weight,
                offset: cfg.offset,
                divide_target_by_weight: cfg.divide_target_by_weight,
                predictors: cfg.predictors,
                tweedie_power: cfg.tweedie_power,
                base: cfg.base,
                penalty: {
                    alpha: mode === 'fixed' ? fixedAlpha : null,
                    cv: mode === 'cv' ? cv : null,
                    n_alphas: pathLength,
                    l1_ratio: l1,
                },
            },
            designs: kinds,
            n_bins: nBins,
            min_level_share: levelShare,
        });
        result = null;
        resultId = '';
        table = null;
        error = '';
        notice = '';
        if (jobs[selected]?.applicable) void loadResults();
    }
    async function refresh(keep = true) {
        loading = true;
        if (known && known !== signature) {
            jobs = {};
            result = null;
            onReady(false);
        }
        const keepDraft = keep && dirty;
        try {
            wb = await api('workbench');
            jobs = wb.jobs;
            known = wb.session_id + ':' + wb.revision;
            loaded = true;
            splitDraft = structuredClone(wb.split);
            if (selected !== '__new__' && !wb.models[selected])
                selected = Object.keys(wb.models)[0] || '__new__';
            if (!cfg && Object.keys(wb.models).length)
                selected =
                    wb.champion && wb.models[wb.champion] ? wb.champion : Object.keys(wb.models)[0];
            if (!keepDraft || !cfg) pickModel();
            else
                notice =
                    'Applied settings changed. Your model draft is kept; save it to validate against the current variables.';
            if (!timer) timer = setInterval(poll, 500);
        } catch (e) {
            error = e.message;
        } finally {
            loading = false;
        }
    }
    async function poll() {
        if (polling) return;
        polling = true;
        try {
            jobs = await api('jobs');
            if (jobs[selected]?.applicable && resultId !== jobs[selected].id) await loadResults();
        } catch (e) {
            error = e.message;
        } finally {
            polling = false;
        }
    }
    async function loadResults() {
        const name = selected,
            id = jobs[name]?.id;
        if (!jobs[name]?.applicable) return;
        try {
            const data = await api('results/' + encodeURIComponent(name));
            if (name !== selected) return;
            result = data;
            resultId = id;
            if (!result.metrics[subset]) subset = Object.keys(result.metrics)[0];
            tableName = result.table_index.some((t) => t.name === tableName)
                ? tableName
                : result.table_index[0]?.name || '';
            tableOffset = 0;
            await loadTable();
        } catch (e) {
            error = e.message;
        }
    }
    async function saveModel() {
        saving = true;
        error = '';
        try {
            const name = selected === '__new__' ? newName : selected;
            const snapshot = await api('models/save', {
                ...rev(),
                name,
                create: selected === '__new__',
                ...payload,
            });
            selected = name;
            onState(snapshot);
            await refresh(false);
            notice = 'Model settings saved. Fit when ready.';
        } catch (e) {
            error = e.message;
        } finally {
            saving = false;
        }
    }
    async function saveSplit() {
        saving = true;
        error = '';
        try {
            const snapshot = await api('split', { ...rev(), ...splitDraft });
            onState(snapshot);
            await refresh(true);
            notice = 'Split applied.';
        } catch (e) {
            error = e.message;
        } finally {
            saving = false;
        }
    }
    async function fit() {
        saving = true;
        error = '';
        notice = '';
        try {
            await api('models/' + encodeURIComponent(selected) + '/fit', rev());
            result = null;
            resultId = '';
            await poll();
        } catch (e) {
            error = e.message;
        } finally {
            saving = false;
        }
    }
    async function cancel() {
        try {
            jobs = await api('models/' + encodeURIComponent(selected) + '/cancel', rev());
        } catch (e) {
            error = e.message;
        }
    }
    function toggle(name, checked) {
        cfg = {
            ...cfg,
            predictors: checked
                ? [...cfg.predictors, name]
                : cfg.predictors.filter((n) => n !== name),
        };
    }
    function kind(name, value) {
        kinds = { ...kinds, [name]: value || null };
    }
    async function loadTable() {
        rowEdits = {};
        if (!tableName || !result) return;
        tableBusy = true;
        tableScroll = 0;
        try {
            table = await api(
                'results/' +
                    encodeURIComponent(selected) +
                    '/table?variable=' +
                    encodeURIComponent(tableName) +
                    '&offset=' +
                    tableOffset +
                    '&limit=200',
            );
            table = { ...table, kind: rateChartKind(table, tableName, wb) };
        } catch (e) {
            error = e.message;
        } finally {
            tableBusy = false;
        }
    }
    function num(value, digits = 5) {
        return value === null || value === undefined
            ? '—'
            : typeof value === 'number'
              ? Math.abs(value) > 0 && Math.abs(value) < 0.00001
                  ? value.toExponential(2)
                  : value.toLocaleString(undefined, { maximumSignificantDigits: digits })
              : String(value);
    }
    function points(field) {
        return lift
            .map(
                (row, i) =>
                    45 +
                    (i * 900) / Math.max(1, lift.length - 1) +
                    ',' +
                    (180 - ((row[field] || 0) / maxRate) * 150),
            )
            .join(' ');
    }
    onDestroy(() => {
        if (timer) clearInterval(timer);
    });
</script>

<div hidden={!['model', 'diagnostics', 'compare', 'tables'].includes(view)} class="model-workbench">
    <div class="heading">
        <div>
            <div class="eyebrow">{view === 'model' ? 'MODEL SETUP' : 'FITTED RESULTS'}</div>
            <h1>
                {view === 'model'
                    ? 'Model design and fit'
                    : view === 'diagnostics'
                      ? 'Diagnostics'
                      : view === 'compare'
                        ? 'Compare'
                        : 'Rate tables'}
            </h1>
            <p>
                {view === 'model'
                    ? 'Configure the model, prepare the split and fit in the background.'
                    : 'Results from the current applied model and full prepared data.'}
            </p>
        </div>
    </div>
    {#if error}<div class="message error" role="alert">
            <span>{error}</span><button onclick={() => refresh(true)} disabled={loading}
                >Refresh model status</button
            >
        </div>{/if}
    {#if notice}<div class="message" role="status">{notice}</div>{/if}
    {#if !wb || !cfg}<div class="loading">Checking applied data and model settings…</div>{:else}
        <div class="model-selector">
            <label
                >Model<select
                    aria-label="Model selection"
                    bind:value={selected}
                    onchange={pickModel}
                    ><option value="__new__">Create a new model</option
                    >{#each Object.keys(wb.models) as name}<option value={name}>{name}</option
                        >{/each}</select
                ></label
            >{#if selected === '__new__'}<label
                    >Name<input aria-label="New model name" bind:value={newName} /></label
                >{/if}
            <div class="spacer"></div>
            <span class="edit-status"
                >{dirty
                    ? 'Unsaved model changes'
                    : selected === '__new__'
                      ? 'New definition'
                      : 'Applied model settings'}</span
            >
        </div>
        {#if view !== 'model' && selected !== '__new__' && (view === 'compare' || Object.values(jobs).filter((j) => j.applicable).length > 1)}<div
                class="results-toolbar comparison-context"
            >
                <label
                    >Compare with (challenger)<select
                        aria-label="Compare with challenger"
                        bind:value={comparison}
                        ><option value="">None</option
                        >{#each Object.keys(jobs).filter((n) => n !== selected && jobs[n].applicable) as name}<option
                                value={name}>{name}</option
                            >{/each}</select
                    ></label
                >
                <span>Project champion: <b>{wb.champion || 'Not designated'}</b></span>
                <button
                    disabled={!applicable || wb.champion === selected}
                    onclick={() => makeChampion(selected)}>Make selected model champion</button
                >
            </div>{/if}
        {#if view === 'model'}
            <nav class="section-nav" aria-label="Model sections">
                <a href="#model-definition">Model definition</a><a href="#factor-design"
                    >Factor design</a
                ><a href="#fit-settings">Fit and results</a>
            </nav>
            <fieldset disabled={saving} class="edit-fieldset">
                <section class="model-card">
                    <h2 id="model-definition">Model definition</h2>
                    <div class="form-grid">
                        <label
                            >Family<select aria-label="Model family" bind:value={cfg.family}
                                >{#each wb.families as family}<option value={family}
                                        >{family.replaceAll('_', ' ')}</option
                                    >{/each}</select
                            ></label
                        >
                        <label
                            >Link<select aria-label="Model link" bind:value={cfg.link}
                                ><option value={null}>Family default</option><option value="log"
                                    >Log</option
                                ><option value="logit">Logit</option></select
                            ></label
                        >
                        {#each ['target', 'weight', 'offset'] as field}<label
                                >{field.charAt(0).toUpperCase() + field.slice(1)}<select
                                    aria-label={'Model ' + field}
                                    bind:value={cfg[field]}
                                    ><option value={null}>None</option
                                    >{#if cfg[field] && !wb.columns.some((c) => c.name === cfg[field])}<option
                                            value={cfg[field]}
                                            >{cfg[field]} (derived or missing)</option
                                        >{/if}{#each wb.columns.filter((c) => c.numeric) as column}<option
                                            value={column.name}>{column.name}</option
                                        >{/each}</select
                                ></label
                            >{/each}
                        {#if cfg.family === 'tweedie'}<label
                                >Tweedie power<input
                                    aria-label="Tweedie power"
                                    type="number"
                                    min="1.01"
                                    max="1.99"
                                    step=".05"
                                    bind:value={cfg.tweedie_power}
                                /></label
                            >{/if}
                        <label class="check-label"
                            ><input
                                aria-label="Divide target by weight"
                                type="checkbox"
                                bind:checked={cfg.divide_target_by_weight}
                            />Divide target by weight</label
                        >
                    </div>
                </section>
                <section class="model-card">
                    <div class="section-heading">
                        <h2 id="factor-design">Factor design</h2>
                        <span>{cfg.predictors.length} selected</span>
                    </div>
                    <p class="help-text">
                        Roles make columns eligible; this selection defines this model. Design kinds
                        and defaults are shared across models.
                    </p>
                    <details>
                        <summary>Defaults for every predictor</summary>
                        <div class="form-grid compact">
                            <label
                                >Default bins<input
                                    aria-label="Default bins"
                                    type="number"
                                    min="2"
                                    max="200"
                                    bind:value={nBins}
                                /></label
                            ><label
                                >Minimum category share<input
                                    aria-label="Minimum category share"
                                    type="number"
                                    min="0"
                                    max=".99"
                                    step=".001"
                                    bind:value={levelShare}
                                /></label
                            >
                        </div>
                    </details>
                    <input
                        class="factor-search"
                        aria-label="Search predictor terms"
                        placeholder="Find a predictor…"
                        bind:value={termSearch}
                        oninput={() => (termScroll = 0)}
                    />
                    <div
                        class="terms-list"
                        onscroll={(e) => (termScroll = e.currentTarget.scrollTop)}
                    >
                        <div style:height={filteredTerms.length * 34 + 'px'} class="virtual-space">
                            <div
                                class="virtual-rows"
                                style:transform={'translateY(' + termStart * 34 + 'px)'}
                            >
                                {#each filteredTerms.slice(termStart, termStart + 18) as name}<div
                                        class="term-row"
                                    >
                                        <label
                                            ><input
                                                aria-label={'Include ' + name}
                                                type="checkbox"
                                                checked={cfg.predictors.includes(name)}
                                                onchange={(e) =>
                                                    toggle(name, e.currentTarget.checked)}
                                            />{name}</label
                                        ><select
                                            aria-label={'Design kind for ' + name}
                                            value={kinds[name] || ''}
                                            onchange={(e) => kind(name, e.currentTarget.value)}
                                            ><option value="">Infer from type</option><option
                                                value="step">Step bands</option
                                            ><option value="linear">Piecewise linear</option><option
                                                value="continuous">Continuous slope</option
                                            ><option value="categorical">Categorical</option
                                            ></select
                                        >
                                    </div>{/each}
                            </div>
                        </div>
                    </div>
                    {#if cfg.interactions?.length}<div class="preserved">
                            Retained two-stage interactions: {cfg.interactions
                                .map((i) => i.a + ' × ' + i.b)
                                .join(', ')}. Their detailed editor is not yet available here.
                        </div>{/if}
                    {#if Object.keys(cfg.monotone || {}).length}<div class="preserved">
                            Retained monotone constraints: {Object.entries(cfg.monotone)
                                .map(([n, v]) => n + ' ' + v)
                                .join(', ')}.
                        </div>{/if}
                    <div class="help-text">
                        Existing knots, clamps, per-term penalties, adjustments and notes are
                        retained. Unsupported combinations are reported when saving or fitting.
                    </div>
                </section>
                <section class="model-card">
                    <h2 id="fit-settings">Fit and results</h2>
                    <details class="split-settings" open={!wb.counts.holdout}>
                        <summary>Train / holdout split</summary>
                        <div class="section-heading">
                            <span>Applied split</span>
                            <span
                                >{wb.counts.train.toLocaleString()} train · {wb.counts.holdout.toLocaleString()}
                                holdout</span
                            >
                        </div>
                        <p class="help-text">
                            A generated random split is valid without a source column assigned the
                            split role.
                        </p>
                        <div class="form-grid split-form">
                            <label
                                >Method<select
                                    aria-label="Split method"
                                    bind:value={splitDraft.mode}
                                    ><option value="random">Seeded random</option><option
                                        value="column">Existing column</option
                                    ></select
                                ></label
                            >
                            {#if splitDraft.mode === 'random'}<label
                                    >Generated column<input
                                        aria-label="Split column name"
                                        bind:value={splitDraft.column}
                                    /></label
                                ><label
                                    >Training fraction<input
                                        aria-label="Training fraction"
                                        type="number"
                                        min=".01"
                                        max=".99"
                                        step=".05"
                                        bind:value={splitDraft.fraction}
                                    /></label
                                ><label
                                    >Seed<input
                                        aria-label="Split seed"
                                        type="number"
                                        min="0"
                                        step="1"
                                        bind:value={splitDraft.seed}
                                    /></label
                                >{:else}<label
                                    >Column<select
                                        aria-label="Existing split column"
                                        bind:value={splitDraft.column}
                                        >{#if !wb.columns.some((c) => c.name === splitDraft.column)}<option
                                                value={splitDraft.column}
                                                >{splitDraft.column} (missing)</option
                                            >{/if}{#each wb.columns as column}<option
                                                value={column.name}>{column.name}</option
                                            >{/each}</select
                                    ></label
                                ><label
                                    >Training value<input
                                        aria-label="Training value"
                                        bind:value={splitDraft.train_value}
                                    /></label
                                >{/if}
                            <button onclick={saveSplit} disabled={saving || loading}
                                >Apply split</button
                            >
                        </div>
                        {#if wb.problems.length}<div class="prerequisites">
                                <strong>Before fitting</strong>{#each wb.problems as problem}<p>
                                        {problem}
                                    </p>{/each}
                            </div>{/if}
                    </details>
                    <h3>Fit settings</h3>
                    <div class="form-grid">
                        <label
                            >Penalty<select aria-label="Penalty mode" bind:value={mode}
                                ><option value="fixed">Fixed alpha</option><option value="cv"
                                    >Cross-validated alpha</option
                                ></select
                            ></label
                        >
                        {#if mode === 'fixed'}<label
                                >Alpha<input
                                    aria-label="Fixed alpha"
                                    type="number"
                                    min=".000001"
                                    step=".001"
                                    bind:value={fixedAlpha}
                                /></label
                            >{:else}<label
                                >CV folds<input
                                    aria-label="CV folds"
                                    type="number"
                                    min="2"
                                    max="10"
                                    bind:value={cv}
                                /></label
                            ><label
                                >Alphas on path<input
                                    aria-label="Alpha path length"
                                    type="number"
                                    min="2"
                                    max="100"
                                    bind:value={pathLength}
                                /></label
                            >{/if}
                        <label
                            >L1 ratio<input
                                aria-label="L1 ratio"
                                type="number"
                                min="0"
                                max="1"
                                step=".1"
                                bind:value={l1}
                            /></label
                        >
                        <label
                            >Table base<select aria-label="Table base" bind:value={cfg.base}
                                ><option value="modal">Most common risk</option><option
                                    value="reference">GLM reference risk</option
                                ></select
                            ></label
                        >
                    </div>
                    <p class="help-text">
                        L1 ratio 1 is lasso; 0 is ridge. Save changes before fitting.
                    </p>
                    <div class="model-actions">
                        <button class="primary" onclick={saveModel}
                            >{selected === '__new__'
                                ? 'Create model'
                                : 'Save model settings'}</button
                        ><button onclick={() => pickModel()}>Reset model draft</button>
                        <div class="spacer"></div>
                        <button
                            class="primary"
                            onclick={fit}
                            disabled={selected === '__new__' ||
                                dirty ||
                                active ||
                                wb.problems.length > 0}>Fit model</button
                        >
                    </div>
                    {#if dirty}<p class="help-text">Save the model changes before fitting.</p>{/if}
                </section>
            </fieldset>
        {/if}
        {#if job}<section
                class="job-card"
                class:compact-result-status={view !== 'model' && job.status === 'complete'}
                role="status"
            >
                <div>
                    <strong
                        >{job.status === 'complete'
                            ? 'Fit complete'
                            : job.status === 'stale'
                              ? 'Fit needs updating'
                              : job.status === 'failed'
                                ? 'Fit failed'
                                : job.status === 'cancelled'
                                  ? 'Fit cancelled'
                                  : 'Fitting in background'}</strong
                    >
                    <p>{job.message}</p>
                </div>
                <span>{num(job.elapsed, 3)} s</span
                >{#if ['queued', 'running'].includes(job.status)}<button onclick={cancel}
                        >Cancel fit</button
                    >{/if}{#if applicable && view !== 'diagnostics'}<button
                        onclick={() => onNavigate('diagnostics')}>View diagnostics</button
                    >{/if}{#if applicable && view !== 'tables'}<button
                        onclick={() => onNavigate('tables')}>View rate tables</button
                    >{/if}
            </section>{/if}
        {#if ['diagnostics', 'compare', 'tables'].includes(view)}
            {#if !applicable}<div class="message">
                    A completed fit matching the current applied settings is required. Open Model
                    and fit the model.
                </div>{:else if !result}<p>
                    Loading fitted results…
                </p>{:else if view === 'diagnostics' || view === 'compare'}
                <div class="results-toolbar">
                    <label
                        >Data subset<select aria-label="Diagnostic subset" bind:value={subset}
                            >{#each Object.keys(result.metrics) as name}<option value={name}
                                    >{{ train: 'Training', holdout: 'Holdout', all: 'All rows' }[
                                        name
                                    ]}</option
                                >{/each}</select
                        ></label
                    ><span
                        >Fit alpha {num(result.summary.alpha)} · {result.summary.features} features ·
                        {result.summary.non_zero} nonzero{#if result.summary.alpha_stage2 !== null}
                            · stage 2 alpha {num(result.summary.alpha_stage2)}{/if}</span
                    >
                </div>
                <div class="metrics-grid">
                    {#each [['ae', 'Actual / expected'], ['gini', 'Normalised Gini'], ['deviance_explained', 'Deviance explained'], ['mean_deviance', 'Mean deviance']] as [key, label]}<div
                        >
                            <span>{label}</span><strong>{num(result.metrics[subset]?.[key])}</strong
                            >
                        </div>{/each}
                </div>
                <div class="result-totals">
                    {#each ['rows', 'exposure', 'actual', 'expected'] as key}<span
                            >{key}: <b>{num(result.metrics[subset]?.[key], 7)}</b></span
                        >{/each}
                </div>
                <p class="help-text">{result.diagnostic_info?.gini_note || ''}</p>
                <details class="model-card" open={view === 'compare'}>
                    <summary>Metrics and model facts side by side</summary><DiagnosticTable
                        rows={metricRows}
                        title="Metrics by model and subset"
                    /><DiagnosticTable
                        rows={factRows}
                        title="Model facts"
                    />{#if savedVersionRows.length}<DiagnosticTable
                            rows={savedVersionRows}
                            title="Saved versions of the rate tables"
                        />{/if}{#if comparisonResult && comparisonResult.summary.family !== result.summary.family}<p
                        >
                            Deviances from different families are not directly comparable.
                        </p>{/if}
                </details>
                <div class="workflow-tabs" role="tablist" aria-label="Diagnostics views">
                    {#each [['variable', 'A/E by variable'], ['pair', 'A/E by pair'], ['lift', 'Lift'], ['double_lift', 'Double lift'], ['residual', 'Residual factors'], ['path', 'Regularisation path'], ['coefficients', 'Coefficients'], ['compare', 'Relativities that differ']] as [key, label]}<button
                            role="tab"
                            aria-selected={diagnosticTab === key}
                            onclick={() => (diagnosticTab = key)}>{label}</button
                        >{/each}
                </div>
                {#if result.dropped_predictors.length}<div class="message">
                        Constant/all-null predictors omitted by the engine: {result.dropped_predictors.join(
                            ', ',
                        )}.
                    </div>{/if}
                <div>
                    <ReviewPanel
                        {diagnosticTab}
                        challenger={effectiveChallenger}
                        {api}
                        {state}
                        name={selected}
                        view={view === 'compare' ? 'diagnostics' : view}
                        {subset}
                        onApplied={reviewed}
                        onClear={clearEdits}
                        {onNavigate}
                    />
                </div>
                {#each result.warnings as warning}<div class="message">{warning}</div>{/each}
            {:else}
                <div class="results-toolbar">
                    <label
                        >Variable<select
                            aria-label="Rate table variable"
                            disabled={Object.keys(rowEdits).length > 0}
                            bind:value={tableName}
                            onchange={() => {
                                tableOffset = 0;
                                loadTable();
                            }}
                            >{#each result.table_index as item}<option value={item.name}
                                    >{item.name} · {item.rows} rows</option
                                >{/each}</select
                        ></label
                    ><strong>Base rate {num(result.base_rate, 8)}</strong><span
                        >{result.link} link · {result.relativity_label}</span
                    >
                </div>
                {#if table}<div class="rate-primary">
                        <RateChart {table} variable={tableName} label={result.relativity_label} />
                        <details class="rate-table-card" aria-label="Editable rate table">
                            <summary>Rate table</summary>
                            <div class="table-heading">
                                <label
                                    ><input type="checkbox" bind:checked={tableDetails} /> All columns</label
                                >
                            </div>
                            <div
                                class="rate-grid"
                                onscroll={(e) => (tableScroll = e.currentTarget.scrollTop)}
                            >
                                <table>
                                    <thead
                                        ><tr
                                            >{#each visibleColumns as column}<th
                                                    >{column === 'relativity'
                                                        ? 'Current relativity'
                                                        : column === 'label'
                                                          ? 'Band / level'
                                                          : column.replaceAll('_', ' ')}</th
                                                >{/each}</tr
                                        ></thead
                                    ><tbody
                                        >{#if tableStart > 0}<tr
                                                aria-hidden="true"
                                                style:height={tableStart * 32 + 'px'}
                                                ><td colspan={visibleColumns.length}></td></tr
                                            >{/if}{#each table.rows.slice(tableStart, tableStart + 24) as row, rowIndex}<tr
                                                >{#each visibleColumns as column}<td
                                                        title={String(row[column] ?? '')}
                                                        >{#if column === 'relativity'}<input
                                                                class="relativity-input"
                                                                aria-label={'Relativity row ' +
                                                                    (table.offset +
                                                                        tableStart +
                                                                        rowIndex +
                                                                        1)}
                                                                type="number"
                                                                min="0.000000001"
                                                                step="any"
                                                                value={rowEdits[
                                                                    table.offset +
                                                                        tableStart +
                                                                        rowIndex
                                                                ] ?? row[column]}
                                                                onchange={(e) =>
                                                                    editRow(
                                                                        table.offset +
                                                                            tableStart +
                                                                            rowIndex,
                                                                        e.currentTarget.value,
                                                                    )}
                                                            />{:else}{num(row[column], 8)}{/if}</td
                                                    >{/each}</tr
                                            >{/each}{#if tableStart + 24 < table.rows.length}<tr
                                                aria-hidden="true"
                                                style:height={(table.rows.length -
                                                    tableStart -
                                                    24) *
                                                    32 +
                                                    'px'}
                                                ><td colspan={visibleColumns.length}></td></tr
                                            >{/if}</tbody
                                    >
                                </table>
                            </div>
                            <div class="results-toolbar table-paging">
                                <span
                                    >Rows {table.offset + 1}–{Math.min(
                                        table.offset + table.rows.length,
                                        table.total,
                                    )} of {table.total}</span
                                >
                                <div class="spacer"></div>
                                <button
                                    disabled={tableBusy ||
                                        Object.keys(rowEdits).length > 0 ||
                                        tableOffset === 0}
                                    onclick={() => {
                                        tableOffset = Math.max(0, tableOffset - 200);
                                        loadTable();
                                    }}>Previous rows</button
                                ><button
                                    disabled={tableBusy ||
                                        Object.keys(rowEdits).length > 0 ||
                                        tableOffset + 200 >= table.total}
                                    onclick={() => {
                                        tableOffset += 200;
                                        loadTable();
                                    }}>Next rows</button
                                >
                            </div>
                            <div class="table-edit-actions">
                                <button
                                    class="primary"
                                    disabled={reviewBusy || !Object.keys(rowEdits).length}
                                    onclick={() => tableReview.previewRowEdits()}
                                    >Preview row edits ({Object.keys(rowEdits).length})</button
                                ><button
                                    disabled={reviewBusy || !Object.keys(rowEdits).length}
                                    onclick={clearEdits}>Discard row edits</button
                                >
                            </div>
                        </details>
                    </div>{/if}
                <p class="help-text">
                    {result.relativity_note} The chart shows applied table values; row edits remain drafts
                    until applied.
                </p>
                <div class="results-toolbar">
                    <label
                        >A/E subset<select aria-label="Table diagnostic subset" bind:value={subset}
                            ><option value="train">Training</option><option value="holdout"
                                >Holdout</option
                            ></select
                        ></label
                    >
                </div>
                <ReviewPanel
                    challenger={effectiveChallenger}
                    bind:this={tableReview}
                    bind:busy={reviewBusy}
                    {api}
                    {state}
                    name={selected}
                    {view}
                    {subset}
                    variable={tableName}
                    edits={rowEdits}
                    onApplied={reviewed}
                    onClear={clearEdits}
                    {onNavigate}
                />
            {/if}
        {/if}
    {/if}
</div>
