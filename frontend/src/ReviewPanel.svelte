<script>
    import { onDestroy } from 'svelte';
    import DiagnosticPlot from './DiagnosticPlot.svelte';
    import DiagnosticTable from './DiagnosticTable.svelte';
    export let api,
        state,
        name,
        view,
        subset = 'train',
        diagnosticTab = 'variable',
        challenger = '',
        variable = '',
        edits = {},
        onApplied,
        onClear,
        onNavigate;
    let activeDiagnosticTab = 'variable',
        inspectedResidual = false;
    let bins = 10,
        tolerance = 0.01,
        kept = true,
        analysis = null,
        aeSets = [],
        aeKind = 'numeric',
        selectedFactors = [],
        pairMetric = 'ae';
    $: if (view === 'diagnostics' && diagnosticTab !== activeDiagnosticTab && !busy) {
        activeDiagnosticTab = diagnosticTab;
        inspectedResidual = false;
        analysis = null;
        runTab();
    }
    function runTab() {
        if (diagnosticTab === 'variable' && diagnosticVariable)
            return run('variable', { variable: diagnosticVariable });
        if (diagnosticTab === 'pair' && a && b && a !== b) return run('pair', { a, b });
        if (['lift', 'double_lift', 'path', 'coefficients', 'compare'].includes(diagnosticTab))
            return run(diagnosticTab, { options: { kept } });
    }
    let info = { variables: [], snapshots: [], undo: false, redo: false };
    $: temporaryBins = (diagnosticTab === 'pair' ? [a, b] : [diagnosticVariable]).some((v) =>
        info.variable_info?.some((x) => x.name === v && x.numeric && !x.kind),
    );
    let shownTitle = '',
        shownSubset = '';
    let diagnosticVariable = '',
        a = '',
        b = '',
        rows = [],
        searchRows = [],
        searchKind = '',
        error = '',
        note = '';
    export let busy = false;
    let taskId = '',
        preview = null,
        loadedKey = '',
        destroyed = false;
    let tool = 'moving',
        windowSize = 3,
        direction = 'increasing',
        ordered = false,
        floor = null,
        cap = 2,
        rounding = 'decimals',
        decimals = 2,
        step = 0.05,
        snapshotName = '',
        chosenSnapshot = '';
    $: selectedVariable = view === 'tables' ? variable : diagnosticVariable;
    $: key = `${name}:${state.session_id}:${state.revision}:${view}:${subset}:${variable}:${challenger}`;
    $: if (name && key !== loadedKey && !busy) load();
    $: series = [
        { key: 'actual_rate', label: 'Actual', color: '#287762' },
        { key: 'fitted_rate', label: 'Original fitted', color: '#737e9b' },
        { key: 'challenger_rate', label: challenger || 'Challenger', color: '#439da5' },
        { key: 'before_rate', label: 'Before preview', color: '#a27c48' },
        {
            key: 'expected_rate',
            label: preview ? 'Preview adjusted' : 'Current adjusted',
            color: '#c35b48',
        },
    ].filter((s) => rows.some((r) => Number.isFinite(r[s.key])));
    $: chartRows = rows.filter((r) => r.exposure > 0);
    $: maxValue = Math.max(0.000001, ...chartRows.flatMap((r) => series.map((s) => r[s.key] || 0)));
    $: pairRows = rows.length && 'label_a' in rows[0];
    $: pairA = [...new Set(rows.map((r) => r.label_a))];
    $: pairB = [...new Set(rows.map((r) => r.label_b))];
    function num(v) {
        return typeof v === 'number'
            ? v.toLocaleString(undefined, { maximumSignificantDigits: 6 })
            : (v ?? '—');
    }
    function rev() {
        return { session_id: state.session_id, revision: state.revision };
    }
    function points(field) {
        return chartRows
            .map(
                (r, i) =>
                    `${50 + (i * 900) / Math.max(1, chartRows.length - 1)},${190 - ((r[field] || 0) / maxValue) * 160}`,
            )
            .join(' ');
    }
    function heat(value) {
        if (!Number.isFinite(value) || value <= 0) return '#e7ece9';
        const strength = Math.min(0.75, Math.abs(Math.log(value)));
        return value > 1
            ? `rgba(194,74,60,${0.15 + strength})`
            : `rgba(39,119,98,${0.15 + strength})`;
    }
    async function load() {
        const startedKey = key;
        loadedKey = startedKey;
        error = '';
        preview = null;
        searchRows = [];
        rows = [];
        busy = true;
        try {
            info = await api('review-info/' + encodeURIComponent(name));
            if (!info.variables.includes(diagnosticVariable))
                diagnosticVariable = info.variables[0] || '';
            if (!info.variables.includes(a)) a = info.variables[0] || '';
            if (!info.variables.includes(b)) b = info.variables[1] || '';
            if (!info.snapshots.includes(chosenSnapshot)) chosenSnapshot = info.snapshots[0] || '';
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
        if (destroyed || key !== startedKey) return;
        if (view === 'diagnostics') {
            if (diagnosticTab === 'residual') inspectedResidual = false;
            else await runTab();
        } else if (variable) await run('variable', { variable });
    }
    export function previewRowEdits() {
        return run('edit', { edits });
    }
    async function run(action, extra = {}) {
        if (busy || destroyed) return;
        busy = true;
        error = '';
        note = '';
        preview = null;
        const started = key;
        try {
            const response = await api('review/' + encodeURIComponent(name), {
                ...rev(),
                action,
                variable: selectedVariable || null,
                subset,
                challenger: challenger || null,
                n_bins: Number(bins),
                tolerance: Number(tolerance),
                options: { both_subsets: view === 'diagnostics' },
                ...extra,
            });
            if (response.snapshot) {
                onApplied(response.snapshot);
                note = 'Snapshot saved in the project.';
                return;
            }
            taskId = response.id;
            if (destroyed || started !== key) {
                await api('reviews/' + taskId + '/cancel', rev());
                return;
            }
            while (!destroyed) {
                const response = await api('reviews/' + taskId);
                if (response.status === 'complete') {
                    if (started !== key) return;
                    const data = response.data;
                    note = data.note || '';
                    if (
                        ['lift', 'double_lift', 'path', 'coefficients', 'compare'].includes(action)
                    ) {
                        analysis = data;
                        rows = [];
                    } else if (action === 'factors' || action === 'interactions') {
                        searchRows = data.rows;
                        searchKind = action;
                        selectedFactors =
                            action === 'factors'
                                ? data.rows.filter((r) => r.signal >= 2).map((r) => r.variable)
                                : [];
                    } else {
                        rows = data.rows || [];
                        aeSets = data.ae_sets || [];
                        aeKind = data.kind || 'numeric';
                        analysis = null;
                        if (diagnosticTab === 'residual') inspectedResidual = true;
                        shownTitle =
                            action === 'pair'
                                ? `${extra.a || a} × ${extra.b || b}`
                                : extra.variable || selectedVariable;
                        shownSubset = { train: 'Training', holdout: 'Holdout', all: 'All rows' }[
                            data.subset || extra.subset || subset
                        ];
                        if (response.can_apply) preview = { ...data, id: taskId };
                    }
                    break;
                }
                if (response.status === 'failed')
                    throw new Error(response.data.error || 'Review could not finish.');
                if (response.status === 'stale')
                    throw new Error('Applied settings changed. Run this review again.');
                if (response.status === 'cancelled') {
                    note = 'Review cancelled.';
                    break;
                }
                await new Promise((resolve) => setTimeout(resolve, 250));
            }
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
            taskId = '';
        }
    }
    async function cancel() {
        if (taskId) await api('reviews/' + taskId + '/cancel', rev());
    }
    async function applyPreview() {
        busy = true;
        error = '';
        try {
            const snapshot = await api('reviews/' + preview.id + '/apply', rev());
            onClear();
            preview = null;
            onApplied(snapshot);
            note = 'Adjustments applied without refitting.';
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function previewTool() {
        let options = {};
        if (tool === 'moving') options = { window: windowSize, ordered };
        if (tool === 'isotonic') options = { direction, ordered };
        if (tool === 'cap') options = { floor: floor || null, cap: cap || null };
        if (tool === 'round') options = rounding === 'decimals' ? { decimals } : { step };
        run(tool, { options });
    }
    async function include(row) {
        busy = true;
        error = '';
        try {
            const response = await api('review/' + encodeURIComponent(name), {
                ...rev(),
                action: row.variables
                    ? 'include_factors'
                    : searchKind === 'factors'
                      ? 'include_factor'
                      : 'include_pair',
                variables: row.variables || [],
                variable: row.variable || null,
                a: row.a || null,
                b: row.b || null,
            });
            onApplied(response.snapshot);
            onNavigate('model');
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    onDestroy(() => {
        destroyed = true;
        if (taskId) void api('reviews/' + taskId + '/cancel', rev()).catch(() => {});
    });
</script>

<section
    class="model-card review-panel"
    aria-label={view === 'tables' ? 'Table adjustments' : 'Detailed diagnostics'}
>
    <h2>
        {view === 'tables'
            ? 'Adjustments & actual versus expected'
            : diagnosticTab === 'residual'
              ? 'Residual factors and interactions'
              : {
                    lift: 'Lift',
                    double_lift: 'Double lift',
                    path: 'Regularisation path',
                    coefficients: 'Coefficients',
                    compare: 'Relativities that differ',
                }[diagnosticTab] || 'Actual versus expected'}
    </h2>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if view === 'diagnostics'}
        {#if ['lift', 'double_lift'].includes(diagnosticTab) || (['variable', 'pair'].includes(diagnosticTab) && temporaryBins)}<div
                class="results-toolbar"
            >
                <label
                    >{['variable', 'pair'].includes(diagnosticTab)
                        ? 'Temporary bands (unfitted numeric variables)'
                        : 'Equal-exposure bins'}<input
                        aria-label="Diagnostic bins"
                        type="number"
                        min="3"
                        max="50"
                        bind:value={bins}
                    /></label
                ><button disabled={busy} onclick={runTab}>Update view</button>
            </div>{/if}
        {#if diagnosticTab === 'coefficients'}<label
                ><input type="checkbox" bind:checked={kept} onchange={runTab} /> Coefficients kept only</label
            >{/if}
        {#if diagnosticTab === 'compare'}<div class="results-toolbar">
                <label
                    >Log-difference tolerance<input
                        aria-label="Difference tolerance"
                        type="number"
                        min="0"
                        max="5"
                        step=".01"
                        bind:value={tolerance}
                    /></label
                ><button disabled={busy} onclick={runTab}>Update differences</button>
            </div>{/if}

        <div class="results-toolbar" hidden={diagnosticTab !== 'variable'}>
            <label
                >Variable<select
                    aria-label="Diagnostic variable"
                    bind:value={diagnosticVariable}
                    disabled={busy}
                    onchange={() => run('variable', { variable: diagnosticVariable })}
                    >{#each info.variables as v}<option value={v}
                            >{v}{info.variable_info?.find((x) => x.name === v)?.kind
                                ? ''
                                : ' (not in model)'}</option
                        >{/each}</select
                ></label
            >
        </div>
        <div>
            <div class="results-toolbar" hidden={diagnosticTab !== 'pair'}>
                <label
                    >First variable<select aria-label="Pair first variable" bind:value={a}
                        >{#each info.variables as v}<option value={v}
                                >{v}{info.variable_info?.find((x) => x.name === v)?.kind
                                    ? ''
                                    : ' (not in model)'}</option
                            >{/each}</select
                    ></label
                >
                <label
                    >Second variable<select aria-label="Pair second variable" bind:value={b}
                        >{#each info.variables as v}<option value={v}
                                >{v}{info.variable_info?.find((x) => x.name === v)?.kind
                                    ? ''
                                    : ' (not in model)'}</option
                            >{/each}</select
                    ></label
                >
                <button disabled={busy || a === b} onclick={() => run('pair', { a, b })}
                    >Show pair A/E</button
                >
            </div>
            <div hidden={diagnosticTab !== 'residual'}>
                <p class="help-text">
                    Searches rank residual signal on training data only. Holdout is reserved for
                    validation. Missing-factor search excludes IDs, explicitly ignored columns and
                    model response/weight fields. Interaction search removes main-effect misfit
                    before ranking pairs.
                </p>
                <div class="results-toolbar">
                    <button disabled={busy} onclick={() => run('factors')}
                        >Find missing factors</button
                    ><button disabled={busy} onclick={() => run('interactions')}
                        >Find missing interactions</button
                    >
                </div>
                {#if searchKind === 'factors' && searchRows.length}<div class="factor-selection">
                        <strong>Factors to add</strong>{#each searchRows as row}<label
                                ><input
                                    type="checkbox"
                                    value={row.variable}
                                    bind:group={selectedFactors}
                                />{row.variable} · signal {num(row.signal)}</label
                            >{/each}<button
                            disabled={busy || !selectedFactors.length}
                            onclick={() => include({ variables: selectedFactors })}
                            >Add selected and review model</button
                        >
                    </div>{/if}
                {#if searchKind}<DiagnosticTable
                        rows={searchRows}
                        title={searchKind === 'factors'
                            ? 'Factor search statistics'
                            : 'Interaction search statistics'}
                    />
                    <h3>
                        {searchKind === 'factors' ? 'Missing factors' : 'Missing interactions'}
                    </h3>
                    {#if !searchRows.length}<p>
                            No eligible residual candidates returned.
                        </p>{:else}<div class="review-scroll">
                            <table>
                                <thead
                                    ><tr
                                        ><th>Candidate</th><th>Signal</th><th>Review</th><th
                                            >Model</th
                                        ></tr
                                    ></thead
                                ><tbody
                                    >{#each searchRows.slice(0, 200) as row}<tr
                                            ><td>{row.variable || row.pair}</td><td
                                                >{num(row.signal)}</td
                                            ><td
                                                ><button
                                                    disabled={busy}
                                                    onclick={() =>
                                                        searchKind === 'factors'
                                                            ? run('variable', {
                                                                  variable: row.variable,
                                                                  subset: 'train',
                                                              })
                                                            : run('pair', {
                                                                  a: row.a,
                                                                  b: row.b,
                                                                  subset: 'train',
                                                                  n_bins: 8,
                                                                  options: { search_preview: true },
                                                              })}>Inspect</button
                                                ></td
                                            ><td
                                                ><button
                                                    disabled={busy}
                                                    onclick={() => include(row)}
                                                    >Add and review model</button
                                                ></td
                                            ></tr
                                        >{/each}</tbody
                                >
                            </table>
                        </div>{/if}
                {/if}
            </div>
        </div>
    {:else}
        <p class="help-text">
            Edit individual relativities in the table above, then preview. The original fitted model
            stays fixed. Charts compare actual experience, the original fit and current or proposed
            adjusted predictions.
        </p>
        <div class="results-toolbar">
            <button disabled={busy || !info.undo} onclick={() => run('undo')}>Preview undo</button
            ><button disabled={busy || !info.redo} onclick={() => run('redo')}>Preview redo</button>
            <button disabled={busy} onclick={() => run('rebalance')}
                >Preview rebalance base rate</button
            >
            <button disabled={busy} onclick={() => run('reset')}>Preview reset to fitted</button>
        </div>
        <details>
            <summary>Smooth, cap / floor and round</summary>
            <div class="results-toolbar">
                <label
                    >Tool<select aria-label="Adjustment tool" bind:value={tool}
                        ><option value="moving">Moving average</option><option value="isotonic"
                            >Monotone smoothing</option
                        ><option value="cap">Cap / floor</option><option value="round">Round</option
                        ></select
                    ></label
                >
                {#if tool === 'moving'}<label
                        >Window<input
                            aria-label="Smoothing window"
                            type="number"
                            min="3"
                            step="2"
                            bind:value={windowSize}
                        /></label
                    >{/if}
                {#if tool === 'isotonic'}<label
                        >Direction<select aria-label="Smoothing direction" bind:value={direction}
                            ><option value="increasing">Increasing</option><option
                                value="decreasing">Decreasing</option
                            ></select
                        ></label
                    >{/if}
                {#if tool === 'moving' || tool === 'isotonic'}<label
                        ><input type="checkbox" bind:checked={ordered} /> Categorical levels have a meaningful
                        order</label
                    >{/if}
                {#if tool === 'cap'}<label
                        >Floor<input
                            aria-label="Relativity floor"
                            type="number"
                            min="0"
                            step=".01"
                            bind:value={floor}
                        /></label
                    ><label
                        >Cap<input
                            aria-label="Relativity cap"
                            type="number"
                            min="0"
                            step=".01"
                            bind:value={cap}
                        /></label
                    >{/if}
                {#if tool === 'round'}<label
                        >Round by<select bind:value={rounding}
                            ><option value="decimals">Decimal places</option><option value="step"
                                >Increment</option
                            ></select
                        ></label
                    >{#if rounding === 'decimals'}<label
                            >Decimals<input
                                aria-label="Rounding decimals"
                                type="number"
                                min="0"
                                max="10"
                                bind:value={decimals}
                            /></label
                        >{:else}<label
                            >Increment<input
                                aria-label="Rounding increment"
                                type="number"
                                min=".00001"
                                step=".01"
                                bind:value={step}
                            /></label
                        >{/if}{/if}
                <button disabled={busy} onclick={previewTool}>Preview tool</button>
            </div>
            <p class="help-text">
                Smoothing uses exposure weights and preserves the mean log relativity, which does
                not preserve the portfolio total. Null / Other rows are untouched by tools.
                Smoothing unordered categories and tooling interaction cells are refused.
            </p>
        </details>
        <details>
            <summary>Named snapshots</summary>
            <div class="results-toolbar">
                <label
                    >New snapshot<input
                        aria-label="Snapshot name"
                        bind:value={snapshotName}
                    /></label
                ><button
                    disabled={busy || !snapshotName.trim()}
                    onclick={() => run('snapshot', { snapshot: snapshotName })}
                    >Save snapshot</button
                ><label
                    >Saved snapshot<select aria-label="Saved snapshot" bind:value={chosenSnapshot}
                        >{#each info.snapshots as s}<option>{s}</option>{/each}</select
                    ></label
                ><button
                    disabled={busy || !chosenSnapshot}
                    onclick={() => run('restore_snapshot', { snapshot: chosenSnapshot })}
                    >Preview snapshot restore</button
                >
            </div>
        </details>
    {/if}
    {#if busy}<div class="message" role="status">
            Computing in background… <button onclick={cancel} disabled={!taskId}
                >Cancel review</button
            >
        </div>{/if}
    {#if note}<p class="help-text">{note}</p>{/if}
    {#if preview}<div class="preview-impact">
            <strong
                >Training expected: {num(preview.before_expected)} → {num(preview.after_expected)} ({num(
                    (preview.change || 0) * 100,
                )}%)</strong
            >
            <p>
                Original fitted total: {num(preview.fitted_expected)}. Preview only; no settings
                have changed.
            </p>
            {#if preview.changes?.length}<details>
                    <summary>{preview.changes.length} changed rows — before and after</summary>
                    <div class="review-scroll">
                        <table>
                            <thead
                                ><tr
                                    ><th>Row</th><th>Band / cell</th><th>Before</th><th>After</th
                                    ><th>Slope before</th><th>Slope after</th></tr
                                ></thead
                            ><tbody
                                >{#each preview.changes.slice(0, 200) as change}<tr
                                        ><td>{change.row}</td><td>{change.label}</td><td
                                            >{num(change.before)}</td
                                        ><td>{num(change.after)}</td><td
                                            >{num(change.before_slope)}</td
                                        ><td>{num(change.after_slope)}</td></tr
                                    >{/each}</tbody
                            >
                        </table>
                        {#if preview.changes.length > 200}<p>First 200 changed rows shown.</p>{/if}
                    </div>
                </details>{/if}
            <button class="primary" disabled={busy} onclick={applyPreview}>Apply adjustment</button
            ><button disabled={busy} onclick={() => run('variable')}>Discard preview</button>
        </div>{/if}
    {#if analysis}
        {#if analysis.base_rate_change !== undefined}<p>
                Base rate change: {num(100 * analysis.base_rate_change)}%
            </p>{/if}
        {#each analysis.charts || [] as chart, i}<DiagnosticPlot
                rows={chart.rows}
                title={chart.title}
                series={chart.series || [
                    { key: 'actual_rate', label: 'Actual' },
                    { key: 'expected_rate', label: 'Expected' },
                ]}
                ariaLabel={diagnosticTab === 'lift' && i === 0
                    ? 'Actual and expected lift on ' + subset
                    : ''}
            />{/each}
        {#if analysis.path}{#each [...new Set(analysis.path.map((r) => r.stage))] as stage}{#each [...new Set(analysis.path
                            .filter((r) => r.stage === stage)
                            .map((r) => r.l1_ratio))] as ratio}<DiagnosticPlot
                        rows={analysis.path.filter(
                            (r) => r.stage === stage && r.l1_ratio === ratio,
                        )}
                        title={'Stage ' + stage + ' · L1 ' + ratio + ' · regularisation path'}
                        xKey="alpha"
                        logX={true}
                        series={[
                            { key: 'cv_deviance', label: 'Mean CV deviance' },
                            { key: 'train_deviance', label: 'Training deviance' },
                        ]}
                    /><DiagnosticPlot
                        rows={analysis.path.filter(
                            (r) => r.stage === stage && r.l1_ratio === ratio,
                        )}
                        title={'Stage ' + stage + ' · L1 ' + ratio + ' · retained coefficients'}
                        xKey="alpha"
                        logX={true}
                        series={[{ key: 'n_nonzero', label: 'Nonzero coefficients' }]}
                    />{/each}{/each}{/if}
        {#each analysis.tables || [] as table}<DiagnosticTable
                rows={table.rows}
                title={table.title}
            />{/each}
    {/if}
    {#if rows.length && (view === 'tables' || (diagnosticTab === 'variable' && !pairRows) || (diagnosticTab === 'pair' && pairRows) || (diagnosticTab === 'residual' && inspectedResidual))}
        {#if pairRows}<h3>{shownTitle} · {shownSubset}</h3>
            <h3>Actual / expected by cell</h3>
            {#if rows.some((r) => r.challenger_ae !== undefined)}<label
                    >Heatmap model<select bind:value={pairMetric}
                        ><option value="ae">{name}</option><option value="challenger_ae"
                            >{challenger}</option
                        ></select
                    ></label
                >{/if}
            <div class="heatmap-scroll">
                <table class="ae-heatmap">
                    <thead
                        ><tr
                            ><th>A \ B</th>{#each pairB as label}<th>{label}</th>{/each}</tr
                        ></thead
                    ><tbody
                        >{#each pairA as label}<tr
                                ><th>{label}</th>{#each pairB as other}{@const cell = rows.find(
                                        (r) => r.label_a === label && r.label_b === other,
                                    )}<td
                                        style:background={heat(cell?.[pairMetric])}
                                        title={`Actual ${num(cell?.actual)}; expected ${num(pairMetric === 'challenger_ae' ? cell?.challenger_expected : cell?.expected)}; exposure ${num(cell?.exposure)}`}
                                        >{num(cell?.ae)}</td
                                    >{/each}</tr
                            >{/each}</tbody
                    >
                </table>
            </div>
        {:else}<DiagnosticPlot
                {rows}
                title={shownTitle + ' · ' + shownSubset}
                {series}
                kind={aeKind}
                ariaLabel="Actual fitted and adjusted by variable"
                showTable={false}
            />{/if}
        {#each aeSets as data}<DiagnosticPlot
                rows={data.rows}
                title={data.title}
                {series}
                kind={aeKind}
            />{/each}
        <details>
            <summary>A/E values and exposure</summary>
            <DiagnosticTable {rows} title="A/E values and exposure" />
        </details>
    {/if}
</section>
