<script>
    import { formatNumber as num, formatLabels } from './format.js';
    import { onDestroy } from 'svelte';
    import RateChart from './RateChart.svelte';
    import DiagnosticPlot from './DiagnosticPlot.svelte';
    import PathChart from './PathChart.svelte';
    import DiagnosticTable from './DiagnosticTable.svelte';
    export let fitIdentity = '',
        comparisonFitIdentity = '',
        table = null,
        children,
        api,
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
        onNavigate,
        onManual = () => {},
        tableKind = '',
        rateLabel = 'relativity';
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
        // Pair diagnostics are scheduled from their complete selection/context key.
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
    let tool = '',
        windowSize = 3,
        direction = 'increasing',
        ordered = false,
        floor = null,
        cap = 2,
        rounding = 'decimals',
        decimals = 2,
        step = 0.05,
        snapshotName = '',
        chosenSnapshot = '',
        snapshotLeft = '__fitted__',
        snapshotRight = '__current__',
        confirmDelete = false,
        feedback = '',
        bookImpact = null,
        orderedVariable = '';
    $: if (variable !== orderedVariable) {
        orderedVariable = variable;
        ordered = false;
    }
    let autoEnabled = false,
        autoPending = false,
        autoRunning = false,
        autoEpoch = 0,
        autoTimer,
        baselineRows = [],
        autoError = '';
    $: if (Object.keys(edits).length && autoEnabled) {
        stopAuto();
        preview = null;
        rows = baselineRows;
    }
    $: if (
        preview?.manualSignature !== undefined &&
        preview.manualSignature !== JSON.stringify(edits)
    ) {
        preview = null;
        rows = baselineRows;
    }
    function stopAuto() {
        autoEnabled = false;
        autoPending = false;
        autoEpoch++;
        clearTimeout(autoTimer);
        if (autoRunning && taskId) void api('reviews/' + taskId + '/cancel', rev()).catch(() => {});
    }
    function toolOptions() {
        if (tool === 'moving') return { window: windowSize, ordered };
        if (tool === 'isotonic') return { direction, ordered };
        if (tool === 'cap') return { floor: floor ?? null, cap: cap ?? null };
        return rounding === 'decimals' ? { decimals } : { step };
    }
    function toolError() {
        if (Object.keys(edits).length) return 'Preview or discard your manual row edits first.';
        if (tableKind === 'interaction') return 'Edit interaction cells in the table.';
        if (['moving', 'isotonic'].includes(tool) && tableKind === 'categorical' && !ordered)
            return 'Confirm that these levels have a meaningful order.';
        if (
            tool === 'moving' &&
            (!Number.isInteger(windowSize) || windowSize < 1 || windowSize > 25)
        )
            return 'Enter a whole-number window from 1 to 25.';
        if (
            tool === 'cap' &&
            ((floor != null && (!Number.isFinite(floor) || floor <= 0)) ||
                (cap != null && (!Number.isFinite(cap) || cap <= 0)) ||
                (floor != null && cap != null && floor > cap))
        )
            return 'Use positive bounds, with floor no greater than cap.';
        if (
            tool === 'round' &&
            (rounding === 'decimals'
                ? !Number.isInteger(decimals) || decimals < 0 || decimals > 6
                : !Number.isFinite(step) || step <= 0)
        )
            return 'Enter valid rounding precision.';
        return '';
    }
    function parametersChanged() {
        stopAuto();
        preview = null;
        rows = baselineRows;
        error = '';
        note = '';
        autoError = '';
        autoEnabled = true;
        autoPending = true;
        const version = autoEpoch;
        queueMicrotask(() => {
            if (!autoEnabled || version !== autoEpoch) return;
            autoError = toolError();
            if (autoError) {
                autoPending = false;
                return;
            }
            const action = tool,
                options = toolOptions();
            const dispatch = async () => {
                if (!autoEnabled || version !== autoEpoch || destroyed) return;
                if (busy) {
                    autoTimer = setTimeout(dispatch, 100);
                    return;
                }
                autoRunning = true;
                await run(action, { options }, version);
                autoRunning = false;
                if (version === autoEpoch) autoPending = false;
            };
            autoTimer = setTimeout(dispatch, 350);
        });
    }
    function chooseTool(value) {
        tool = value;
        parametersChanged();
    }
    function enterManual() {
        stopAuto();
        preview = null;
        rows = baselineRows;
        autoError = '';
        onManual();
    }
    function discardPreview() {
        stopAuto();
        preview = null;
        rows = baselineRows;
        note = '';
        error = '';
        autoError = '';
    }
    $: smoothing = tool === 'moving' || tool === 'isotonic';
    $: selectedVariable = view === 'tables' ? variable : diagnosticVariable;
    $: key = `${name}:${state.session_id}:${state.revision}:${view}:${subset}:${variable}:${challenger}:${fitIdentity}:${comparisonFitIdentity}`;
    let scheduledPairKey = '',
        pairEpoch = 0,
        pairTimer,
        pairPending = false,
        pairMessage = '',
        activePair = false;
    $: pairRequestKey =
        view === 'diagnostics' && diagnosticTab === 'pair'
            ? JSON.stringify([key, a, b, temporaryBins ? bins : 10])
            : '';
    $: if (pairRequestKey !== scheduledPairKey) {
        scheduledPairKey = pairRequestKey;
        schedulePair();
    }
    function schedulePair() {
        const version = ++pairEpoch;
        clearTimeout(pairTimer);
        if (activePair && taskId) void api('reviews/' + taskId + '/cancel', rev()).catch(() => {});
        pairPending = false;
        pairMessage = '';
        rows = [];
        aeSets = [];
        shownTitle = '';
        analysis = null;
        if (!pairRequestKey) return;
        error = '';
        note = '';
        if (!a || !b || a === b) {
            pairMessage = 'Choose two different variables.';
            return;
        }
        if (temporaryBins && (!Number.isInteger(bins) || bins < 3 || bins > 50)) {
            pairMessage = 'Enter a whole number of bands from 3 to 50.';
            return;
        }
        const requestKey = pairRequestKey,
            selection = { a, b, n_bins: temporaryBins ? bins : 10 };
        pairPending = true;
        const dispatch = async () => {
            if (destroyed || version !== pairEpoch || requestKey !== pairRequestKey) return;
            if (busy || loadedKey !== key) {
                pairTimer = setTimeout(dispatch, 100);
                return;
            }
            activePair = true;
            await run('pair', selection, null, { version, requestKey });
            activePair = false;
            if (version === pairEpoch) pairPending = false;
        };
        pairTimer = setTimeout(dispatch, 200);
    }

    $: if (key !== loadedKey && autoEnabled) stopAuto();
    $: if (name && key !== loadedKey && !busy) load();
    $: series = [
        { key: 'actual_rate', label: 'Actual', color: '#c35b48' },
        { key: 'fitted_rate', label: 'Original fit', color: '#737e9b' },
        { key: 'challenger_rate', label: challenger || 'Challenger', color: '#439da5' },
        {
            key: 'expected_rate',
            label: 'Adjusted',
            color: '#287762',
        },
    ].filter((s) => rows.some((r) => Number.isFinite(r[s.key])));
    $: chartRows = rows.filter((r) => r.exposure > 0);
    $: maxValue = Math.max(0.000001, ...chartRows.flatMap((r) => series.map((s) => r[s.key] || 0)));
    $: pairRows = rows.length && 'label_a' in rows[0];
    $: pairA = [...new Set(rows.map((r) => r.label_a))];
    $: pairB = [...new Set(rows.map((r) => r.label_b))];

    $: pairLabelsA = formatLabels(pairA);
    $: pairLabelsB = formatLabels(pairB);
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
        stopAuto();
        autoError = '';
        baselineRows = [];
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
    export async function previewRowEdits() {
        await run('edit', { edits });
    }
    async function run(action, extra = {}, autoVersion = null, pairVersion = null) {
        if (autoVersion === null) stopAuto();
        if (busy || destroyed) return;
        busy = true;
        error = '';
        note = '';
        if (action !== 'variable') feedback = '';
        preview = null;
        const started = key;
        const startedTab = diagnosticTab;
        const manualSignature = action === 'edit' ? JSON.stringify(edits) : undefined;
        const valid = () =>
            !destroyed &&
            started === key &&
            (view !== 'diagnostics' || startedTab === diagnosticTab) &&
            (pairVersion === null ||
                (pairVersion.version === pairEpoch && pairVersion.requestKey === pairRequestKey)) &&
            (manualSignature === undefined || manualSignature === JSON.stringify(edits)) &&
            (autoVersion === null ||
                (autoVersion === autoEpoch && autoEnabled && !Object.keys(edits).length));
        try {
            const payload = {
                ...rev(),
                action,
                variable: selectedVariable || null,
                subset,
                challenger: challenger || null,
                n_bins: Number(bins),
                tolerance: Number(tolerance),
                options: { both_subsets: view === 'diagnostics' },
                ...extra,
            };
            let response;
            // A previous view can still be cancelling its worker during navigation.
            // Retry only this explicit refusal, before any new work was started.
            for (let attempt = 0; attempt < 50; attempt++) {
                if (!valid()) return;
                try {
                    response = await api('review/' + encodeURIComponent(name), payload);
                    break;
                } catch (e) {
                    if (!e.message.includes('A review is running') || attempt === 49) throw e;
                    await new Promise((resolve) => setTimeout(resolve, 100));
                }
            }
            if (response.snapshot) {
                await onApplied(response.snapshot);
                feedback =
                    action === 'delete_snapshot'
                        ? 'Snapshot deleted.'
                        : 'Snapshot saved in the project.';
                confirmDelete = false;
                return;
            }
            taskId = response.id;
            if (!valid()) {
                await api('reviews/' + taskId + '/cancel', rev());
                return;
            }
            while (!destroyed) {
                if (!valid()) {
                    await api('reviews/' + taskId + '/cancel', rev());
                    return;
                }
                const response = await api('reviews/' + taskId);
                if (!valid()) return;
                if (response.status === 'complete') {
                    if (!valid()) return;
                    const data = response.data;
                    if (data.book_impact) bookImpact = data.book_impact;
                    note = data.note || '';
                    if (
                        [
                            'lift',
                            'double_lift',
                            'path',
                            'coefficients',
                            'compare',
                            'compare_snapshots',
                        ].includes(action)
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
                        if (action === 'variable') baselineRows = rows;
                        if (!rows.some((r) => r[pairMetric] !== undefined)) pairMetric = 'ae';
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
                        if (response.can_apply || data.preview_table) {
                            preview = {
                                ...data,
                                id: taskId,
                                canApply: response.can_apply,
                                autoVersion,
                                manualSignature,
                            };
                        }
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
            if (valid()) error = e.message;
        } finally {
            busy = false;
            taskId = '';
        }
    }
    async function cancel() {
        stopAuto();
        if (activePair || pairPending) {
            pairEpoch++;
            clearTimeout(pairTimer);
            pairPending = false;
            pairMessage = 'Pair review cancelled.';
        }
        if (taskId) await api('reviews/' + taskId + '/cancel', rev());
    }
    async function applyPreview() {
        if (
            busy ||
            autoPending ||
            !preview?.canApply ||
            (preview.manualSignature !== undefined &&
                preview.manualSignature !== JSON.stringify(edits)) ||
            (preview.autoVersion != null && preview.autoVersion !== autoEpoch)
        )
            return;
        stopAuto();
        busy = true;
        error = '';
        try {
            const snapshot = await api('reviews/' + preview.id + '/apply', rev());
            onClear();
            preview = null;
            await onApplied(snapshot);
            feedback = 'Adjustments applied.';
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
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
            await onApplied(response.snapshot);
            onNavigate('model');
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    onDestroy(() => {
        pairEpoch++;
        clearTimeout(pairTimer);
        stopAuto();
        destroyed = true;
        if (taskId) void api('reviews/' + taskId + '/cancel', rev()).catch(() => {});
    });
</script>

{#snippet aeContent()}
    {#if rows.length && (view === 'tables' || (diagnosticTab === 'variable' && !pairRows) || (diagnosticTab === 'pair' && pairRows) || (diagnosticTab === 'residual' && inspectedResidual))}
        {#if pairRows}<h3>{shownTitle} · {shownSubset}</h3>
            <h3>Actual / expected by cell</h3>
            {#if rows.some((r) => r.challenger_ae !== undefined || r.fitted_ae !== undefined)}<label
                    >Heatmap model<select bind:value={pairMetric}
                        ><option value="ae">Adjusted</option
                        >{#if rows.some((r) => r.fitted_ae !== undefined)}<option value="fitted_ae"
                                >Original fit</option
                            >{/if}{#if rows.some((r) => r.challenger_ae !== undefined)}<option
                                value="challenger_ae">{challenger}</option
                            >{/if}</select
                    ></label
                >{/if}
            <div class="heatmap-scroll">
                <table class="ae-heatmap">
                    <thead
                        ><tr
                            ><th>A \ B</th>{#each pairB as label, i}<th title={label}
                                    >{pairLabelsB[i]}</th
                                >{/each}</tr
                        ></thead
                    ><tbody
                        >{#each pairA as label, i}<tr
                                ><th title={label}>{pairLabelsA[i]}</th
                                >{#each pairB as other}{@const cell = rows.find(
                                        (r) => r.label_a === label && r.label_b === other,
                                    )}<td
                                        style:background={heat(cell?.[pairMetric])}
                                        title={`Actual ${num(cell?.actual)}; expected ${num(pairMetric === 'challenger_ae' ? cell?.challenger_expected : pairMetric === 'fitted_ae' ? cell?.fitted_expected : cell?.expected)}; exposure ${num(cell?.exposure)}`}
                                        >{num(cell?.[pairMetric])}</td
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
{/snippet}

{#snippet previewControls()}
    {#if preview}<div class="preview-impact">
            {#if tool === 'isotonic' && preview.tool_details && preview.changes?.length === 0 && !preview.canApply && preview.before_base_rate === preview.after_base_rate}<p
                >
                    No changes needed.
                </p>{/if}
            <strong
                >Training expected: {num(preview.before_expected)} → {num(preview.after_expected)} ({num(
                    (preview.change || 0) * 100,
                )}%)</strong
            >

            <button
                class="primary"
                disabled={busy || autoPending || !preview.canApply}
                onclick={applyPreview}>Apply adjustment</button
            ><button onclick={discardPreview}>Discard preview</button>
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
            {#if preview.before_base_rate !== preview.after_base_rate}<p>
                    Base rate: {num(preview.before_base_rate)} → {num(preview.after_base_rate)}
                </p>{/if}
            {#if preview.tool_details}<details>
                    <summary>Calculation details</summary>
                    <p class="help-text">
                        Weighted mean log relativity: {num(preview.tool_details.log_mean_before)} → {num(
                            preview.tool_details.log_mean_after,
                        )}. {note}
                    </p>
                </details>{/if}
        </div>{/if}
{/snippet}

<div class="review-layout">
    <section
        class="model-card review-panel"
        class:rate-relativities={view === 'tables'}
        aria-label={view === 'tables' ? 'Table adjustments' : 'Detailed diagnostics'}
    >
        <h2>
            {view === 'tables'
                ? 'Relativities'
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
        {#if view === 'tables' && table}
            <p class="rate-preview-state help-text">
                {preview ? 'Preview · not applied' : 'Applied adjustments'}
            </p>
            <RateChart
                table={preview?.preview_table || table}
                {variable}
                label={rateLabel}
                fittedLabel="Original fit"
                currentLabel="Adjusted"
                preview={!!preview?.preview_table}
            />

            <h3 class="adjustments-heading">Adjustments</h3>
        {/if}
        {#if feedback}<div class="message success review-feedback" role="status">
                {feedback}
            </div>{/if}
        {#if error && view !== 'tables'}<div class="message error" role="alert">{error}</div>{/if}
        {#if view === 'diagnostics'}
            {#if ['lift', 'double_lift'].includes(diagnosticTab) || (diagnosticTab === 'variable' && temporaryBins)}<div
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
                    ><input type="checkbox" bind:checked={kept} onchange={runTab} /> Coefficients kept
                    only</label
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
                <div class="pair-controls" hidden={diagnosticTab !== 'pair'}>
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
                    {#if temporaryBins}<label
                            >Temporary bands<input
                                aria-label="Pair diagnostic bins"
                                type="number"
                                min="3"
                                max="50"
                                step="1"
                                bind:value={bins}
                            /></label
                        >{/if}
                </div>
                {#if diagnosticTab === 'pair' && (pairMessage || pairPending)}<p
                        class="pair-status"
                        role="status"
                    >
                        {pairMessage || 'Updating pair A/E…'}
                    </p>{/if}
                <div hidden={diagnosticTab !== 'residual'}>
                    <p class="help-text">
                        Searches rank residual signal on training data only. Holdout is reserved for
                        validation. Missing-factor search excludes IDs, explicitly ignored columns
                        and model response/weight fields. Interaction search removes main-effect
                        misfit before ranking pairs.
                    </p>
                    <div class="results-toolbar">
                        <button disabled={busy} onclick={() => run('factors')}
                            >Find missing factors</button
                        ><button disabled={busy} onclick={() => run('interactions')}
                            >Find missing interactions</button
                        >
                    </div>
                    {#if searchKind === 'factors' && searchRows.length}<div
                            class="factor-selection"
                        >
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
                                                                      options: {
                                                                          search_preview: true,
                                                                      },
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
            <div class="adjustment-modes" role="group" aria-label="Adjustment method">
                {#each [['moving', 'Moving average'], ['isotonic', 'Isotonic smoothing'], ['cap', 'Cap / floor'], ['round', 'Round']] as [value, label]}
                    <button
                        aria-pressed={tool === value}
                        disabled={(busy && !autoRunning) ||
                            tableKind === 'interaction' ||
                            Object.keys(edits).length > 0}
                        onclick={() => chooseTool(value)}>{label}</button
                    >
                {/each}
                <button disabled={busy} onclick={enterManual}
                    >Edit individual or multiple rows</button
                >
            </div>
            {#if tableKind === 'interaction'}<p>
                    Interaction cells are edited individually in the rate table. Smoothing applies
                    to main factors.
                </p>
            {:else}
                <div class="adjustment-parameters">
                    {#if tool === 'moving'}<label
                            >Window (points)<input
                                aria-label="Smoothing window"
                                type="number"
                                min="1"
                                max="25"
                                step="1"
                                bind:value={windowSize}
                                oninput={parametersChanged}
                            /></label
                        >
                    {/if}
                    {#if tool === 'isotonic'}<label
                            >Direction<select
                                aria-label="Smoothing direction"
                                bind:value={direction}
                                onchange={parametersChanged}
                                ><option value="increasing">Increasing</option><option
                                    value="decreasing">Decreasing</option
                                ></select
                            ></label
                        >
                    {/if}
                    {#if tool === 'cap'}<label
                            >Floor (empty = none)<input
                                aria-label="Relativity floor"
                                type="number"
                                min=".0001"
                                step=".05"
                                bind:value={floor}
                                oninput={parametersChanged}
                            /></label
                        ><label
                            >Cap (empty = none)<input
                                aria-label="Relativity cap"
                                type="number"
                                min=".0001"
                                step=".05"
                                bind:value={cap}
                                oninput={parametersChanged}
                            /></label
                        >{/if}
                    {#if tool === 'round'}<label
                            >Round to<select
                                aria-label="Rounding mode"
                                bind:value={rounding}
                                onchange={parametersChanged}
                                ><option value="decimals">Decimal places</option><option
                                    value="step">A step</option
                                ></select
                            ></label
                        >{#if rounding === 'decimals'}<label
                                >Decimal places<input
                                    aria-label="Decimal places"
                                    type="number"
                                    min="0"
                                    max="6"
                                    bind:value={decimals}
                                    oninput={parametersChanged}
                                /></label
                            >{:else}<label
                                >Step<input
                                    aria-label="Rounding step"
                                    type="number"
                                    min=".0001"
                                    step=".01"
                                    bind:value={step}
                                    oninput={parametersChanged}
                                /></label
                            >
                        {/if}{/if}
                </div>
                {#if smoothing && tableKind === 'categorical'}<label class="ordered-confirmation"
                        ><input
                            type="checkbox"
                            bind:checked={ordered}
                            onchange={parametersChanged}
                        /> The levels of this factor are in a meaningful order</label
                    >
                    <p class="help-text">
                        Levels are normally ordered by exposure. Confirm only if neighbouring levels
                        represent a real ordered scale.
                    </p>{/if}
                <details class="tool-help">
                    <summary>About this adjustment</summary>
                    <p>
                        {#if tool === 'moving'}{windowSize === 1
                                ? 'Keeps each point unchanged.'
                                : windowSize === 2
                                  ? 'Averages the current point and the previous point.'
                                  : windowSize === 3
                                    ? 'Averages the current point and the previous two.'
                                    : `Averages the current point and the previous ${windowSize - 1} points.`}{:else if tool === 'isotonic'}{direction ===
                            'increasing'
                                ? 'Removes dips so rates only rise or stay flat. Rates already following this pattern stay unchanged.'
                                : 'Removes upward reversals so rates only fall or stay flat. Rates already following this pattern stay unchanged.'}{:else if tool === 'cap'}Keeps
                            values within the limits you set.{:else if tool === 'round'}{rounding ===
                            'decimals'
                                ? `Rounds to ${decimals} decimal ${decimals === 1 ? 'place' : 'places'}.`
                                : `Rounds to the nearest multiple of ${num(step)}.`}{/if}
                    </p>
                </details>
                {#if Object.keys(edits).length}<p>
                        Preview or discard your {Object.keys(edits).length} manual row edits before using
                        a tool.
                    </p>{/if}
            {/if}
            <div class="auto-preview-status" role="status">
                {#if error || autoError}{error || autoError}{:else if autoPending}Updating preview…{:else if !preview}Choose
                    a tool to preview its effect.{/if}
                {#if autoEnabled && !preview}<div class="auto-preview-actions">
                        <button disabled>Apply adjustment</button><button onclick={discardPreview}
                            >Discard preview</button
                        >
                    </div>{/if}
            </div>
            {@render previewControls()}
            {#if bookImpact}<p class="book-impact">
                    <strong>Current training expected: {num(bookImpact.current)}</strong> · fitted: {num(
                        bookImpact.fitted,
                    )} ({num(100 * (bookImpact.current / bookImpact.fitted - 1))}% from fitted)
                </p>{/if}
            <div class="adjustment-history results-toolbar">
                <button disabled={busy || !info.undo} onclick={() => run('undo')}
                    >Preview undo</button
                ><button disabled={busy || !info.redo} onclick={() => run('redo')}
                    >Preview redo</button
                >
                <button disabled={busy || info.link === 'logit'} onclick={() => run('rebalance')}
                    >Preview rebalance base rate</button
                >
                <button disabled={busy} onclick={() => run('reset_variable')}
                    >Reset this variable</button
                ><button disabled={busy} onclick={() => run('reset')}>Reset all adjustments</button>
            </div>
            <p class="help-text">
                Rebalance restores the original fitted training total by changing only the base
                rate. Every applied edit, tool, reset or rebalance is one undo step.
            </p>
            <details class="table-snapshots">
                <summary>Snapshots · save, restore and compare</summary>
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
                    >
                </div>
                {#if info.snapshots.length}<div class="results-toolbar">
                        <label
                            >Saved snapshot<select
                                aria-label="Saved snapshot"
                                bind:value={chosenSnapshot}
                                onchange={() => (confirmDelete = false)}
                                >{#each info.snapshots as s}<option>{s}</option>{/each}</select
                            ></label
                        ><button
                            disabled={busy || !chosenSnapshot}
                            onclick={() => run('restore_snapshot', { snapshot: chosenSnapshot })}
                            >Preview snapshot restore</button
                        ><button
                            disabled={busy || !chosenSnapshot}
                            onclick={() => (confirmDelete = true)}>Delete snapshot</button
                        >
                    </div>{/if}
                {#if confirmDelete}<div class="message">
                        Delete “{chosenSnapshot}” permanently? Undo cannot restore this snapshot.
                        <button
                            disabled={busy}
                            onclick={() =>
                                run('delete_snapshot', {
                                    snapshot: chosenSnapshot,
                                    options: { confirmed: true },
                                })}>Confirm delete snapshot</button
                        ><button onclick={() => (confirmDelete = false)}>Keep snapshot</button>
                    </div>{/if}
                <div class="results-toolbar">
                    <label
                        >Compare<select aria-label="First table version" bind:value={snapshotLeft}
                            ><option value="__fitted__">Original fitted</option><option
                                value="__current__">Current tables</option
                            >{#each info.snapshots as s}<option>{s}</option>{/each}</select
                        ></label
                    ><label
                        >With<select aria-label="Second table version" bind:value={snapshotRight}
                            ><option value="__fitted__">Original fitted</option><option
                                value="__current__">Current tables</option
                            >{#each info.snapshots as s}<option>{s}</option>{/each}</select
                        ></label
                    ><label
                        >Log difference tolerance<input
                            aria-label="Snapshot comparison tolerance"
                            type="number"
                            min="0"
                            max="1"
                            step=".005"
                            bind:value={tolerance}
                        /></label
                    ><button
                        disabled={busy || snapshotLeft === snapshotRight}
                        onclick={() =>
                            run('compare_snapshots', {
                                options: { left: snapshotLeft, right: snapshotRight },
                            })}>Compare table versions</button
                    >
                </div>
            </details>
            {#if info.adjustments?.length}<details>
                    <summary>Applied adjustments ({info.adjustments.length})</summary
                    ><DiagnosticTable rows={info.adjustments} title="Applied adjustments" />
                </details>{/if}
        {/if}
        {#if busy && !autoRunning && !pairPending}<div class="message" role="status">
                Computing in background… <button onclick={cancel} disabled={!taskId}
                    >Cancel review</button
                >
            </div>{/if}
        {#if note && view !== 'tables'}<p class="help-text">{note}</p>{/if}
        {#if view !== 'tables'}{@render previewControls()}{/if}
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
                                .map((r) => r.l1_ratio))] as ratio}<PathChart
                            rows={analysis.path.filter(
                                (r) => r.stage === stage && r.l1_ratio === ratio,
                            )}
                            title={'Stage ' +
                                stage +
                                ' · L1 ' +
                                num(ratio) +
                                ' · regularisation path'}
                        />{/each}{/each}{/if}
            {#each analysis.tables || [] as table}<DiagnosticTable
                    rows={table.rows}
                    title={table.title}
                />{/each}
        {/if}
        {@render children?.()}
        {#if view !== 'tables'}{@render aeContent()}{/if}
    </section>
    {#if view === 'tables'}<section
            class={view === 'tables' ? 'model-card rate-ae review-panel' : 'review-ae'}
            aria-label={view === 'tables' ? 'Actual versus expected' : undefined}
        >
            {#if view === 'tables'}<h2>Actual versus expected</h2>
                <label class="table-ae-subset"
                    >A/E subset<select aria-label="Table diagnostic subset" bind:value={subset}
                        ><option value="train">Training</option><option value="holdout"
                            >Holdout</option
                        ><option value="all">All rows</option></select
                    ></label
                >
            {/if}
            {@render aeContent()}
        </section>{/if}
</div>
