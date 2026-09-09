<script>
    import { onDestroy } from 'svelte';
    export let api,
        state,
        name,
        view,
        subset = 'train',
        variable = '',
        edits = {},
        onApplied,
        onClear,
        onNavigate;
    let info = { variables: [], snapshots: [], undo: false, redo: false };
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
    $: key = `${name}:${state.session_id}:${state.revision}:${view}:${subset}:${variable}`;
    $: if (name && key !== loadedKey && !busy) load();
    $: series = [
        { key: 'actual_rate', label: 'Actual', color: '#287762' },
        { key: 'fitted_rate', label: 'Original fitted', color: '#737e9b' },
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
        const v = view === 'tables' ? variable : diagnosticVariable;
        if (v) await run('variable', { variable: v });
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
                    if (action === 'factors' || action === 'interactions') {
                        searchRows = data.rows;
                        searchKind = action;
                    } else {
                        rows = data.rows || [];
                        shownTitle =
                            action === 'pair'
                                ? `${extra.a || a} × ${extra.b || b}`
                                : extra.variable || selectedVariable;
                        shownSubset = (extra.subset || subset) === 'train' ? 'Training' : 'Holdout';
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
                action: searchKind === 'factors' ? 'include_factor' : 'include_pair',
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
        {view === 'tables' ? 'Adjustments & actual versus expected' : 'Actual versus expected'}
    </h2>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if view === 'diagnostics'}
        <div class="results-toolbar">
            <label
                >Variable<select
                    aria-label="Diagnostic variable"
                    bind:value={diagnosticVariable}
                    disabled={busy}
                    onchange={() => run('variable', { variable: diagnosticVariable })}
                    >{#each info.variables as v}<option value={v}>{v}</option>{/each}</select
                ></label
            >
        </div>
        <details>
            <summary>Two-variable A/E and missing terms</summary>
            <div class="results-toolbar">
                <label
                    >First variable<select aria-label="Pair first variable" bind:value={a}
                        >{#each info.variables as v}<option value={v}>{v}</option>{/each}</select
                    ></label
                >
                <label
                    >Second variable<select aria-label="Pair second variable" bind:value={b}
                        >{#each info.variables as v}<option value={v}>{v}</option>{/each}</select
                    ></label
                >
                <button disabled={busy || a === b} onclick={() => run('pair', { a, b })}
                    >Show pair A/E</button
                >
            </div>
            <p class="help-text">
                Searches rank residual signal on training data only. Holdout is reserved for
                validation. Missing-factor search excludes IDs, explicitly ignored columns and model
                response/weight fields. Interaction search removes main-effect misfit before ranking
                pairs.
            </p>
            <div class="results-toolbar">
                <button disabled={busy} onclick={() => run('factors')}>Find missing factors</button
                ><button disabled={busy} onclick={() => run('interactions')}
                    >Find missing interactions</button
                >
            </div>
            {#if searchKind}<h3>
                    {searchKind === 'factors' ? 'Missing factors' : 'Missing interactions'}
                </h3>
                {#if !searchRows.length}<p>No eligible residual candidates returned.</p>{:else}<div
                        class="review-scroll"
                    >
                        <table>
                            <thead
                                ><tr
                                    ><th>Candidate</th><th>Signal</th><th>Review</th><th>Model</th
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
                                                          })}>Inspect</button
                                            ></td
                                        ><td
                                            ><button disabled={busy} onclick={() => include(row)}
                                                >Add and review model</button
                                            ></td
                                        ></tr
                                    >{/each}</tbody
                            >
                        </table>
                    </div>{/if}
            {/if}
        </details>
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
    {#if rows.length}
        <h3>{shownTitle} · {shownSubset}</h3>
        {#if pairRows}<h3>Actual / expected by cell</h3>
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
                                        style:background={heat(cell?.ae)}
                                        title={`Actual ${num(cell?.actual)}; expected ${num(cell?.expected)}; exposure ${num(cell?.exposure)}`}
                                        >{num(cell?.ae)}</td
                                    >{/each}</tr
                            >{/each}</tbody
                    >
                </table>
            </div>
        {:else}<div class="chart-legend">
                {#each series as s}<span style:color={s.color}>● {s.label}</span>{/each}
            </div>
            <svg
                class="lift-chart"
                viewBox="0 0 1000 230"
                role="img"
                aria-label="Actual fitted and adjusted by variable"
                ><title>Actual, original fitted and adjusted rates</title
                >{#each [0, 0.5, 1] as tick}<line
                        x1="50"
                        x2="950"
                        y1={190 - tick * 160}
                        y2={190 - tick * 160}
                        stroke="#e1e9e4"
                    /><text x="0" y={194 - tick * 160}>{num(tick * maxValue)}</text
                    >{/each}{#each series as s}<polyline
                        points={points(s.key)}
                        fill="none"
                        stroke={s.color}
                        stroke-width="2"
                    />{#each chartRows as row, i}<circle
                            cx={50 + (i * 900) / Math.max(1, chartRows.length - 1)}
                            cy={190 - ((row[s.key] || 0) / maxValue) * 160}
                            r="4"
                            fill={s.color}
                            ><title
                                >{row.label}: {s.label}
                                {num(row[s.key])}; exposure {num(row.exposure)}</title
                            ></circle
                        >{/each}{/each}{#each chartRows as row, i}{#if i % Math.max(1, Math.ceil(chartRows.length / 8)) === 0}<text
                            x={50 + (i * 900) / Math.max(1, chartRows.length - 1)}
                            y="220"
                            text-anchor="middle">{String(row.label).slice(0, 16)}</text
                        >{/if}{/each}</svg
            >{/if}
        {#if !pairRows}<div class="exposure-caption">
                Exposure by band / level · largest {num(
                    Math.max(0, ...chartRows.map((r) => r.exposure)),
                )}
            </div>
            <svg
                class="exposure-chart"
                viewBox="0 0 1000 95"
                role="img"
                aria-label="Exposure by variable band"
                ><title>Exposure aligned with actual and expected rates</title><line
                    x1="50"
                    x2="950"
                    y1="80"
                    y2="80"
                    stroke="#d8e2dc"
                />{#each chartRows as row, i}<rect
                        x={50 +
                            (i * 900) / Math.max(1, chartRows.length - 1) -
                            Math.min(12, 350 / Math.max(1, chartRows.length)) / 2}
                        y={80 -
                            (row.exposure / Math.max(1, ...chartRows.map((r) => r.exposure))) * 65}
                        width={Math.min(12, 350 / Math.max(1, chartRows.length))}
                        height={(row.exposure / Math.max(1, ...chartRows.map((r) => r.exposure))) *
                            65}
                        fill="#91afa1"
                        ><title>{row.label}: exposure {num(row.exposure)}</title></rect
                    >{/each}</svg
            >{/if}
        <details>
            <summary>A/E values and exposure</summary>
            <div class="review-scroll">
                <table>
                    <thead
                        ><tr
                            >{#each Object.keys(rows[0]) as column}<th
                                    >{column.replaceAll('_', ' ')}</th
                                >{/each}</tr
                        ></thead
                    ><tbody
                        >{#each rows.slice(0, 200) as row}<tr
                                >{#each Object.values(row) as value}<td>{num(value)}</td>{/each}</tr
                            >{/each}</tbody
                    >
                </table>
                {#if rows.length > 200}<p>First 200 groups shown.</p>{/if}
            </div>
        </details>
    {/if}
</section>
