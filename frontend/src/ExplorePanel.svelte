<script>
    import { onDestroy } from 'svelte';
    import { formatNumber as num } from './format.js';
    import DiagnosticTable from './DiagnosticTable.svelte';
    import ExploreChart from './ExploreChart.svelte';
    export let api,
        state,
        active = false,
        onNavigate,
        onReconnect,
        requestColumn = null;
    let column = '',
        model = '',
        nBins = 20,
        result = null,
        metadata = null;
    let pending = false,
        error = '',
        requestId = 0,
        requestedKey = '',
        cacheContext = '',
        seenColumnRequest = null,
        wasActive = false,
        refreshOnActivation = false;
    const cache = new Map();
    $: context = `${state?.project_id}:${state?.session_id}:${state?.revision}`;
    $: if (context !== cacheContext) {
        cacheContext = context;
        cache.clear();
        requestId += 1;
        result = null;
        metadata = null;
        requestedKey = '';
        error = '';
        if (model && !state?.models.includes(model)) model = '';
    }
    $: if (active !== wasActive) {
        wasActive = active;
        if (active) {
            refreshOnActivation = true;
            requestedKey = '';
        } else {
            requestId += 1;
            pending = false;
        }
    }
    $: if (active && requestColumn && requestColumn.id !== seenColumnRequest) {
        seenColumnRequest = requestColumn.id;
        column = requestColumn.name || '';
    }
    $: key = makeKey(context, model, column, nBins);
    $: if (active && state?.columns.length && key !== requestedKey) void load();
    $: tableRows =
        result?.table.map((row) => ({
            band: row.label,
            [result.exposure_label === 'Rows' ? 'rows' : 'exposure']: row.exposure,
            share: row.share,
            ...(result.target ? { observed_rate: row.rate } : {}),
        })) || [];
    function makeKey(scope, chosenModel, chosenColumn, bands) {
        return JSON.stringify([scope, chosenModel, chosenColumn, bands]);
    }
    async function load(force = false) {
        if (!active || !state?.columns.length) return;
        force = force || refreshOnActivation;
        refreshOnActivation = false;
        const wanted = makeKey(context, model, column, nBins),
            scope = context,
            sequence = ++requestId;
        requestedKey = wanted;
        result = null;
        error = '';
        pending = true;
        try {
            let fresh = force ? null : cache.get(wanted);
            if (!fresh) {
                const query = new URLSearchParams({ n_bins: String(nBins) });
                if (column) query.set('column', column);
                if (model) query.set('model', model);
                fresh = await api('explore?' + query);
                if (sequence !== requestId || scope !== context) return;
                cache.set(wanted, fresh);
            }
            if (sequence !== requestId || scope !== context) return;
            const freshContext = `${fresh.project_id}:${fresh.session_id}:${fresh.revision}`;
            if (freshContext !== scope)
                throw new Error('Data settings changed. Reconnect to load the current view.');
            column = fresh.column || '';
            model = fresh.model || '';
            const resolved = makeKey(scope, model, column, nBins);
            cache.set(resolved, fresh);
            requestedKey = resolved;
            result = fresh;
            metadata = fresh;
        } catch (e) {
            if (sequence === requestId && scope === context) error = e.message;
        } finally {
            if (sequence === requestId) pending = false;
        }
    }
    async function reconnect() {
        pending = true;
        error = '';
        try {
            await onReconnect();
            await load(true);
        } catch (e) {
            error = e.message;
        } finally {
            pending = false;
        }
    }
    function changeBands(event) {
        const value = Number(event.currentTarget.value);
        if (Number.isFinite(value)) nBins = Math.max(5, Math.min(50, Math.round(value)));
    }
    onDestroy(() => {
        requestId += 1;
    });
</script>

<section class="explore-panel" hidden={!active}>
    <div class="heading">
        <div>
            <h1>One-way effects</h1>
            <p>Observed response and exposure by variable.</p>
        </div>
    </div>
    {#if !state?.columns.length}
        <section class="model-card">
            <p>Open data to explore its variables.</p>
            <button onclick={() => onNavigate('project')}>Open data</button>
        </section>
    {:else}
        <section class="model-card one-way-panel" aria-label="One-way effects">
            <div class="explore-controls">
                <label class="variable-choice"
                    >Variable
                    <select
                        aria-label="Plot variable"
                        bind:value={column}
                        disabled={!metadata?.columns.length}
                    >
                        {#if !metadata}<option value={column}
                                >{column || 'Loading variables…'}</option
                            >{/if}
                        {#each metadata?.columns || [] as item}<option value={item.name}
                                >{item.name}</option
                            >{/each}
                    </select>
                </label>
                <label class="band-choice"
                    >Bands
                    <input
                        aria-label="Bands"
                        disabled={metadata?.columns.find((item) => item.name === column)?.kind ===
                            'categorical'}
                        title="Bands apply to numeric variables."
                        type="number"
                        min="5"
                        max="50"
                        step="1"
                        value={nBins}
                        oninput={(event) => {
                            const value = Number(event.currentTarget.value);
                            if (Number.isInteger(value) && value >= 5 && value <= 50) nBins = value;
                        }}
                        onchange={changeBands}
                    />
                </label>
                {#if metadata?.models.length > 1}
                    <label class="model-choice"
                        >Model
                        <select aria-label="Explore model" bind:value={model}>
                            {#each metadata.models as item}<option value={item.name}
                                    >{item.name}</option
                                >{/each}
                        </select>
                    </label>
                {/if}
            </div>
            {#if error}
                <div class="explore-error" role="alert">
                    <p>{error}</p>
                    <div>
                        {#if /split|train(?:ing)? (?:flag|column)|holdout/i.test(error)}
                            <button onclick={() => onNavigate('model')}
                                >Set up train / holdout</button
                            >
                        {:else}
                            <button onclick={() => load(true)} disabled={pending}>Retry</button
                            ><button onclick={reconnect} disabled={pending}>Reconnect</button>
                        {/if}
                    </div>
                </div>
            {:else if pending}<p class="help-text" role="status">Loading one-way effects…</p>
            {:else if result}
                <p class="response-basis">
                    {result.target
                        ? result.rate_label +
                          ' · ' +
                          (result.weight ? 'Weight: ' + result.weight : 'Unweighted')
                        : result.exposure_label + ' by variable'}
                </p>
                <div class="explore-summary">
                    <span>{num(result.rows)} {result.rows === 1 ? 'row' : 'rows'}</span><span
                        >{num(result.n_unique)} distinct {result.n_unique === 1
                            ? 'value'
                            : 'values'}</span
                    ><span>{num(result.null_share * 100)}% missing</span>
                </div>
                {#if !result.target}<div class="target-missing">
                        <span>Assign a target to see the observed response.</span><button
                            onclick={() => onNavigate('variables')}
                            >Assign target in Variables</button
                        >
                    </div>{/if}
                {#if result.column && result.table.length}
                    <ExploreChart
                        rows={result.table}
                        column={result.column}
                        kind={result.kind}
                        rateLabel={result.rate_label}
                        exposureLabel={result.exposure_label}
                        hasTarget={Boolean(result.target)}
                    />
                {:else}<p class="help-text">No variables are available for this selection.</p>{/if}
                <p class="help-text training-caption">
                    {result.sampled
                        ? `Training sample: ${num(result.rows)} of ${num(result.training_rows)} rows.`
                        : 'Training rows only.'}
                </p>
                {#if result.rate_excluded_rows > 0}<p class="help-text">
                        Observed rates exclude {num(result.rate_excluded_rows)}
                        {result.rate_excluded_rows === 1 ? 'row' : 'rows'} with missing target.
                    </p>{/if}
                <details>
                    <summary>Data table</summary><DiagnosticTable
                        rows={tableRows}
                        title={'One-way effects · ' + (result.column || 'Data')}
                    />
                </details>
            {/if}
        </section>
    {/if}
</section>

<style>
    .explore-controls {
        display: flex;
        gap: 16px;
        align-items: end;
        flex-wrap: wrap;
        margin-bottom: 22px;
    }
    .explore-controls label {
        display: grid;
        gap: 7px;
        font-size: 14px;
        color: var(--muted, #616a75);
    }
    .variable-choice {
        flex: 1 1 220px;
    }
    .band-choice {
        flex: 0 0 85px;
    }
    .model-choice {
        flex: 1 1 180px;
    }
    input,
    select {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    .response-basis {
        margin: 0 0 10px;
        color: var(--text, #252b33);
        font-size: 14px;
    }
    .explore-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 10px 25px;
        margin-bottom: 24px;
        font-size: 13px;
        color: var(--muted, #616a75);
    }
    .target-missing {
        display: flex;
        gap: 15px;
        flex-wrap: wrap;
        align-items: center;
        margin-bottom: 22px;
        font-size: 14px;
    }
    .explore-error {
        color: #a13e34;
    }
    .explore-error div {
        display: flex;
        gap: 10px;
    }
    .training-caption {
        margin: 0 0 18px;
    }
    details {
        border-top: 1px solid var(--border, #dedfe2);
        padding-top: 15px;
    }
    summary {
        cursor: pointer;
        font-weight: 600;
    }
</style>
