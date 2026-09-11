<script>
    import { onDestroy } from 'svelte';
    import DiagnosticPlot from './DiagnosticPlot.svelte';
    import DiagnosticTable from './DiagnosticTable.svelte';
    import { formatNumber as num } from './format.js';
    export let api,
        state,
        name,
        fitIdentity = '',
        onNavigate;
    let bands = 5,
        factor = '',
        relative = false;
    let data = null,
        error = '',
        busy = false,
        destroyed = false,
        task = '';
    let loadedKey = '',
        timer;
    const cache = new Map();
    $: timeColumn = state?.setup?.assignments?.time;
    $: context = JSON.stringify([name, fitIdentity, state?.revision, timeColumn, bands, factor]);
    $: if (timeColumn && context !== loadedKey && !busy) load(context);
    $: factorSeries = (data?.series || []).map((s) => ({
        ...s,
        key: s.key + (relative ? '_relative' : ''),
    }));
    function wait() {
        return new Promise((resolve) => {
            timer = setTimeout(resolve, 200);
        });
    }
    async function load(key) {
        loadedKey = key;
        error = '';
        data = null;
        if (!Number.isInteger(Number(bands)) || bands < 3 || bands > 20) {
            error = 'Choose between 3 and 20 time bands.';
            return;
        }
        if (cache.has(key)) {
            data = cache.get(key);
            return;
        }
        busy = true;
        try {
            const started = await api('review/' + encodeURIComponent(name), {
                session_id: state.session_id,
                revision: state.revision,
                action: 'time',
                subset: 'all',
                n_bins: Number(bands),
                variable: factor || null,
            });
            task = started.id;
            while (!destroyed) {
                const result = await api('reviews/' + task);
                if (result.status === 'complete') {
                    if (key === context) {
                        data = result.data;
                        cache.set(key, data);
                    }
                    break;
                }
                if (['failed', 'stale', 'cancelled'].includes(result.status))
                    throw new Error(
                        result.data?.error || 'Settings changed. Reopen Time stability to refresh.',
                    );
                await wait();
            }
        } catch (e) {
            if (!destroyed && key === context) error = e.message;
        } finally {
            busy = false;
            task = '';
        }
    }
    onDestroy(() => {
        destroyed = true;
        clearTimeout(timer);
        if (task)
            api('reviews/' + task + '/cancel', {
                session_id: state.session_id,
                revision: state.revision,
            }).catch(() => {});
    });
</script>

<section class="model-card" aria-label="Time stability">
    <h2>Time stability</h2>
    <p class="help-text">
        Original fit · all training and holdout rows. A/E above 1 means actual exceeds expected.
    </p>
    {#if !timeColumn}
        <p>Assign a Time role to a numeric column on Variables.</p>
        <button onclick={() => onNavigate('variables')}>Set Time on Variables</button>
    {:else}
        <div class="time-controls">
            <label
                >Time bands<input
                    aria-label="Time bands"
                    type="number"
                    min="3"
                    max="20"
                    step="1"
                    bind:value={bands}
                    disabled={busy}
                /></label
            >
            <span class="help-text">Earliest to latest · equal time values stay together.</span>
        </div>
        {#if error}<div role="alert" class="message error">{error}</div>{/if}
        {#if busy}<p role="status">Calculating time diagnostics…</p>{/if}
        {#if data}
            {#if data.synthetic_time}<p class="help-text">
                    Synthetic years for demonstration; these are not the dataset’s actual dates.
                </p>{/if}
            <p class="help-text">
                {data.time_column} · {num(data.rows)} rows · {data.bands} time bands{#if data.excluded_rows}
                    · {num(data.excluded_rows)} rows excluded: missing time{/if}
            </p>
            <DiagnosticPlot
                rows={data.overall}
                title="Actual / expected over time"
                series={[{ key: 'ae', label: 'Original fit A/E' }]}
                referenceValue={1}
            />
            <details class="factor-time" open={Boolean(factor)}>
                <summary>Compare a factor across time</summary>
                <label
                    >Factor<select
                        aria-label="Time comparison factor"
                        bind:value={factor}
                        disabled={busy}
                    >
                        <option value="">Choose a factor…</option>
                        {#each data.factors as item}<option value={item}>{item}</option>{/each}
                    </select></label
                >
                {#if factor}
                    <label class="relative-toggle"
                        ><input type="checkbox" bind:checked={relative} /> Relative to the whole book
                        in each period</label
                    >
                    <DiagnosticPlot
                        rows={data.factor_rows}
                        title={factor + ' · A/E by period'}
                        series={factorSeries}
                        referenceValue={1}
                        showTable={false}
                    />
                    <p class="help-text">
                        Bars show exposure across all periods. See the table for each period’s
                        exposure and actual total. Small groups can be noisy.
                    </p>
                    <details>
                        <summary>Period values and exposure</summary><DiagnosticTable
                            rows={data.cells}
                            title="Factor by time"
                        />
                    </details>
                {/if}
            </details>
            <p class="help-text">
                Descriptive stability check, not a future-period test. Compare claims at similar
                maturity; a changing pattern can also reflect changes in the portfolio.
            </p>
        {/if}
    {/if}
</section>

<style>
    .time-controls {
        display: flex;
        align-items: end;
        gap: 20px;
        flex-wrap: wrap;
    }
    .time-controls label {
        display: grid;
        gap: 6px;
        max-width: 130px;
    }
    .factor-time {
        margin-top: 24px;
        border-top: 1px solid var(--border, #ddd);
        padding-top: 18px;
    }
    .factor-time > label {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 16px;
    }
    .factor-time select {
        max-width: 360px;
    }
</style>
