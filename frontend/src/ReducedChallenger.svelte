<script>
    import { formatNumber as num } from './format.js';
    export let api, state, name, fitIdentity, info, rows, onCreated;
    let selected = [],
        topN = 1,
        newName = '',
        search = '',
        busy = false,
        error = '',
        loaded = '';
    $: predictors = info.predictors || [];
    $: rank = new Map(
        rows.filter((r) => Number.isFinite(r.importance)).map((r) => [r.variable, r.importance]),
    );
    $: ranked = [...predictors].sort(
        (a, b) => (rank.get(b) ?? -Infinity) - (rank.get(a) ?? -Infinity) || a.localeCompare(b),
    );
    $: key = JSON.stringify([name, fitIdentity]);
    $: if (key !== loaded && predictors.length) {
        loaded = key;
        selected = [...predictors];
        topN = Math.min(15, Math.max(1, predictors.length - 1));
        let proposal = name + ' reduced',
            suffix = 2;
        while (info.model_names?.includes(proposal)) proposal = name + ' reduced ' + suffix++;
        newName = proposal;
        error = '';
    }
    $: removedInteractions = (info.interactions || []).filter(
        (i) => !selected.includes(i.a) || !selected.includes(i.b),
    );
    $: valid =
        selected.length > 0 &&
        selected.length < predictors.length &&
        newName.trim() &&
        !info.model_names?.includes(newName.trim());
    function toggle(p, checked) {
        selected = checked ? [...selected, p] : selected.filter((x) => x !== p);
    }
    async function create() {
        busy = true;
        error = '';
        try {
            const result = await api('models/' + encodeURIComponent(name) + '/reduce', {
                session_id: state.session_id,
                revision: state.revision,
                name: newName.trim(),
                fit_id: fitIdentity,
                predictors: selected,
            });
            await onCreated(result);
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
</script>

<section class="reduction" aria-label="Build a smaller challenger">
    <h3>Build a smaller challenger</h3>
    <p class="help-text">
        Keep the strongest predictors, then refine the selection. The original model stays
        available.
    </p>
    {#if predictors.length < 2}<p>At least two predictors are needed to reduce this model.</p>
    {:else}
        <div class="reduction-controls">
            <label
                >Top N<input
                    aria-label="Top N predictors"
                    type="number"
                    min="1"
                    max={predictors.length}
                    step="1"
                    bind:value={topN}
                    disabled={busy}
                /></label
            >
            <button
                disabled={busy || !Number.isInteger(topN) || topN < 1 || topN > predictors.length}
                onclick={() => (selected = ranked.slice(0, topN))}>Keep top N</button
            >
            <button disabled={busy} onclick={() => (selected = [...predictors])}>Select all</button>
            <span>{selected.length} of {predictors.length} predictors kept</span>
        </div>
        <input
            aria-label="Find predictor to keep"
            placeholder="Find a predictor…"
            bind:value={search}
        />
        <div class="predictor-options">
            {#each ranked.filter((p) => p.toLowerCase().includes(search.toLowerCase())) as p}
                <label
                    ><input
                        type="checkbox"
                        aria-label={'Keep predictor ' + p}
                        checked={selected.includes(p)}
                        disabled={busy}
                        onchange={(e) => toggle(p, e.currentTarget.checked)}
                    /><span>{p}</span><span class="help-text"
                        >{rank.has(p) ? num(rank.get(p)) : 'Not ranked'}</span
                    ></label
                >
            {/each}
        </div>
        <p class="help-text">
            {removedInteractions.length} interaction tables removed{#if removedInteractions.length}: {removedInteractions
                    .map((i) => i.a + ' × ' + i.b)
                    .join(', ')}{/if}.
        </p>
        <p class="help-text">
            Same split and fitting settings; a fresh fit without saved rate adjustments. Importance
            includes interactions and is a guide, not a significance test.
        </p>
        <div class="reduction-controls">
            <label
                >Challenger name<input
                    aria-label="Reduced challenger name"
                    bind:value={newName}
                    disabled={busy}
                /></label
            >
            <button class="primary" disabled={busy || !valid} onclick={create}
                >{busy ? 'Creating challenger…' : 'Create and fit challenger'}</button
            >
        </div>
        {#if info.model_names?.includes(newName.trim())}<p class="help-text">
                Choose a new model name.
            </p>{/if}
        {#if !selected.length}<p class="help-text">Keep at least one predictor.</p>{/if}
        {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {/if}
</section>

<style>
    .reduction {
        margin-top: 24px;
        padding-top: 20px;
        border-top: 1px solid var(--border, #ddd);
    }
    .reduction-controls {
        display: flex;
        flex-wrap: wrap;
        align-items: end;
        gap: 12px;
        margin: 16px 0;
    }
    .reduction-controls label {
        display: grid;
        gap: 6px;
    }
    input[type='number'] {
        width: 90px;
    }
    .predictor-options {
        max-height: 300px;
        overflow-y: auto;
        margin-top: 12px;
        border: 1px solid var(--border, #ddd);
        border-radius: 6px;
    }
    .predictor-options label {
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 8px 12px;
        border-bottom: 1px solid var(--border, #eee);
    }
    .predictor-options label span:last-child {
        margin-left: auto;
    }
</style>
