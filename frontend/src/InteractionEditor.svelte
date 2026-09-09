<script>
    import { formatNumber as num } from './format.js';
    export let interactions = [];
    export let predictors = [];
    export let valid = true;
    export let onchange = () => {};
    let first = '',
        second = '',
        minimum = 0.5;
    const eligibleShare = (value) => Number.isFinite(value) && value >= 0 && value < 1;
    const eligibleWeight = (value) => Number.isFinite(value) && value >= 0;
    $: duplicate = interactions.some(
        (pair) =>
            (pair.a === first && pair.b === second) || (pair.a === second && pair.b === first),
    );
    $: canAdd =
        predictors.includes(first) &&
        predictors.includes(second) &&
        first !== second &&
        !duplicate &&
        Number.isFinite(minimum) &&
        eligibleShare(minimum / 100);
    $: valid = interactions.every(
        (pair) =>
            pair.a !== pair.b &&
            predictors.includes(pair.a) &&
            predictors.includes(pair.b) &&
            eligibleShare(pair.min_cell_exposure) &&
            eligibleWeight(pair.penalty_weight),
    );
    function change(index, field, value) {
        onchange(interactions.map((pair, i) => (i === index ? { ...pair, [field]: value } : pair)));
    }
    function share(index, value) {
        if (
            value !== '' &&
            Number(value) ===
                Number(num(interactions[index].min_cell_exposure * 100, { grouping: false }))
        )
            return;
        change(index, 'min_cell_exposure', value === '' ? null : Number(value) / 100);
    }
    function weight(index, value) {
        if (
            value !== '' &&
            Number(value) === Number(num(interactions[index].penalty_weight, { grouping: false }))
        )
            return;
        change(index, 'penalty_weight', value === '' ? null : Number(value));
    }
    function add() {
        if (!canAdd) return;
        onchange([
            ...interactions,
            { a: first, b: second, min_cell_exposure: minimum / 100, penalty_weight: 1 },
        ]);
        first = '';
        second = '';
    }
</script>

<section class="interaction-editor" aria-label="Defined interactions">
    <div class="interaction-heading">
        <h3 id="model-interactions">Interactions</h3>
        <span>{interactions.length} defined</span>
    </div>
    <p class="help-text">Fitted in stage 2, with main effects held fixed.</p>
    {#if interactions.length}
        <ul class="interaction-list">
            {#each interactions as pair, index}
                <li class="interaction-row">
                    <div class="pair-line">
                        <strong class="pair-name">{pair.a} × {pair.b}</strong>
                        <label class="share-control"
                            >Minimum cell exposure
                            <span
                                ><input
                                    aria-label={'Minimum cell exposure for ' +
                                        pair.a +
                                        ' × ' +
                                        pair.b}
                                    type="number"
                                    min="0"
                                    max="99.999"
                                    step="any"
                                    value={pair.min_cell_exposure == null
                                        ? ''
                                        : num(pair.min_cell_exposure * 100, { grouping: false })}
                                    oninput={(e) => share(index, e.currentTarget.value)}
                                />%</span
                            >
                        </label>
                        <button
                            class="remove-pair"
                            aria-label={'Remove interaction ' + pair.a + ' × ' + pair.b}
                            onclick={() => onchange(interactions.filter((_, i) => i !== index))}
                            >Remove</button
                        >
                    </div>
                    {#if !predictors.includes(pair.a) || !predictors.includes(pair.b)}
                        <p class="interaction-error" role="alert">
                            Select both main effects above, or remove this interaction.
                        </p>
                    {/if}
                    {#if !eligibleShare(pair.min_cell_exposure)}
                        <p class="interaction-error" role="alert">
                            Minimum cell exposure must be at least 0% and below 100%.
                        </p>
                    {/if}
                    <details class="interaction-advanced">
                        <summary>Penalty settings</summary>
                        <label
                            >Penalty weight<input
                                aria-label={'Penalty weight for ' + pair.a + ' × ' + pair.b}
                                type="number"
                                min="0"
                                step="any"
                                value={pair.penalty_weight == null
                                    ? ''
                                    : num(pair.penalty_weight, { grouping: false })}
                                oninput={(e) => weight(index, e.currentTarget.value)}
                            /></label
                        >
                        {#if pair.alpha != null}<span class="help-text"
                                >Stored alpha {num(pair.alpha)}</span
                            >{/if}
                        {#if !eligibleWeight(pair.penalty_weight)}<p
                                class="interaction-error"
                                role="alert"
                            >
                                Penalty weight must be 0 or greater.
                            </p>{/if}
                    </details>
                </li>
            {/each}
        </ul>
    {:else}<p class="no-interactions">No interactions defined.</p>{/if}
    <div class="add-interaction" aria-label="Add an interaction">
        <label
            >First factor<select aria-label="Interaction first factor" bind:value={first}>
                <option value="">Select a factor</option>
                {#each predictors as name}<option value={name}>{name}</option>{/each}
            </select></label
        >
        <label
            >Second factor<select aria-label="Interaction second factor" bind:value={second}>
                <option value="">Select a factor</option>
                {#each predictors.filter((name) => name !== first) as name}<option value={name}
                        >{name}</option
                    >{/each}
            </select></label
        >
        <label class="share-control"
            >Minimum cell exposure<span>
                <input
                    aria-label="New interaction minimum cell exposure"
                    type="number"
                    min="0"
                    max="99.999"
                    step="any"
                    bind:value={minimum}
                />%</span
            ></label
        >
        <button disabled={!canAdd} onclick={add}>Add interaction</button>
    </div>
    {#if duplicate}<p class="help-text">This pair is already defined.</p>
    {:else if predictors.length < 2}<p class="help-text">
            Select at least two main effects to add an interaction.
        </p>{/if}
</section>

<style>
    .interaction-editor {
        border-top: 1px solid #d8e3dc;
        margin-top: 22px;
        padding-top: 20px;
        min-width: 0;
    }
    .interaction-heading {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 16px;
    }
    h3 {
        margin: 0;
        font-size: 18px;
    }
    .interaction-heading > span,
    .no-interactions {
        color: #73877d;
        font-size: 13px;
    }
    .interaction-editor > .help-text {
        margin: 8px 0 14px;
    }
    .interaction-list {
        padding: 0;
        margin: 0;
        list-style: none;
    }
    .interaction-row {
        border-bottom: 1px solid #e0e8e3;
        padding: 12px 0;
    }
    .pair-line {
        display: flex;
        align-items: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .pair-name {
        flex: 1;
        min-width: 150px;
        overflow-wrap: anywhere;
    }
    label {
        display: flex;
        flex-direction: column;
        gap: 6px;
        color: #516b60;
        font-size: 13px;
    }
    .share-control > span {
        display: flex;
        gap: 6px;
        align-items: center;
    }
    input {
        width: 80px;
        min-height: 34px;
    }
    .remove-pair {
        align-self: flex-end;
        min-height: 34px;
        font-size: 13px;
    }
    .interaction-advanced {
        margin-top: 10px;
        border: 0;
        padding: 0;
    }
    .interaction-advanced summary {
        color: #516b60;
        font-size: 13px;
        font-weight: 400;
    }
    .interaction-advanced label {
        margin-top: 10px;
    }
    .add-interaction {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto auto;
        align-items: end;
        gap: 12px;
        margin-top: 18px;
    }
    .add-interaction select {
        width: 100%;
        min-width: 0;
    }
    .add-interaction button {
        white-space: nowrap;
    }
    .interaction-error {
        color: #9e3f35;
        font-size: 13px;
        margin: 8px 0 0;
    }
    @media (max-width: 1100px) {
        .add-interaction {
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
        }
        .add-interaction button {
            justify-self: end;
        }
    }
</style>
