<script>
    import { formatNumber as num } from './format.js';

    export let stages = [];
    export let eligible = [];
    export let defaults = null;
    export let statuses = [];
    export let job = null;
    export let mainChanged = false;
    export let newModel = false;
    export let savedStages = [];
    export let onchange = () => {};
    export let valid = true;

    let first = '';
    let second = '';
    const labels = {
        not_fitted: 'Not fitted',
        up_to_date: 'Up to date',
        needs_refitting: 'Needs refitting',
        fitting: 'Fitting',
        no_improvement: 'No improvement',
        failed: 'Failed',
    };
    const pairName = (stage) => `${stage.a} × ${stage.b}`;
    const samePair = (left, right) =>
        (left.a === right.a && left.b === right.b) || (left.a === right.b && left.b === right.a);
    const integer = (value, low, high) => Number.isInteger(value) && value >= low && value <= high;
    const positive = (value) => Number.isFinite(value) && value > 0;
    const automatic = (stage) => stage.search?.method === 'optuna';
    const compact = (value) =>
        Number(value)
            .toPrecision(4)
            .replace(/\.?0+$/, '');
    const preciseLoss = (value) => Number(value).toPrecision(7);
    const lossChange = (before, after) => {
        const change = after - before;
        return `${change > 0 ? '+' : ''}${change.toExponential(3)}`;
    };
    const stageValid = (stage) =>
        stage.a !== stage.b &&
        eligible.includes(stage.a) &&
        eligible.includes(stage.b) &&
        (automatic(stage)
            ? integer(stage.search.trials, 1, 16) &&
              integer(stage.search.prefix_trials, 1, 8) &&
              stage.search.prefix_trials <= stage.search.trials &&
              Array.isArray(stage.candidates)
            : stage.search == null &&
              Array.isArray(stage.candidates) &&
              stage.candidates.length >= 1 &&
              stage.candidates.length <= 3 &&
              stage.candidates.every(
                  (candidate) =>
                      integer(candidate.depth, 1, 6) &&
                      integer(candidate.iterations, 1, 200) &&
                      positive(candidate.learning_rate) &&
                      candidate.learning_rate <= 1 &&
                      positive(candidate.l2_leaf_reg),
              )) &&
        Number.isFinite(stage.min_weight_share) &&
        stage.min_weight_share >= 0 &&
        stage.min_weight_share < 1 &&
        integer(stage.seed, 0, 2147483647) &&
        stage.cv_folds === 5;

    $: statusById = new Map(statuses.map((stage) => [stage.stage_id, stage]));
    $: changedAt = mainChanged
        ? 0
        : stages.findIndex(
              (stage, index) =>
                  !savedStages[index] ||
                  JSON.stringify(stage) !== JSON.stringify(savedStages[index]),
          );
    $: duplicate = stages.some((stage) => samePair(stage, { a: first, b: second }));
    $: canAdd =
        defaults?.search?.method === 'optuna' &&
        stages.length < 8 &&
        eligible.includes(first) &&
        eligible.includes(second) &&
        first !== second &&
        !duplicate;
    $: valid =
        stages.every(stageValid) &&
        !stages.some((stage, index) =>
            stages.slice(index + 1).some((other) => samePair(stage, other)),
        );

    function update(index, field, value) {
        onchange(stages.map((stage, at) => (at === index ? { ...stage, [field]: value } : stage)));
    }
    function searchBudget(index, field, value) {
        const search = {
            ...stages[index].search,
            [field]: value === '' ? null : Number(value),
        };
        update(index, 'search', search);
    }
    function switchToAutomatic(index) {
        if (defaults?.search?.method !== 'optuna') return;
        onchange(
            stages.map((stage, at) =>
                at === index
                    ? { ...stage, search: structuredClone(defaults.search), candidates: [] }
                    : stage,
            ),
        );
    }
    function add() {
        if (!canAdd) return;
        onchange([
            ...stages,
            {
                min_weight_share: defaults.min_weight_share,
                seed: defaults.seed,
                cv_folds: defaults.cv_folds,
                search: structuredClone(defaults.search),
                stage_id: crypto.randomUUID(),
                a: first,
                b: second,
                candidates: [],
            },
        ]);
        first = '';
        second = '';
    }
    function move(index, direction) {
        const reordered = [...stages];
        [reordered[index], reordered[index + direction]] = [
            reordered[index + direction],
            reordered[index],
        ];
        onchange(reordered);
    }
    function stageStatus(stage, index) {
        if (
            ['queued', 'running'].includes(job?.status) &&
            job?.progress?.stage_number === index + 2
        )
            return 'Fitting';
        if (changedAt >= 0 && index >= changedAt) {
            return savedStages.some((saved) => saved.stage_id === stage.stage_id)
                ? 'Needs refitting'
                : 'Not fitted';
        }
        const packet = statusById.get(stage.stage_id);
        return packet ? labels[packet.status] || packet.status : 'Status unavailable';
    }
    function mainStatus(job, newModel, mainChanged, statuses) {
        if (['queued', 'running'].includes(job?.status) && job?.progress?.stage_number === 1) {
            return 'Fitting';
        }
        if (newModel) return 'Not fitted';
        if (mainChanged) return 'Needs refitting';
        const recorded = statuses.find((stage) => stage.stage_id === 'main');
        if (recorded) return labels[recorded.status] || recorded.status;
        // A main-only fit has no pair-stage status record yet.
        if (job?.applicable) return 'Up to date';
        if (['queued', 'running'].includes(job?.status)) return 'Fitting';
        if (job?.status === 'failed') return 'Failed';
        if (job?.status === 'complete') return 'Needs refitting';
        return 'Not fitted';
    }
    function baseline(stage, index) {
        const packet = statusById.get(stage.stage_id);
        if (Array.isArray(packet?.baseline) && (changedAt < 0 || index < changedAt)) {
            const names = packet.baseline.map((stageId) => {
                if (stageId === 'main') return 'Main effects';
                const upstream = savedStages.find((saved) => saved.stage_id === stageId);
                return upstream ? pairName(upstream) : stageId;
            });
            if (names.length) return names.join(' + ');
        }
        return index === 0
            ? 'Main effects'
            : `Main effects + ${stages.slice(0, index).map(pairName).join(' + ')}`;
    }
</script>

<section class="pair-stages" aria-label="Sequential pair stages">
    <h3 id="model-pair-stages">Interactions — fitted in order</h3>
    <p class="help-text">
        Fit each interaction in order. Each becomes a rate table; the next builds on the main
        effects and earlier tables. You can use a predictor here without selecting it as a main
        effect.
    </p>
    <div class="stage-card main-stage" aria-label="Stage 1 Main effects">
        <div class="stage-line">
            <strong>1 · Main effects</strong>
            <span class="stage-state">{mainStatus(job, newModel, mainChanged, statuses)}</span>
        </div>
        <p>GLM settings and selected main effects</p>
    </div>
    {#if ['queued', 'running'].includes(job?.status) && job?.progress}
        <p class="help-text" role="status">
            {job.progress.message || `Stage ${job.progress.stage_number || '?'} · training`}{job
                .progress.prefix_reused
                ? ' · earlier full fit reused'
                : ''}
        </p>
    {/if}
    {#each stages as stage, index (stage.stage_id)}
        {@const packet = statusById.get(stage.stage_id)}
        <div class="stage-card" aria-label={`Stage ${index + 2} ${pairName(stage)}`}>
            <div class="stage-line">
                <strong>{index + 2} · {pairName(stage)}</strong>
                <span class="stage-state">{stageStatus(stage, index)}</span>
            </div>
            <p class="stage-baseline">Baseline: {baseline(stage, index)}</p>
            <p>
                {automatic(stage)
                    ? `Automatic tuning · ${stage.search.trials} trials`
                    : 'Saved fixed settings'}
            </p>
            {#if (packet && changedAt < 0) || (packet && index < changedAt)}
                {#if packet.dimensions}
                    <p>Rate table: {packet.dimensions[0]} × {packet.dimensions[1]} cells</p>
                {/if}
                {#if packet.support}
                    <p>
                        Supported cells: {packet.support.supported_cells ?? 0} of {packet.support
                            .total_cells ?? 0}
                    </p>
                {/if}
                {#if packet.chosen_candidate === null && packet.status === 'no_improvement'}
                    <p>The fit selected no change because this pair did not improve the model.</p>
                {/if}
                {#if packet.chosen_candidate}
                    <details class="chosen-settings">
                        <summary>Selected settings</summary>
                        <p>
                            Depth {packet.chosen_candidate.depth} · {packet.chosen_candidate
                                .iterations} iterations · learning rate {compact(
                                packet.chosen_candidate.learning_rate,
                            )} · L2 {compact(packet.chosen_candidate.l2_leaf_reg)}
                        </p>
                    </details>
                {/if}
                {#if Number.isFinite(packet.loss?.prefix) && Number.isFinite(packet.loss?.table)}
                    <p>
                        Training CV loss: before {preciseLoss(packet.loss.prefix)} · with this table {preciseLoss(
                            packet.loss.table,
                        )} · change {lossChange(packet.loss.prefix, packet.loss.table)}
                    </p>
                {/if}
                {#if Number.isFinite(packet.loss?.approximation)}
                    <p>Table approximation loss: {preciseLoss(packet.loss.approximation)}</p>
                {/if}
                {#if packet.reused}<p>Reused from the previous fit.</p>{/if}
                {#if packet.message}<p>{packet.message}</p>{/if}
            {/if}
            <div class="stage-order">
                <button
                    type="button"
                    aria-label={`Move ${pairName(stage)} up`}
                    disabled={index === 0}
                    onclick={() => move(index, -1)}>↑ Up</button
                >
                <button
                    type="button"
                    aria-label={`Move ${pairName(stage)} down`}
                    disabled={index === stages.length - 1}
                    onclick={() => move(index, 1)}>↓ Down</button
                >
                <button
                    type="button"
                    aria-label={`Remove pair stage ${pairName(stage)}`}
                    onclick={() => onchange(stages.filter((_, at) => at !== index))}>Remove</button
                >
            </div>
            <details>
                <summary>Stage settings</summary>
                <div class="stage-settings">
                    <label
                        >First predictor<select
                            aria-label={`First predictor for stage ${index + 2}`}
                            value={stage.a}
                            onchange={(event) => update(index, 'a', event.currentTarget.value)}
                        >
                            {#each eligible as name}<option value={name}>{name}</option>{/each}
                        </select></label
                    >
                    <label
                        >Second predictor<select
                            aria-label={`Second predictor for stage ${index + 2}`}
                            value={stage.b}
                            onchange={(event) => update(index, 'b', event.currentTarget.value)}
                        >
                            {#each eligible as name}<option value={name}>{name}</option>{/each}
                        </select></label
                    >
                    <label
                        >Minimum fitting-weight share (%)<input
                            aria-label={`Minimum fitting-weight share for stage ${index + 2}`}
                            type="number"
                            min="0"
                            max="99.999"
                            step="any"
                            value={num(stage.min_weight_share * 100, { grouping: false })}
                            oninput={(event) =>
                                update(
                                    index,
                                    'min_weight_share',
                                    event.currentTarget.value === ''
                                        ? null
                                        : Number(event.currentTarget.value) / 100,
                                )}
                        /></label
                    >
                    <label
                        >Seed<input
                            aria-label={`Seed for stage ${index + 2}`}
                            type="number"
                            min="0"
                            max="2147483647"
                            step="1"
                            value={stage.seed}
                            oninput={(event) =>
                                update(
                                    index,
                                    'seed',
                                    event.currentTarget.value === ''
                                        ? null
                                        : Number(event.currentTarget.value),
                                )}
                        /></label
                    >
                </div>
                {#if automatic(stage)}
                    <p class="help-text">
                        Automatic tuning chooses CatBoost settings using five training folds. No
                        change is always considered.
                    </p>
                    <details class="tuning-budget">
                        <summary>Tuning budget</summary>
                        <div class="stage-settings">
                            <label
                                >Trials for this pair<input
                                    aria-label={`Tuning trials for stage ${index + 2}`}
                                    type="number"
                                    min="1"
                                    max="16"
                                    step="1"
                                    value={stage.search.trials}
                                    oninput={(event) =>
                                        searchBudget(index, 'trials', event.currentTarget.value)}
                                /></label
                            >
                            <label
                                >Trials for the earlier-stage baseline<input
                                    aria-label={`Prefix tuning trials for stage ${index + 2}`}
                                    type="number"
                                    min="1"
                                    max="8"
                                    step="1"
                                    value={stage.search.prefix_trials}
                                    oninput={(event) =>
                                        searchBudget(
                                            index,
                                            'prefix_trials',
                                            event.currentTarget.value,
                                        )}
                                /></label
                            >
                        </div>
                        <p class="help-text">
                            Used when there are earlier interactions. Cannot exceed this pair's
                            trial count.
                        </p>
                    </details>
                {:else}
                    <p class="help-text">
                        This stage keeps the fixed CatBoost settings saved with the model.
                    </p>
                    <button
                        type="button"
                        disabled={defaults?.search?.method !== 'optuna'}
                        onclick={() => switchToAutomatic(index)}>Switch to automatic tuning</button
                    >
                    <details class="saved-settings">
                        <summary>View saved fixed settings</summary>
                        {#each stage.candidates as candidate, candidateIndex}
                            <p>
                                Setting {candidateIndex + 1}: depth {candidate.depth}, {candidate.iterations}
                                iterations, learning rate {candidate.learning_rate}, L2 {candidate.l2_leaf_reg}
                            </p>
                        {/each}
                    </details>
                {/if}
                {#if !stageValid(stage)}<p class="stage-error" role="alert">
                        Check the pair predictors and bounded stage settings.
                    </p>{/if}
            </details>
        </div>
    {/each}
    {#if stages.length === 0}<p class="help-text">No interactions added yet.</p>{/if}
    <div class="add-stage">
        <label
            >First predictor<select aria-label="New pair first predictor" bind:value={first}
                ><option value="">Select predictor</option>{#each eligible as name}<option
                        value={name}>{name}</option
                    >{/each}</select
            ></label
        >
        <label
            >Second predictor<select aria-label="New pair second predictor" bind:value={second}
                ><option value="">Select predictor</option
                >{#each eligible.filter((name) => name !== first) as name}<option value={name}
                        >{name}</option
                    >{/each}</select
            ></label
        >
        <button type="button" disabled={!canAdd} onclick={add}>Add interaction</button>
    </div>
    {#if duplicate}<p class="stage-error">This pair is already defined.</p>{/if}
    {#if defaults?.search?.method !== 'optuna'}<p class="help-text">
            Automatic tuning defaults are loading from the workbench.
        </p>{/if}
    {#if !valid}<p class="stage-error" role="alert">
            Correct the pair-stage settings before saving.
        </p>{/if}
</section>

<style>
    .pair-stages {
        border-top: 1px solid #d8e3dc;
        margin-top: 18px;
        padding-top: 16px;
    }
    .stage-card {
        border: 1px solid #d8e3dc;
        border-radius: 6px;
        padding: 14px;
        margin-top: 12px;
        background: #fbfdfb;
    }
    .main-stage {
        background: #edf5ef;
    }
    .stage-line,
    .stage-order,
    .add-stage {
        display: flex;
        align-items: center;
        gap: 12px;
        flex-wrap: wrap;
    }
    .stage-line {
        justify-content: space-between;
    }
    .stage-state {
        font-size: 12px;
        font-weight: 650;
        color: #205e4e;
    }
    .stage-baseline {
        font-size: 12px;
        font-weight: 550;
    }
    .stage-card p {
        font-size: 12px;
    }
    .stage-order {
        margin-top: 12px;
    }
    .stage-order button {
        padding: 5px 9px;
    }
    .stage-settings {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .stage-settings label,
    .add-stage label {
        display: flex;
        flex-direction: column;
        gap: 5px;
        font-size: 12px;
    }
    .stage-settings input,
    .stage-settings select,
    .add-stage select {
        width: 100%;
    }
    .tuning-budget,
    .saved-settings,
    .chosen-settings {
        margin-top: 10px;
    }
    .add-stage {
        align-items: end;
        margin-top: 16px;
    }
    .add-stage label {
        min-width: 160px;
        flex: 1;
    }
    .stage-error {
        color: #9f352d;
        font-size: 12px;
    }
</style>
