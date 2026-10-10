<script>
    import { tick } from 'svelte';

    export let api;
    export let state;
    export let planState = null;
    export let onChanged = () => {};
    export let onNavigate = () => {};
    export let onRevisionIntent = () => {};
    export let confirmDiscardStageDraft = () => true;
    export let showHistory = true;
    export let singleWorkflowManaged = false;
    export let routeChoiceOnly = false;
    export let onCancelNewPlan = () => {};
    let error = '',
        busy = false,
        starting = false,
        revising = false;
    $: plan = planState?.plan || null;
    $: stages = plan?.stages || [];
    $: fittedCount = planState?.fit_refs?.length || 0;
    $: adoptedRoot = Boolean(plan?.adopted_model_name);
    function stage(kind, index, name = index === 0 ? 'Main effects' : `Model ${index + 1}`) {
        return {
            component_id: crypto.randomUUID(),
            name,
            kind,
            factors: [],
            settings:
                kind === 'glm'
                    ? {
                          alpha: 0.001,
                          cv: null,
                          n_alphas: 20,
                          l1_ratio: 1,
                          min_alpha_ratio: 0.01,
                          base: 'modal',
                      }
                    : kind === 'interaction'
                      ? {
                            candidates: [
                                { iterations: 300, depth: 4, learning_rate: 0.05, l2_leaf_reg: 3 },
                            ],
                            min_weight_share: 0.0025,
                            cv_folds: 3,
                            seed: 42,
                            time_limit_minutes: 15,
                        }
                      : {
                            tune: true,
                            tuning_budget: 'trials',
                            trials: 8,
                            cv: 3,
                            seed: 42,
                            thread_count: 4,
                            iterations: 500,
                            depth: 6,
                            learning_rate: 0.05,
                            l2_leaf_reg: 3,
                            time_limit_minutes: 15,
                        },
            amendments: [],
            design: {
                defaults: {
                    n_bins: 8,
                    min_level_share: 0.0025,
                    null_indicator: true,
                    max_integer_knots: 30,
                },
                variables: {},
            },
        };
    }
    function assignmentName(name) {
        return state?.setup?.renames?.[name] || name;
    }
    $: externalOffset = state?.setup?.assignments?.offset
        ? assignmentName(state.setup.assignments.offset)
        : state?.setup?.assignments?.current_premium
          ? `log(${assignmentName(state.setup.assignments.current_premium)}) (derived from current premium)`
          : null;
    function stageOffset(index) {
        if (adoptedRoot && index === 0)
            return `Accepted saved scorer: ${plan.adopted_model_name}.`;
        if (index === 0)
            return externalOffset
                ? `Starts from project data. No earlier model. External offset: ${externalOffset}.`
                : 'Starts from project data. No earlier model.';
        const priorNames = stages.slice(0, index).map((item) => item.name);
        if (adoptedRoot) priorNames[0] = plan.adopted_model_name;
        return `Offset: complete score from ${priorNames.join(' → ')}.`;
    }
    function nextModelName() {
        const names = new Set(stages.map((item) => item.name));
        let number = stages.length + 1;
        while (names.has(`Model ${number}`)) number += 1;
        return `Model ${number}`;
    }
    async function save(operation, nextPlan) {
        if (!confirmDiscardStageDraft()) return;
        busy = true;
        error = '';
        try {
            onChanged(
                await api('model-plan', {
                    session_id: state.session_id,
                    revision: state.revision,
                    operation,
                    plan: nextPlan,
                }),
            );
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function adopt(name) {
        if (!confirmDiscardStageDraft()) return;
        busy = true;
        error = '';
        try {
            onChanged(
                await api('model-plan', {
                    session_id: state.session_id,
                    revision: state.revision,
                    operation: 'adopt',
                    adopt_model_name: name,
                }),
            );
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function start(mode) {
        if (starting) return;
        if (!confirmDiscardStageDraft()) return;
        if (
            plan &&
            !window.confirm('Replace the current working plan? Saved fitted versions are kept.')
        )
            return;
        starting = true;
        try {
            await save('start', {
                mode,
                family: 'poisson',
                link: 'log',
                tweedie_power: 1.5,
                divide_target_by_weight: Boolean(state?.setup?.assignments?.weight),
                stages: [stage('glm', 0)],
            });
        } finally {
            starting = false;
        }
    }
    function stageLocked(index) {
        return (adoptedRoot && index === 0) || (index < fittedCount && !revising);
    }
    function updatePlan(patch) {
        if (!confirmDiscardStageDraft()) return;
        save(fittedCount && revising ? 'revise' : 'save', { ...plan, ...patch });
    }
    function updateStage(index, patch) {
        if (stageLocked(index) || !confirmDiscardStageDraft()) return;
        save(index < fittedCount ? 'revise' : 'save', {
            ...plan,
            stages: stages.map((item, itemIndex) =>
                itemIndex === index ? { ...item, ...patch } : item,
            ),
        });
    }
    function changeMethod(index, kind) {
        const previous = stages[index];
        const settings = stage(kind, index).settings;
        const design =
            ['catboost', 'catalogue'].includes(kind) && previous.design
                ? {
                      ...previous.design,
                      variables: Object.fromEntries(
                          (previous.factors || []).map((factor) => [
                              factor,
                              previous.design.variables?.[factor]?.kind === 'categorical'
                                  ? { kind: 'categorical' }
                                  : {},
                          ]),
                      ),
                  }
                : previous.design;
        updateStage(index, {
            kind,
            settings,
            design,
            ...(kind === 'catalogue' ? { factors: [] } : {}),
        });
    }
    function changeSelect(event, currentValue, apply) {
        if (!confirmDiscardStageDraft()) {
            event.currentTarget.value = currentValue ?? '';
            return;
        }
        apply();
    }
    async function addStage() {
        if (!plan || busy || !confirmDiscardStageDraft()) return;
        const nextPlan = {
            ...plan,
            mode: 'sequential',
            stages: [...stages, stage('glm', stages.length, nextModelName())],
        };
        busy = true;
        error = '';
        try {
            const saved = await api('model-plan', {
                session_id: state.session_id,
                revision: state.revision,
                operation: 'save',
                plan: nextPlan,
            });
            onChanged(saved);
            const stageId = saved.plan?.stages?.at(-1)?.component_id;
            busy = false;
            await tick();
            await tick();
            const card = document.getElementById(`model-stage-${stageId}`);
            card?.scrollIntoView({ behavior: 'smooth', block: 'center' });
            card?.querySelector('select')?.focus();
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function moveStage(index, direction) {
        const target = index + direction;
        if (target < 0 || target >= stages.length) return;
        const next = [...stages];
        [next[index], next[target]] = [next[target], next[index]];
        if (stageLocked(index) || stageLocked(target) || !confirmDiscardStageDraft()) return;
        save(Math.min(index, target) < fittedCount ? 'revise' : 'save', { ...plan, stages: next });
    }
    function removeStage(index) {
        if (stageLocked(index) || !confirmDiscardStageDraft()) return;
        if (
            stages.length < 2 ||
            !window.confirm(`Remove ${stages[index].name} from this working plan?`)
        )
            return;
        save(index < fittedCount ? 'revise' : 'save', {
            ...plan,
            stages: stages.filter((_, itemIndex) => itemIndex !== index),
        });
    }
</script>

<section class="model-plan" aria-label="Model plan">
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if !plan || routeChoiceOnly}
        <section class="model-card start-plan">
            <div class="plan-header">
                <div>
                    <h2>{routeChoiceOnly ? 'Start a new plan' : 'Model structure'}</h2>
                    <p>
                        {routeChoiceOnly
                            ? 'Choose a route first. Your current plan is retained until you confirm the replacement.'
                            : 'Choose how this model will be built. Variables and fitting follow this choice.'}
                    </p>
                </div>
                {#if routeChoiceOnly}<button onclick={onCancelNewPlan} disabled={busy}
                        >Keep current plan</button
                    >{/if}
            </div>
            <div class="route-actions">
                <button
                    class="route-choice"
                    aria-label="Single model"
                    disabled={starting}
                    onclick={() => start('single')}
                >
                    <span>Single model</span>
                    <small>Fit one model using the project data.</small>
                </button>
                <button
                    class="route-choice"
                    aria-label="Build in stages"
                    disabled={starting}
                    onclick={() => start('sequential')}
                >
                    <span>Build in stages</span>
                    <small>Fit models in sequence, each building on the score before it.</small>
                </button>
            </div>
        </section>
        {#if !routeChoiceOnly && (planState?.history || []).length}<section
                class="model-card saved-versions"
            >
                <h2>Use a saved fitted version</h2>
                <p>
                    Use a saved fit as the starting point for a new staged plan. Its complete score
                    is retained.
                </p>
                <ul>
                    {#each planState.history as fitted}<li>
                            {fitted.name}{#if fitted.can_adopt}<button
                                    onclick={() => adopt(fitted.name)}
                                    disabled={busy}>Use as current plan</button
                                >{:else}<span class="help-text"
                                    >This saved fit cannot be used as a log-link starting model.</span
                                >{/if}
                        </li>{/each}
                </ul>
            </section>{/if}
    {:else if !(singleWorkflowManaged && plan.mode === 'single' && !fittedCount)}
        <section class="model-card configured-plan">
            <div class="plan-header">
                <div>
                    <h2>
                        {singleWorkflowManaged && plan.mode === 'single'
                            ? 'Plan actions'
                            : plan.mode === 'single'
                              ? 'Model details'
                              : 'Model stages'}
                    </h2>
                    <p>
                        {singleWorkflowManaged && plan.mode === 'single'
                            ? 'Save a new plan or create a revision without overwriting fitted versions.'
                            : plan.mode === 'single'
                              ? 'Choose the fitting method for this model. Its variables are set below.'
                              : 'Each later stage uses the complete score from the preceding stages as its offset.'}
                    </p>
                </div>
                {#if fittedCount}<div class="plan-actions">
                        {#if !revising}<button
                                onclick={() => {
                                    if (!confirmDiscardStageDraft()) return;
                                    revising = true;
                                    onRevisionIntent();
                                }}
                                disabled={busy}>Create revised plan</button
                            >{:else}<button
                                class="primary"
                                onclick={() => save('revise', plan)}
                                disabled={busy}>Save revised plan</button
                            >{/if}
                    </div>{/if}
            </div>
            {#if !(singleWorkflowManaged && plan.mode === 'single')}
                {#if adoptedRoot}<div class="message">
                        Existing fitted model: {plan.adopted_model_name}. It is an accepted scorer
                        and cannot be changed here. Add a model to build on it, or start a new plan.
                    </div>{/if}{#if fittedCount}<p class="help-text">
                        {revising
                            ? 'You are revising the fitted prefix. Save revised plan to create the new version.'
                            : 'Choose Create revised plan before changing a fitted stage.'} Earlier fitted
                        stages are retained as saved versions. Editing one creates a revised plan and
                        refits that stage and those after it.
                    </p>{/if}
                <div class="two-column-grid common-settings">
                    <label class="field">
                        <span class="field-label">Response family</span>
                        <select
                            aria-label="Model family"
                            disabled={busy || adoptedRoot || (fittedCount && !revising)}
                            value={plan.family}
                            onchange={(event) =>
                                changeSelect(event, plan.family, () =>
                                    updatePlan({
                                        family: event.currentTarget.value,
                                        link: plan.mode === 'sequential' ? 'log' : null,
                                    }),
                                )}
                        >
                            <option value="poisson">Poisson</option>
                            <option value="tweedie">Tweedie</option>
                            <option value="gaussian">Gaussian</option>
                            {#if plan.mode === 'single'}
                                <option value="gamma">Gamma</option>
                                <option value="inverse_gaussian">Inverse Gaussian</option>
                                <option value="binomial">Binomial</option>
                            {/if}
                        </select>
                    </label>
                    <label class="field">
                        <span class="field-label">Response link</span>
                        <select
                            aria-label="Model link"
                            value={plan.mode === 'sequential' ? 'log' : plan.link || ''}
                            onchange={(event) =>
                                changeSelect(event, plan.link, () =>
                                    updatePlan({ link: event.currentTarget.value || null }),
                                )}
                            disabled={busy ||
                                plan.mode === 'sequential' ||
                                adoptedRoot ||
                                (fittedCount && !revising)}
                        >
                            {#if plan.mode === 'single'}<option value="">Family default</option
                                >{/if}
                            <option value="log">Log</option>
                            {#if plan.mode === 'single'}<option value="logit">Logit</option>{/if}
                        </select>
                    </label>
                </div>
                {#if plan.mode === 'sequential'}
                    <p class="help-text link-note">
                        Staged models use a log link so scores can be multiplied.
                    </p>
                {/if}
                {#if plan.family === 'tweedie'}
                    <div class="two-column-grid tweedie-row">
                        <label class="field">
                            <span class="field-label">Tweedie power</span>
                            <input
                                aria-label="Tweedie power"
                                type="number"
                                min="1.01"
                                max="1.99"
                                step="0.01"
                                value={plan.tweedie_power}
                                onchange={(event) =>
                                    updatePlan({
                                        tweedie_power: Number(event.currentTarget.value),
                                    })}
                                disabled={busy || adoptedRoot || (fittedCount && !revising)}
                            />
                        </label>
                    </div>
                {/if}
                <label class="checkbox-field">
                    <input
                        type="checkbox"
                        disabled={busy || adoptedRoot || (fittedCount && !revising)}
                        checked={plan.divide_target_by_weight}
                        onchange={(event) =>
                            updatePlan({ divide_target_by_weight: event.currentTarget.checked })}
                    />
                    <span
                        >Divide target by weight <span class="help-text">(use a unit response)</span
                        ></span
                    >
                </label>
                {#if plan.mode === 'single'}
                    <div class="two-column-grid single-model-details">
                        <label class="field">
                            <span class="field-label">Model name</span>
                            <input
                                aria-label="Model name"
                                value={stages[0]?.name || ''}
                                onchange={(event) =>
                                    updateStage(0, { name: event.currentTarget.value })}
                                disabled={busy || stageLocked(0)}
                            />
                        </label>
                        <label class="field">
                            <span class="field-label">Fitting method</span>
                            <select
                                aria-label="Fitting method"
                                value={stages[0]?.kind || 'glm'}
                                onchange={(event) =>
                                    changeSelect(event, stages[0]?.kind, () =>
                                        changeMethod(0, event.currentTarget.value),
                                    )}
                                disabled={busy || stageLocked(0)}
                            >
                                <option value="glm">GLM</option>
                                <option value="catboost">CatBoost</option>
                                <option value="interaction">Interaction</option>
                                <option value="catalogue">Catalogue</option>
                            </select>
                        </label>
                    </div>
                {:else}
                    <ol class="stage-list">
                        {#each stages as item, index}
                            <li class="stage-card" id={`model-stage-${item.component_id}`}>
                                <h3>Model {index + 1}</h3>
                                <p class="stage-offset">{stageOffset(index)}</p>
                                <div class="two-column-grid stage-fields">
                                    <label class="field">
                                        <span class="field-label">Model name</span>
                                        <input
                                            aria-label={`Model ${index + 1} name`}
                                            value={adoptedRoot && index === 0
                                                ? plan.adopted_model_name
                                                : item.name}
                                            readonly={adoptedRoot && index === 0}
                                            onchange={(event) =>
                                                updateStage(index, {
                                                    name: event.currentTarget.value,
                                                })}
                                            disabled={busy ||
                                                (stageLocked(index) && !(adoptedRoot && index === 0))}
                                        />
                                    </label>
                                    <label class="field">
                                        <span class="field-label">Model type</span>
                                        {#if adoptedRoot && index === 0}
                                            <input
                                                aria-label={`Model ${index + 1} type`}
                                                value="Saved fitted model"
                                                readonly
                                            />
                                        {:else}
                                            <select
                                                aria-label={`Model ${index + 1} type`}
                                                value={item.kind}
                                                onchange={(event) =>
                                                    changeSelect(event, item.kind, () =>
                                                        changeMethod(index, event.currentTarget.value),
                                                    )}
                                                disabled={busy || stageLocked(index)}
                                            >
                                                <option value="glm">GLM</option>
                                                <option value="catboost">CatBoost</option>
                                                <option value="interaction">Interaction table</option>
                                                <option value="catalogue">Catalogue</option>
                                            </select>
                                        {/if}
                                    </label>
                                </div>
                                <div class="stage-footer">
                                    <span class="selected-count"
                                        >{adoptedRoot && index === 0
                                            ? 'Saved fitted variables are preserved'
                                            : `${item.factors?.length || 0} variable${
                                                  item.factors?.length === 1 ? '' : 's'
                                              } selected`}</span
                                    >
                                    <div class="stage-actions">
                                        {#if !(adoptedRoot && index === 0)}<button
                                                onclick={() => onNavigate('variables', item.component_id)}
                                                >Choose variables</button
                                            >{/if}{#if stages.length > 1}<button
                                                aria-label={`Move ${item.name} up`}
                                                onclick={() => moveStage(index, -1)}
                                                disabled={busy ||
                                                    index === 0 ||
                                                    stageLocked(index) ||
                                                    stageLocked(index - 1)}>Up</button
                                            ><button
                                                aria-label={`Move ${item.name} down`}
                                                onclick={() => moveStage(index, 1)}
                                                disabled={busy ||
                                                    index === stages.length - 1 ||
                                                    stageLocked(index) ||
                                                    stageLocked(index + 1)}>Down</button
                                            ><button
                                                onclick={() => removeStage(index)}
                                                disabled={busy || stageLocked(index)}>Remove</button
                                            >{/if}
                                    </div>
                                </div>
                            </li>
                        {/each}
                    </ol>
                    <div class="add-stage">
                        <button onclick={addStage} disabled={busy}>Add model</button>
                    </div>
                {/if}
            {/if}
        </section>
        {#if showHistory}
            <section class="model-card saved-versions">
                <h2>Saved fitted versions</h2>
                {#if (planState?.history || []).length}<ul>
                        {#each planState.history as fitted}<li>{fitted.name}</li>{/each}
                    </ul>{:else}<p>
                        No fitted versions yet. Starting a new plan clears this working plan only.
                    </p>{/if}
            </section>
        {/if}
    {/if}
</section>

<style>
    .model-plan {
        width: 100%;
    }
    .plan-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0;
    }
    .two-column-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
    }
    .field {
        display: grid;
        gap: 8px;
        min-width: 0;
    }
    .field-label {
        display: block;
    }
    .field input,
    .field select {
        width: 100%;
        min-width: 0;
        height: 38px;
        box-sizing: border-box;
    }
    .common-settings {
        margin-top: 24px;
    }
    .tweedie-row,
    .checkbox-field {
        margin-top: 16px;
    }
    .configured-plan .link-note {
        margin: 16px 0 0;
    }
    .checkbox-field {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .checkbox-field input {
        flex: none;
        margin: 0;
    }
    .plan-header {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
    }
    .route-actions {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 14px;
        margin-top: 20px;
    }
    .route-choice {
        display: grid;
        min-height: 112px;
        padding: 18px;
        border-color: #cbd8d2;
        background: #fff;
        text-align: left;
        white-space: normal;
        align-content: start;
        gap: 8px;
    }
    .route-choice:hover {
        background: #f5faf7;
        border-color: #548f7b;
    }
    .route-choice span {
        color: var(--text);
        font-size: 16px;
        font-weight: 650;
    }
    .route-choice small {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.5;
    }
    .stage-list {
        padding-left: 0;
        margin: 24px 0 0;
        list-style: none;
        display: grid;
        gap: 16px;
    }
    .single-model-details {
        margin-top: 24px;
    }
    .stage-card {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 16px;
        background: #fff;
    }
    .stage-card h3 {
        margin: 0;
        font-size: 16px;
    }
    .stage-offset {
        margin: 8px 0 16px;
        color: var(--muted);
    }
    .stage-footer {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-top: 16px;
    }
    .selected-count {
        color: var(--muted);
        font-size: 13px;
    }
    .configured-plan .plan-actions button,
    .stage-actions button {
        min-height: 38px;
    }
    .stage-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .add-stage {
        margin-top: 24px;
    }
    .add-stage button {
        min-height: 38px;
    }
    .configured-plan + .saved-versions {
        margin-top: 24px;
    }
    .plan-header .plan-actions {
        align-self: start;
    }
    @media (max-width: 900px) {
        .plan-header {
            display: block;
        }
        .plan-header .plan-actions {
            margin-top: 16px;
        }
        .two-column-grid {
            grid-template-columns: 1fr;
        }
        .route-actions {
            grid-template-columns: 1fr;
        }
    }
</style>
