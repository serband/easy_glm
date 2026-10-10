<script>
    export let api;
    export let state;
    export let planState = null;
    export let onChanged = () => {};
    export let onNavigate = () => {};
    export let onRevisionIntent = () => {};
    let error = '',
        busy = false,
        newKind = 'glm',
        revising = false;
    $: plan = planState?.plan || null;
    $: stages = plan?.stages || [];
    $: fittedCount = planState?.fit_refs?.length || 0;
    $: adoptedRoot = Boolean(plan?.adopted_model_name);
    function stage(kind, index) {
        const label =
            kind === 'glm'
                ? 'Main effects'
                : kind === 'catboost'
                  ? 'CatBoost correction'
                  : kind === 'catalogue'
                    ? 'Catalogue correction'
                    : 'Interaction';
        return {
            component_id: crypto.randomUUID(),
            name: index ? `${label} ${index + 1}` : label,
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
    async function save(operation, nextPlan) {
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
    function start(mode) {
        if (
            plan &&
            !window.confirm('Replace the current working plan? Saved fitted versions are kept.')
        )
            return;
        save('start', {
            mode,
            family: 'poisson',
            link: 'log',
            tweedie_power: 1.5,
            divide_target_by_weight: Boolean(state?.setup?.assignments?.weight),
            stages: [stage('glm', 0)],
        });
    }
    function stageLocked(index) {
        return (adoptedRoot && index === 0) || (index < fittedCount && !revising);
    }
    function updatePlan(patch) {
        save(fittedCount && revising ? 'revise' : 'save', { ...plan, ...patch });
    }
    function updateStage(index, patch) {
        if (stageLocked(index)) return;
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
    function addStage(kind) {
        save('save', {
            ...plan,
            mode: 'sequential',
            stages: [...stages, stage(kind, stages.length)],
        });
    }
    function moveStage(index, direction) {
        const target = index + direction;
        if (target < 0 || target >= stages.length) return;
        const next = [...stages];
        [next[index], next[target]] = [next[target], next[index]];
        if (stageLocked(index) || stageLocked(target)) return;
        save(Math.min(index, target) < fittedCount ? 'revise' : 'save', { ...plan, stages: next });
    }
    function removeStage(index) {
        if (stageLocked(index)) return;
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
    <div class="heading">
        <div>
            <div class="eyebrow">MODEL PLAN</div>
            <h1>Project model plan</h1>
            <p>Choose the modelling route here. Variables and fitting follow this plan.</p>
        </div>
    </div>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if !plan}
        <section class="model-card start-plan">
            <h2>How will you build this model?</h2>
            <p>Choose once for this working plan. Saved fitted models are kept separately.</p>
            <div class="plan-actions">
                <button class="primary" disabled={busy} onclick={() => start('single')}
                    >Single model</button
                ><button class="primary" disabled={busy} onclick={() => start('sequential')}
                    >Build in stages</button
                >
            </div>
            <p class="help-text">
                Next, set the shared roles and then choose variables for each stage.
            </p>
        </section>
        {#if (planState?.history || []).length}<section class="model-card saved-versions">
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
    {:else}
        <section class="model-card">
            <div class="plan-header">
                <div>
                    <h2>{plan.mode === 'single' ? 'Single model' : 'Model stages'}</h2>
                    <p>
                        {plan.mode === 'single'
                            ? 'One model using the shared data.'
                            : 'Each later stage uses the complete score from the preceding stages as its offset.'}
                    </p>
                </div>
                <div class="plan-actions">
                    {#if fittedCount && !revising}<button
                            onclick={() => {
                                revising = true;
                                onRevisionIntent();
                            }}
                            disabled={busy}>Create revised plan</button
                        >{:else if revising}<button
                            class="primary"
                            onclick={() => save('revise', plan)}
                            disabled={busy}>Save revised plan</button
                        >{/if}<button onclick={() => start(plan.mode)} disabled={busy}
                        >Start a new plan</button
                    >
                </div>
            </div>
            {#if adoptedRoot}<div class="message">
                    Existing fitted model: {plan.adopted_model_name}. It is an accepted scorer and
                    cannot be changed here. Add a stage to build on it, or start a new plan.
                </div>{/if}{#if fittedCount}<p class="help-text">
                    {revising
                        ? 'You are revising the fitted prefix. Save revised plan to create the new version.'
                        : 'Choose Create revised plan before changing a fitted stage.'} Earlier fitted
                    stages are retained as saved versions. Editing one creates a revised plan and refits
                    that stage and those after it.
                </p>{/if}
            <div class="common-settings">
                <label
                    >Family<select
                        aria-label="Model family"
                        disabled={busy || adoptedRoot || (fittedCount && !revising)}
                        value={plan.family}
                        onchange={(event) =>
                            updatePlan({
                                family: event.currentTarget.value,
                                link: plan.mode === 'sequential' ? 'log' : null,
                            })}
                        ><option value="poisson">Poisson</option><option value="tweedie"
                            >Tweedie</option
                        ><option value="gaussian">Gaussian</option
                        >{#if plan.mode === 'single'}<option value="gamma">Gamma</option><option
                                value="inverse_gaussian">Inverse Gaussian</option
                            ><option value="binomial">Binomial</option>{/if}</select
                    ></label
                ><label
                    >Link<select
                        aria-label="Model link"
                        value={plan.mode === 'sequential' ? 'log' : plan.link || ''}
                        onchange={(event) =>
                            updatePlan({ link: event.currentTarget.value || null })}
                        disabled={busy ||
                            plan.mode === 'sequential' ||
                            adoptedRoot ||
                            (fittedCount && !revising)}
                        >{#if plan.mode === 'single'}<option value="">Family default</option
                            >{/if}<option value="log">Log</option
                        >{#if plan.mode === 'single'}<option value="logit">Logit</option
                            >{/if}</select
                    ></label
                >{#if plan.mode === 'sequential'}<p class="help-text">
                        Staged models use a log link so scores can be multiplied.
                    </p>{/if}{#if plan.family === 'tweedie'}<label
                        >Tweedie power<input
                            aria-label="Tweedie power"
                            type="number"
                            min="1.01"
                            max="1.99"
                            step="0.01"
                            value={plan.tweedie_power}
                            onchange={(event) =>
                                updatePlan({ tweedie_power: Number(event.currentTarget.value) })}
                            disabled={busy || adoptedRoot || (fittedCount && !revising)}
                        /></label
                    >{/if}<label
                    ><input
                        type="checkbox"
                        disabled={busy || adoptedRoot || (fittedCount && !revising)}
                        checked={plan.divide_target_by_weight}
                        onchange={(event) =>
                            updatePlan({ divide_target_by_weight: event.currentTarget.checked })}
                    />
                    Divide target by weight
                    <span class="help-text">(use a unit response)</span></label
                >
            </div>
            <ol class="stage-list">
                {#each stages as item, index}<li>
                        <label
                            >Stage name<input
                                aria-label={`Stage ${index + 1} name`}
                                value={item.name}
                                onchange={(event) =>
                                    updateStage(index, { name: event.currentTarget.value })}
                                disabled={busy || stageLocked(index)}
                            /></label
                        ><label
                            >Method<select
                                aria-label={`Stage ${index + 1} method`}
                                value={item.kind}
                                onchange={(event) => changeMethod(index, event.currentTarget.value)}
                                disabled={busy || stageLocked(index)}
                                ><option value="glm">GLM</option><option value="catboost"
                                    >CatBoost</option
                                ><option value="interaction">Interaction</option><option
                                    value="catalogue">Catalogue</option
                                ></select
                            ></label
                        >
                        <div class="stage-actions">
                            <button onclick={() => onNavigate('variables', item.component_id)}
                                >Choose variables</button
                            >{#if stages.length > 1}<button
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
                    </li>{/each}
            </ol>
            <div class="plan-actions">
                <label
                    >Add stage<select aria-label="Add stage method" bind:value={newKind}
                        ><option value="glm">GLM</option><option value="catboost">CatBoost</option
                        ><option value="interaction">Interaction</option><option value="catalogue"
                            >Catalogue</option
                        ></select
                    ></label
                ><button onclick={() => addStage(newKind)} disabled={busy}
                    >{plan.mode === 'single'
                        ? 'Add a stage and build in stages'
                        : 'Add stage'}</button
                >
            </div>
        </section>
        <section class="model-card saved-versions">
            <h2>Saved fitted versions</h2>
            {#if (planState?.history || []).length}<ul>
                    {#each planState.history as fitted}<li>{fitted.name}</li>{/each}
                </ul>{:else}<p>
                    No fitted versions yet. Starting a new plan clears this working plan only.
                </p>{/if}
        </section>
    {/if}
</section>

<style>
    .model-plan {
        max-width: 980px;
    }
    .plan-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 16px;
    }
    .common-settings {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        align-items: end;
        margin: 16px 0;
    }
    .plan-header {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
    }
    .stage-list {
        padding-left: 24px;
        margin: 20px 0;
    }
    .stage-list li {
        display: grid;
        grid-template-columns: 1fr auto auto;
        gap: 12px;
        align-items: center;
        border-top: 1px solid var(--line);
        padding: 12px 0;
    }
    .stage-actions {
        display: flex;
        gap: 5px;
        flex-wrap: wrap;
    }
    @media (max-width: 620px) {
        .plan-header {
            display: block;
        }
        .stage-list li {
            grid-template-columns: 1fr;
            gap: 5px;
        }
    }
</style>
