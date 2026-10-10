<script>
    import { onDestroy, onMount } from 'svelte';
    export let api;
    export let state;
    export let planState = null;
    export let onChanged = () => {};
    export let onFitted = () => {};
    let error = '',
        busy = false,
        polling = false,
        workingSettings = {},
        settingsKey = '',
        pollCancelled = false,
        pollGeneration = 0;
    $: plan = planState?.plan || null;
    $: statuses = planState?.stage_statuses || [];
    $: fittingStage = statuses.find((item) => item.state === 'fitting') || null;
    $: next = statuses.find((item) => item.stage_id === planState?.next_stage_id) || null;
    $: nextIndex = next ? statuses.indexOf(next) : -1;
    $: stage = plan?.stages?.find((item) => item.component_id === next?.stage_id);
    $: predecessor = nextIndex > 0 ? statuses[nextIndex - 1] : null;
    $: stageKey = JSON.stringify({
        plan_id: plan?.plan_id,
        id: stage?.component_id,
        settings: stage?.settings || {},
    });
    $: if (stage && stageKey !== settingsKey && !busy) {
        workingSettings = structuredClone(stage.settings || {});
        settingsKey = stageKey;
    }
    function setSettings(patch) {
        workingSettings = { ...workingSettings, ...patch };
    }
    function settingsForSave() {
        const saved = structuredClone(workingSettings);
        if (saved.tune === false) {
            delete saved.tuning_budget;
            delete saved.trials;
            delete saved.time_limit_minutes;
            delete saved.cv;
        } else if (Object.hasOwn(saved, 'tuning_budget')) {
            if (saved.tuning_budget === 'time') delete saved.trials;
            else {
                saved.tuning_budget = 'trials';
                delete saved.time_limit_minutes;
            }
        }
        if (stage?.kind === 'interaction' && saved.tune === false) {
            delete saved.tuning_budget;
            delete saved.trials;
            delete saved.time_limit_minutes;
            delete saved.cv;
        }
        if (stage?.kind === 'glm') {
            if ((saved.penalty_method || (saved.cv ? 'cv' : 'alpha')) === 'cv') delete saved.alpha;
            else delete saved.cv;
        }
        return saved;
    }
    async function waitForFit(stageId, name) {
        const generation = ++pollGeneration;
        const planId = plan?.plan_id;
        polling = true;
        while (!pollCancelled && generation === pollGeneration) {
            await new Promise((resolve) => setTimeout(resolve, 1000));
            try {
                const fresh = await api('model-plan');
                if (
                    fresh?.plan?.plan_id !== planId ||
                    generation !== pollGeneration ||
                    pollCancelled
                ) {
                    polling = false;
                    return;
                }
                onChanged(fresh);
                const status = fresh.stage_statuses?.find((item) => item.stage_id === stageId);
                if (status?.state === 'fitted') {
                    onFitted(status.fit?.model_name || name);
                    polling = false;
                    return;
                }
                if (status?.state !== 'fitting') {
                    polling = false;
                    return;
                }
            } catch (e) {
                error = `Cannot refresh fit status. ${e.message}`;
                polling = false;
                return;
            }
            continue;
        }
        polling = false;
    }
    onMount(() => {
        if (fittingStage && !polling)
            waitForFit(fittingStage.stage_id, fittingStage.fit?.model_name || '');
    });
    onDestroy(() => {
        pollCancelled = true;
        pollGeneration += 1;
    });
    async function cancelFit() {
        const name = fittingStage?.fit?.model_name || fittingStage?.last_attempt?.model_name;
        if (!name) return;
        busy = true;
        error = '';
        try {
            await api(`models/${encodeURIComponent(name)}/cancel`, {
                session_id: state.session_id,
                revision: state.revision,
            });
            onChanged(await api('model-plan'));
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function fit() {
        if (!next || !stage || polling) return;
        busy = true;
        error = '';
        try {
            const stages = plan.stages.map((item) =>
                item.component_id === stage.component_id
                    ? { ...item, settings: settingsForSave() }
                    : item,
            );
            const saved = await api('model-plan', {
                session_id: state.session_id,
                revision: state.revision,
                operation: 'save',
                plan: { ...plan, stages },
            });
            onChanged(saved);
            const started = await api('model-plan/fit-next', {
                session_id: state.session_id,
                revision: saved.revision,
            });
            onChanged(started.snapshot || saved);
            waitForFit(started.stage_id, started.model_name || '');
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
</script>

<section class="plan-fitting workflow-page" aria-label="Fit model plan">
    <div class="heading">
        <div>
            <div class="eyebrow">MODEL PLAN</div>
            <h1>Fit the model plan</h1>
            <p>
                Fit each planned stage in order. This page saves the next stage's fitting settings
                and starts its fit.
            </p>
        </div>
    </div>
    {#if !plan}<section class="model-card">
            <p>Start a model plan in Project & data before fitting.</p>
        </section>
    {:else}<section class="model-card">
            <ol class="fit-stages">
                {#each statuses as item, index}<li
                        class:fitted={item.state === 'fitted'}
                        class:current={item.stage_id === next?.stage_id}
                    >
                        <div>
                            <strong>{index + 1}. {item.name}</strong><small
                                >{plan?.stages?.[index]?.kind === 'glm'
                                    ? 'GLM'
                                    : plan?.stages?.[index]?.kind === 'catboost'
                                      ? 'CatBoost'
                                      : plan?.stages?.[index]?.kind === 'catalogue'
                                        ? 'Catalogue'
                                        : 'Interaction'} · {(plan?.stages?.[index]?.factors || [])
                                    .length
                                    ? (plan?.stages?.[index]?.factors || []).join(', ')
                                    : 'No variables selected'}</small
                            >
                            <p>
                                {item.state === 'fitted'
                                    ? 'Fitted and kept as a saved version.'
                                    : item.state === 'fitting'
                                      ? 'Fit in progress.'
                                      : item.stage_id === next?.stage_id
                                        ? index
                                            ? `Will use the complete score through ${predecessor?.name} as its offset.`
                                            : 'Uses the project data and outcome.'
                                        : 'Waiting for the earlier stage.'}
                            </p>
                        </div>
                        <span>{item.message}</span>
                    </li>{/each}
            </ol>
            {#if error}<div class="message error" role="alert">{error}</div>{/if}
            {#if fittingStage}<div class="message">
                    <strong>Fitting {fittingStage.name}</strong> · {fittingStage.message ||
                        'The fit is running.'}<button onclick={cancelFit} disabled={busy}
                        >Cancel fit</button
                    >
                </div>{/if}
            {#if next?.last_attempt}<div class="message error">
                    <strong>Previous attempt: {next.last_attempt.status}</strong> · {next
                        .last_attempt.message || 'Fit did not complete. You can retry this stage.'}
                </div>{/if}
            {#if next && stage}<section class="fit-settings">
                    <h2>Settings for {next.name}</h2>
                    {#if stage.kind === 'glm'}<fieldset class="mode-choice">
                            <legend>Penalty selection</legend><label
                                class:chosen={(workingSettings.penalty_method ||
                                    (workingSettings.cv ? 'cv' : 'alpha')) === 'alpha'}
                                ><input
                                    type="radio"
                                    name="glm-penalty"
                                    checked={(workingSettings.penalty_method ||
                                        (workingSettings.cv ? 'cv' : 'alpha')) === 'alpha'}
                                    onchange={() =>
                                        setSettings({
                                            penalty_method: 'alpha',
                                            cv: null,
                                            alpha: workingSettings.alpha ?? 0.001,
                                        })}
                                /> Set alpha</label
                            ><label
                                class:chosen={(workingSettings.penalty_method ||
                                    (workingSettings.cv ? 'cv' : 'alpha')) === 'cv'}
                                ><input
                                    type="radio"
                                    name="glm-penalty"
                                    checked={(workingSettings.penalty_method ||
                                        (workingSettings.cv ? 'cv' : 'alpha')) === 'cv'}
                                    onchange={() =>
                                        setSettings({
                                            penalty_method: 'cv',
                                            alpha: null,
                                            cv: workingSettings.cv ?? 5,
                                        })}
                                /> Cross-validate alpha</label
                            >
                        </fieldset>
                        {#if (workingSettings.penalty_method || (workingSettings.cv ? 'cv' : 'alpha')) === 'cv'}<label
                                >CV folds<input
                                    aria-label="CV folds"
                                    type="number"
                                    min="2"
                                    value={workingSettings.cv ?? 5}
                                    onchange={(event) =>
                                        setSettings({ cv: Number(event.currentTarget.value) })}
                                /></label
                            >{:else}<label
                                >Penalty alpha<input
                                    aria-label="Penalty alpha"
                                    type="number"
                                    min="0"
                                    step="0.001"
                                    value={workingSettings.alpha ?? 0.001}
                                    onchange={(event) =>
                                        setSettings({ alpha: Number(event.currentTarget.value) })}
                                /></label
                            >{/if}<label
                            >L1 ratio<input
                                aria-label="L1 ratio"
                                type="number"
                                min="0"
                                max="1"
                                step="0.05"
                                value={workingSettings.l1_ratio ?? 1}
                                onchange={(event) =>
                                    setSettings({ l1_ratio: Number(event.currentTarget.value) })}
                            /></label
                        ><label
                            >Alphas on path<input
                                aria-label="Alphas on path"
                                type="number"
                                min="2"
                                value={workingSettings.n_alphas ?? 20}
                                onchange={(event) =>
                                    setSettings({ n_alphas: Number(event.currentTarget.value) })}
                            /></label
                        >
                    {:else if stage.kind === 'interaction'}<p>
                            Interaction tuning uses training rows only and searches settings for
                            this pair table.
                        </p>
                        <fieldset class="mode-choice">
                            <legend>Interaction fitting</legend><label
                                class:chosen={workingSettings.tune !== false}
                                ><input
                                    type="radio"
                                    name="interaction-method"
                                    checked={workingSettings.tune !== false}
                                    onchange={() =>
                                        setSettings({
                                            tune: true,
                                            tuning_budget:
                                                workingSettings.tuning_budget || 'trials',
                                            trials: workingSettings.trials ?? 8,
                                            cv: workingSettings.cv ?? 5,
                                        })}
                                /><span>Automatic tuning</span></label
                            ><label class:chosen={workingSettings.tune === false}
                                ><input
                                    type="radio"
                                    name="interaction-method"
                                    checked={workingSettings.tune === false}
                                    onchange={() => setSettings({ tune: false })}
                                /><span>Manual candidate</span></label
                            >
                        </fieldset>
                        {#if workingSettings.tune !== false}<label
                                >Budget<select
                                    aria-label="Interaction tuning budget"
                                    value={workingSettings.tuning_budget || 'trials'}
                                    onchange={(event) =>
                                        setSettings({ tuning_budget: event.currentTarget.value })}
                                    ><option value="trials">Number of trials</option><option
                                        value="time">Tuning time (minutes)</option
                                    ></select
                                ></label
                            >{#if (workingSettings.tuning_budget || 'trials') === 'time'}<label
                                    >Tuning time (minutes)<input
                                        aria-label="Interaction tuning time"
                                        type="number"
                                        min="1"
                                        value={workingSettings.time_limit_minutes ?? 15}
                                        onchange={(event) =>
                                            setSettings({
                                                time_limit_minutes: Number(
                                                    event.currentTarget.value,
                                                ),
                                            })}
                                    /></label
                                >{:else}<label
                                    >Trials<input
                                        aria-label="Interaction trials"
                                        type="number"
                                        min="1"
                                        value={workingSettings.trials ?? 8}
                                        onchange={(event) =>
                                            setSettings({
                                                trials: Number(event.currentTarget.value),
                                            })}
                                    /></label
                                >{/if}<label
                                >CV folds<input
                                    aria-label="Interaction CV folds"
                                    type="number"
                                    min="2"
                                    value={workingSettings.cv ?? 5}
                                    onchange={(event) =>
                                        setSettings({ cv: Number(event.currentTarget.value) })}
                                /></label
                            >{:else}<label
                                >Iterations<input
                                    aria-label="Interaction iterations"
                                    type="number"
                                    min="1"
                                    value={workingSettings.candidate?.iterations ?? 120}
                                    onchange={(event) =>
                                        setSettings({
                                            candidate: {
                                                ...(workingSettings.candidate || {}),
                                                iterations: Number(event.currentTarget.value),
                                            },
                                        })}
                                /></label
                            ><label
                                >Depth<input
                                    aria-label="Interaction depth"
                                    type="number"
                                    min="1"
                                    value={workingSettings.candidate?.depth ?? 3}
                                    onchange={(event) =>
                                        setSettings({
                                            candidate: {
                                                ...(workingSettings.candidate || {}),
                                                depth: Number(event.currentTarget.value),
                                            },
                                        })}
                                /></label
                            ><label
                                >Learning rate<input
                                    aria-label="Interaction learning rate"
                                    type="number"
                                    min="0.001"
                                    step="0.01"
                                    value={workingSettings.candidate?.learning_rate ?? 0.05}
                                    onchange={(event) =>
                                        setSettings({
                                            candidate: {
                                                ...(workingSettings.candidate || {}),
                                                learning_rate: Number(event.currentTarget.value),
                                            },
                                        })}
                                /></label
                            ><label
                                >L2 regularisation<input
                                    aria-label="Interaction L2 regularisation"
                                    type="number"
                                    min="0"
                                    value={workingSettings.candidate?.l2_leaf_reg ?? 3}
                                    onchange={(event) =>
                                        setSettings({
                                            candidate: {
                                                ...(workingSettings.candidate || {}),
                                                l2_leaf_reg: Number(event.currentTarget.value),
                                            },
                                        })}
                                /></label
                            >{/if}<label
                            >Minimum support share<input
                                aria-label="Minimum support share"
                                type="number"
                                min="0"
                                step="0.0001"
                                value={workingSettings.min_weight_share ?? 0.0025}
                                onchange={(event) =>
                                    setSettings({
                                        min_weight_share: Number(event.currentTarget.value),
                                    })}
                            /></label
                        >
                    {:else if stage.kind === 'catalogue' && workingSettings.method === 'lookup'}<p>
                            This catalogue is a fixed lookup. It does not tune or fit CatBoost
                            settings.
                        </p>
                    {:else if stage.kind === 'catboost' || stage.kind === 'catalogue'}<p>
                            Optuna tuning uses training rows only. Holdout rows are not used. Every
                            trial is evaluated across the selected training folds.
                        </p>
                        <fieldset class="mode-choice">
                            <legend>Fitting method</legend><label
                                class:chosen={workingSettings.tune !== false}
                                ><input
                                    type="radio"
                                    name="catboost-method"
                                    checked={workingSettings.tune !== false}
                                    onchange={() =>
                                        setSettings({
                                            tune: true,
                                            tuning_budget:
                                                workingSettings.tuning_budget || 'trials',
                                            trials: workingSettings.trials ?? 8,
                                            cv: workingSettings.cv ?? 3,
                                            seed: workingSettings.seed ?? 42,
                                        })}
                                /><span>Optuna tuning</span></label
                            ><label class:chosen={workingSettings.tune === false}
                                ><input
                                    type="radio"
                                    name="catboost-method"
                                    checked={workingSettings.tune === false}
                                    onchange={() => setSettings({ tune: false })}
                                /><span>Manual settings</span></label
                            >
                        </fieldset>
                        {#if workingSettings.tune !== false}<label
                                >Budget<select
                                    aria-label="Optuna budget"
                                    value={workingSettings.tuning_budget || 'trials'}
                                    onchange={(event) =>
                                        setSettings({ tuning_budget: event.currentTarget.value })}
                                    ><option value="trials">Number of trials</option><option
                                        value="time">Tuning time (minutes)</option
                                    ></select
                                ></label
                            >{#if (workingSettings.tuning_budget || 'trials') === 'time'}<label
                                    >Tuning time (minutes)<input
                                        aria-label="Tuning time minutes"
                                        type="number"
                                        min="1"
                                        value={workingSettings.time_limit_minutes ?? 15}
                                        onchange={(event) =>
                                            setSettings({
                                                time_limit_minutes: Number(
                                                    event.currentTarget.value,
                                                ),
                                            })}
                                    /></label
                                >{:else}<label
                                    >Optuna trials<input
                                        aria-label="Optuna trials"
                                        type="number"
                                        min="1"
                                        value={workingSettings.trials ?? 8}
                                        onchange={(event) =>
                                            setSettings({
                                                trials: Number(event.currentTarget.value),
                                            })}
                                    /></label
                                >{/if}<label
                                >CV folds<input
                                    aria-label="CV folds"
                                    type="number"
                                    min="2"
                                    value={workingSettings.cv ?? 3}
                                    onchange={(event) =>
                                        setSettings({ cv: Number(event.currentTarget.value) })}
                                /></label
                            ><label
                                >Seed<input
                                    aria-label="Seed"
                                    type="number"
                                    value={workingSettings.seed ?? 42}
                                    onchange={(event) =>
                                        setSettings({ seed: Number(event.currentTarget.value) })}
                                /></label
                            ><label
                                >Threads<input
                                    aria-label="Threads"
                                    type="number"
                                    min="1"
                                    value={workingSettings.thread_count ?? 4}
                                    onchange={(event) =>
                                        setSettings({
                                            thread_count: Number(event.currentTarget.value),
                                        })}
                                /></label
                            >{:else}<label
                                >Iterations<input
                                    aria-label="Iterations"
                                    type="number"
                                    min="1"
                                    value={workingSettings.iterations ?? 500}
                                    onchange={(event) =>
                                        setSettings({
                                            iterations: Number(event.currentTarget.value),
                                        })}
                                /></label
                            ><label
                                >Depth<input
                                    aria-label="Depth"
                                    type="number"
                                    min="1"
                                    value={workingSettings.depth ?? 6}
                                    onchange={(event) =>
                                        setSettings({ depth: Number(event.currentTarget.value) })}
                                /></label
                            ><label
                                >Learning rate<input
                                    aria-label="Learning rate"
                                    type="number"
                                    min="0.001"
                                    step="0.01"
                                    value={workingSettings.learning_rate ?? 0.05}
                                    onchange={(event) =>
                                        setSettings({
                                            learning_rate: Number(event.currentTarget.value),
                                        })}
                                /></label
                            ><label
                                >L2 regularisation<input
                                    aria-label="L2 regularisation"
                                    type="number"
                                    min="0"
                                    value={workingSettings.l2_leaf_reg ?? 3}
                                    onchange={(event) =>
                                        setSettings({
                                            l2_leaf_reg: Number(event.currentTarget.value),
                                        })}
                                /></label
                            ><label
                                >Seed<input
                                    aria-label="Seed"
                                    type="number"
                                    value={workingSettings.seed ?? 42}
                                    onchange={(event) =>
                                        setSettings({ seed: Number(event.currentTarget.value) })}
                                /></label
                            ><label
                                >Threads<input
                                    aria-label="Threads"
                                    type="number"
                                    min="1"
                                    value={workingSettings.thread_count ?? 4}
                                    onchange={(event) =>
                                        setSettings({
                                            thread_count: Number(event.currentTarget.value),
                                        })}
                                /></label
                            >{/if}{/if}
                    <div class="fit-next">
                        <div><p>Saving these settings starts this fit.</p></div>
                        <button class="primary" onclick={fit} disabled={busy || polling}
                            >{busy
                                ? 'Starting fit…'
                                : polling
                                  ? 'Fitting…'
                                  : `Fit ${next.name}`}</button
                        >
                    </div>
                </section>{:else if fittingStage}<div class="message">
                    Fitting is in progress. This page will refresh when it completes.
                </div>{:else}<div class="message">
                    All stages are fitted. Use Diagnostics to inspect each saved prefix and the full
                    model.
                </div>{/if}
        </section>{/if}
</section>

<style>
    .mode-choice {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        flex-basis: 100%;
        border: 0;
        padding: 0;
        margin: 0;
    }
    .mode-choice legend {
        width: 100%;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .mode-choice label {
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 9px 12px;
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .mode-choice label.chosen {
        background: #246d59;
        color: #fff;
        border-color: #246d59;
        font-weight: 700;
    }
    .fit-stages {
        padding: 0;
        margin: 0;
        list-style: none;
    }
    .fit-stages li {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        border-bottom: 1px solid var(--border);
        padding: 14px 0;
    }
    .fit-stages p {
        margin: 4px 0 0;
        color: var(--muted);
    }
    .fit-stages span {
        white-space: nowrap;
        color: var(--muted);
    }
    .fit-stages .fitted span {
        color: var(--green);
    }
    .fit-settings {
        display: flex;
        gap: 14px;
        align-items: end;
        flex-wrap: wrap;
        margin-top: 20px;
    }
    .fit-settings h2,
    .fit-settings p {
        flex-basis: 100%;
        margin: 0;
    }
    .fit-next {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: center;
        flex-basis: 100%;
        margin-top: 10px;
    }
    @media (max-width: 620px) {
        .fit-stages li,
        .fit-next {
            display: block;
        }
        .fit-stages span {
            display: block;
            margin-top: 8px;
        }
        .fit-next button {
            width: 100%;
            margin-top: 10px;
        }
    }
</style>
