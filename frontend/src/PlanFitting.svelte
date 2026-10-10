<script>
    import { onDestroy, onMount, tick } from 'svelte';
    import CoreInteractions from './CoreInteractions.svelte';
    import { isLogicalPlan, logicalModels, modelSourceLabel } from './modelPlan.js';
    export let api;
    export let state;
    export let planState = null;
    export let onChanged = () => {};
    export let onFitted = () => {};
    export let onNavigate = () => {};
    export let onDirty = () => {};
    let error = '',
        busy = false,
        polling = false,
        workingSettings = {},
        settingsKey = '',
        pollCancelled = false,
        pollGeneration = 0,
        settingsBaseline = '',
        reviewPanelsInitialisedFor = '';
    let openReviewIds = [],
        visitedReviewIds = [],
        interactionPanels = {};
    $: plan = planState?.plan || null;
    $: logicalPlan = isLogicalPlan(plan);
    $: interactionsSupported =
        plan?.link === 'log' ||
        (['poisson', 'tweedie', 'gaussian'].includes(plan?.family) && plan?.link == null);
    $: plannedModels = logicalModels(plan);
    $: statuses = planState?.stage_statuses || [];
    $: modelStatuses = Object.fromEntries(
        plannedModels.map((model) => [
            model.model_id,
            statuses.find(
                (item) =>
                    item.model_id === model.model_id ||
                    item.stage_id === model.model_id ||
                    item.stage_id === model.core?.component_id,
            ),
        ]),
    );
    $: reviewableGlms = plannedModels.filter(
        (model) =>
            interactionsSupported &&
            model.core?.kind === 'glm' &&
            Boolean(modelStatuses[model.model_id]?.fit),
    );
    $: reviewableIds = reviewableGlms.map((model) => model.model_id);
    $: reviewPlanId = plan?.plan_id || '';
    $: if (reviewPlanId !== reviewPanelsInitialisedFor) {
        reviewPanelsInitialisedFor = reviewPlanId;
        openReviewIds = [];
        visitedReviewIds = [];
    }
    $: if (openReviewIds.some((modelId) => !reviewableIds.includes(modelId)))
        openReviewIds = openReviewIds.filter((modelId) => reviewableIds.includes(modelId));
    $: if (visitedReviewIds.some((modelId) => !reviewableIds.includes(modelId)))
        visitedReviewIds = visitedReviewIds.filter((modelId) => reviewableIds.includes(modelId));
    $: if (reviewableGlms.length && !visitedReviewIds.length) {
        const preferred =
            reviewableGlms.find(
                (model) => modelStatuses[model.model_id]?.state === 'needs_refit',
            ) || reviewableGlms[0];
        openReviewIds = preferred ? [preferred.model_id] : [];
        visitedReviewIds = preferred ? [preferred.model_id] : [];
    }
    $: reviewOpenById = Object.fromEntries(openReviewIds.map((modelId) => [modelId, true]));
    $: reviewVisitedById = Object.fromEntries(visitedReviewIds.map((modelId) => [modelId, true]));
    function reviewModelForStatus(status) {
        return plannedModels.find(
            (model) =>
                model.model_id === status?.stage_id ||
                model.core?.component_id === status?.stage_id,
        );
    }
    function reviewModelIndex(model) {
        return plannedModels.findIndex((item) => item.model_id === model?.model_id);
    }
    function displayStageState(item) {
        return item.state === 'ready' && !item.fit && item.last_attempt?.status === 'complete'
            ? 'needs_refit'
            : item.state;
    }
    function displayStageMessage(item) {
        return displayStageState(item) === 'needs_refit' && item.state !== 'needs_refit'
            ? 'Needs refit'
            : item.message;
    }
    async function toggleInteractions(model) {
        const modelId = model.model_id;
        if (reviewOpenById[modelId]) {
            openReviewIds = openReviewIds.filter((item) => item !== modelId);
            return;
        }
        openReviewIds = [...openReviewIds, modelId];
        if (!reviewVisitedById[modelId]) visitedReviewIds = [...visitedReviewIds, modelId];
        await tick();
        document
            .getElementById(`glm-interaction-review-${modelId}`)
            ?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    function manageInteractions(model) {
        toggleInteractions(model);
    }
    $: fittingStage = statuses.find((item) => item.state === 'fitting') || null;
    $: next = statuses.find((item) => item.stage_id === planState?.next_stage_id) || null;
    $: nextIndex = next ? statuses.indexOf(next) : -1;
    $: stage = logicalPlan
        ? plannedModels.find((model) => model.model_id === next?.stage_id)?.core || null
        : plan?.stages?.find((item) => item.component_id === next?.stage_id);
    $: predecessor = nextIndex > 0 ? statuses[nextIndex - 1] : null;
    $: stageKey = JSON.stringify({
        plan_id: plan?.plan_id,
        id: stage?.component_id,
        settings: stage?.settings || {},
    });
    $: if (stage && stageKey !== settingsKey && !busy) {
        workingSettings = structuredClone(stage.settings || {});
        settingsBaseline = JSON.stringify(workingSettings);
        settingsKey = stageKey;
    }
    $: settingsDirty = Boolean(stage && JSON.stringify(workingSettings) !== settingsBaseline);
    $: onDirty(settingsDirty);
    function setupProblem(model) {
        if (model?.kind === 'catalogue') {
            return !model.settings?.source_path || !model.settings?.key
                ? 'Upload the catalogue and choose its join key in Variables.'
                : '';
        }
        if (model?.kind === 'interaction' && model.factors?.length !== 2)
            return 'Choose two variables for this interaction in Variables.';
        return model && !model.factors?.length
            ? 'Choose variables for this model before fitting.'
            : '';
    }
    $: nextProblem = setupProblem(stage);
    $: fittedCount = statuses.filter((item) => item.state === 'fitted').length;
    $: completeOffset =
        nextIndex > 0
            ? statuses
                  .slice(0, nextIndex)
                  .map((item) => item.name)
                  .join(' → ')
            : '';
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
        onDirty(false);
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
    async function saveSettings() {
        const settings = settingsForSave();
        const nextPlan = logicalPlan
            ? {
                  ...plan,
                  models: (plan.models || []).map((model) =>
                      model.core?.component_id === stage.component_id
                          ? { ...model, core: { ...model.core, settings } }
                          : model,
                  ),
              }
            : {
                  ...plan,
                  stages: plan.stages.map((item) =>
                      item.component_id === stage.component_id ? { ...item, settings } : item,
                  ),
              };
        const saved = await api('model-plan', {
            session_id: state.session_id,
            revision: state.revision,
            operation: 'save',
            plan: nextPlan,
        });
        const savedStage = saved.plan.stages?.find(
            (item) => item.component_id === stage.component_id,
        );
        workingSettings = structuredClone(savedStage?.settings || settings);
        settingsBaseline = JSON.stringify(workingSettings);
        onChanged(saved);
        return saved;
    }
    async function saveWithoutFit() {
        busy = true;
        error = '';
        try {
            await saveSettings();
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    async function fit() {
        if (!next || !stage || polling || nextProblem) return;
        busy = true;
        error = '';
        try {
            const saved = await saveSettings();
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
    async function fitRemainingInteractions(reviewStatus) {
        if (!reviewStatus || reviewStatus.state !== 'needs_refit' || polling) return;
        busy = true;
        error = '';
        try {
            const started = await api('model-plan/fit-next', {
                session_id: state.session_id,
                revision: state.revision,
            });
            onChanged(started.snapshot || (await api('model-plan')));
            waitForFit(
                reviewStatus.stage_id,
                started.model_name || reviewStatus.fit?.model_name || '',
            );
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
            <h1>Fit models</h1>
            <p>
                {plan?.mode === 'single'
                    ? 'Fit the model you defined in Project & data.'
                    : 'Fit the models in order. Each later model uses all preceding fitted predictions as its offset.'}
            </p>
        </div>
        <button onclick={() => onNavigate('models')}>Edit model setup</button>
    </div>
    {#if !plan}<section class="model-card">
            <p>Start a model plan in Project & data before fitting.</p>
        </section>
    {:else}<section class="model-card">
            <ol class="fit-stages" aria-label="Model fit order">
                {#each statuses as item, index (item.stage_id)}
                    {@const stageModel = logicalPlan ? reviewModelForStatus(item) : null}
                    {@const stageModelIndex = reviewModelIndex(stageModel)}
                    {@const stageHasReview = Boolean(
                        interactionsSupported && stageModel?.core?.kind === 'glm' && item.fit,
                    )}
                    {@const visibleStageState = displayStageState(item)}
                    <li
                        class:fitted={item.state === 'fitted'}
                        class:needs-refit={visibleStageState === 'needs_refit'}
                        class:current={item.stage_id === next?.stage_id}
                    >
                        <div class="fit-stage-row">
                            <div>
                                <strong>{index + 1}. {item.name}</strong><small
                                    >{plan.adopted_model_name && index === 0
                                        ? 'Saved fitted model · accepted variables preserved'
                                        : plan?.stages?.[index]?.kind === 'glm'
                                          ? 'GLM'
                                          : plan?.stages?.[index]?.kind === 'catboost'
                                            ? 'CatBoost'
                                            : plan?.stages?.[index]?.kind === 'catalogue'
                                              ? 'Catalogue'
                                              : 'Interaction'}{#if !(plan.adopted_model_name && index === 0)}
                                        · {(plan?.stages?.[index]?.factors || []).length
                                            ? (plan?.stages?.[index]?.factors || []).join(', ')
                                            : 'No variables selected'}{/if}</small
                                >
                                <p>
                                    {item.state === 'fitted'
                                        ? 'Fitted and kept as a saved version.'
                                        : item.state === 'fitting'
                                          ? 'Fit in progress.'
                                          : visibleStageState === 'needs_refit'
                                            ? 'The saved fit no longer matches this model’s settings or preceding score.'
                                            : item.stage_id === next?.stage_id
                                              ? index
                                                  ? logicalPlan
                                                      ? `${modelSourceLabel(plan, index)} This is the offset for this model.`
                                                      : `Will use the complete score through ${predecessor?.name} as its offset.`
                                                  : 'Uses the project data and outcome.'
                                              : 'Waiting for the earlier stage.'}
                                </p>
                            </div>
                            <div class="stage-actions">
                                <span
                                    >{item.state === 'ready' && setupProblem(plan?.stages?.[index])
                                        ? 'Variables needed'
                                        : displayStageMessage(item)}</span
                                >
                                {#if item.state === 'fitted'}
                                    <button
                                        onclick={() => {
                                            onFitted(item.fit?.model_name || item.name);
                                            onNavigate('diagnostics');
                                        }}>Inspect {item.name}</button
                                    >
                                    {#if stageHasReview}
                                        <button onclick={() => manageInteractions(stageModel)}
                                            >{reviewOpenById[stageModel.model_id]
                                                ? 'Hide interactions'
                                                : 'Manage interactions'}</button
                                        >
                                    {/if}
                                {:else if item.state === 'needs_refit'}
                                    {#if stageHasReview}
                                        <button
                                            onclick={() => {
                                                onFitted(item.fit?.model_name || item.name);
                                                onNavigate('diagnostics');
                                            }}>Inspect {item.name}</button
                                        >
                                        <button onclick={() => manageInteractions(stageModel)}
                                            >{reviewOpenById[stageModel.model_id]
                                                ? 'Hide interactions'
                                                : 'Manage interactions'}</button
                                        >
                                        <button
                                            class="primary"
                                            onclick={() => fitRemainingInteractions(item)}
                                            disabled={busy || polling}
                                            >Fit remaining interaction tables</button
                                        >
                                    {/if}
                                {:else if item.state !== 'fitting'}
                                    <button onclick={() => onNavigate('variables', item.stage_id)}
                                        >Edit variables</button
                                    >
                                {/if}
                            </div>
                        </div>
                        {#if stageHasReview && reviewVisitedById[stageModel.model_id]}
                            <section
                                id={`glm-interaction-review-${stageModel.model_id}`}
                                class="glm-review-inline"
                                aria-label={`Review ${stageModel.name}`}
                                hidden={!reviewOpenById[stageModel.model_id]}
                            >
                                <div class="review-heading">
                                    <div>
                                        <h2>Review {stageModel.name}</h2>
                                        <p>
                                            {stageModelIndex > 0
                                                ? modelSourceLabel(plan, stageModelIndex)
                                                : 'Starts from project data.'} Inspect this model’s accepted
                                            score, then add an interaction only when it is useful.
                                        </p>
                                    </div>
                                </div>
                                <div class="review-actions">
                                    <button
                                        onclick={() => {
                                            onFitted(item.fit?.model_name || stageModel.name);
                                            onNavigate('diagnostics');
                                        }}>Check A/E</button
                                    >
                                    <button
                                        onclick={() =>
                                            interactionPanels[stageModel.model_id]?.runSearch(
                                                'factors',
                                            )}>Find missing variables</button
                                    >
                                    <button
                                        onclick={() =>
                                            interactionPanels[stageModel.model_id]?.runSearch(
                                                'interactions',
                                            )}>Find interactions</button
                                    >
                                </div>
                                <CoreInteractions
                                    bind:this={interactionPanels[stageModel.model_id]}
                                    {api}
                                    {state}
                                    {planState}
                                    model={stageModel}
                                    {onChanged}
                                />
                                {#if stageModelIndex < plannedModels.length - 1}
                                    <button
                                        class="primary continue-model"
                                        onclick={() =>
                                            onNavigate(
                                                'variables',
                                                plannedModels[stageModelIndex + 1].core
                                                    .component_id,
                                            )}
                                        >Continue to {plannedModels[stageModelIndex + 1]
                                            .name}</button
                                    >
                                {/if}
                            </section>
                        {/if}
                    </li>
                {/each}
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
                    <h2>Fit Model {nextIndex + 1} — {next.name}</h2>
                    <p>
                        {completeOffset
                            ? `Offset: complete predictions from ${completeOffset}.`
                            : 'Starts from the project data and shared response.'}
                    </p>
                    {#if nextProblem}<div class="message setup-needed" role="status">
                            <p>{nextProblem}</p>
                            <button onclick={() => onNavigate('variables', stage.component_id)}
                                >Choose variables for {next.name}</button
                            >
                        </div>{/if}
                    <p class="settings-state" role="status">
                        {settingsDirty
                            ? 'Unsaved fitting settings. Save them below, or fit to save and run.'
                            : 'Fitting settings saved.'}
                    </p>
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
                        <button
                            onclick={saveWithoutFit}
                            disabled={busy || polling || !settingsDirty}
                            >Save fitting settings</button
                        >
                        <button
                            class="primary"
                            onclick={fit}
                            disabled={busy || polling || !!nextProblem}
                            >{busy
                                ? 'Starting fit…'
                                : polling
                                  ? 'Fitting…'
                                  : `Fit Model ${nextIndex + 1} — ${next.name}`}</button
                        >
                    </div>
                </section>{:else if fittingStage}<div class="message">
                    Fitting is in progress. This page will refresh when it completes.
                </div>{:else}<div class="message">
                    All models are fitted.
                    <button
                        class="primary"
                        onclick={() => {
                            onFitted(statuses.at(-1)?.fit?.model_name || '');
                            onNavigate('diagnostics');
                        }}>Inspect complete model</button
                    >
                </div>{/if}
            {#if fittedCount && next}<div class="fit-progress" role="status">
                    <span>{fittedCount} of {statuses.length} models fitted.</span>
                    <button
                        onclick={() => {
                            onFitted(statuses[fittedCount - 1]?.fit?.model_name || '');
                            onNavigate('diagnostics');
                        }}>Inspect fitted models</button
                    >
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
        border-bottom: 1px solid var(--border);
        padding: 14px 0;
    }
    .fit-stage-row {
        display: flex;
        justify-content: space-between;
        gap: 16px;
    }
    .fit-stages p {
        margin: 4px 0 0;
        color: var(--muted);
    }
    .fit-stages small {
        display: block;
        margin-top: 6px;
        color: var(--muted);
    }
    .fit-stages span {
        white-space: nowrap;
        color: var(--muted);
    }
    .stage-actions {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
    }
    .setup-needed,
    .settings-state {
        flex-basis: 100%;
    }
    .fit-progress {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-top: 20px;
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
        .fit-stage-row,
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
    .glm-review-inline {
        border-top: 1px solid var(--border);
        margin-top: 18px;
        padding-top: 18px;
    }
    .review-heading {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
    }
    .review-heading h2 {
        margin: 0;
    }
    .review-heading p {
        margin: 8px 0 0;
    }
    .review-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 18px 0;
    }
    .continue-model {
        margin-top: 16px;
    }
    @media (max-width: 900px) {
        .review-heading {
            flex-direction: column;
        }
    }
</style>
