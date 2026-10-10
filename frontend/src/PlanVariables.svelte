<script>
    import { tick } from 'svelte';
    import ModelVariables from './ModelVariables.svelte';
    export let api;
    export let uploadCatalogue;
    export let state;
    export let planState = null;
    export let selectedStageId = '';
    export let onChanged = () => {};
    export let onDirty = () => {};
    export let revisionIntent = false;
    export let onSharedSetup = () => {};
    export let embedded = false;
    export let singleStep = 'selection';
    export let onSingleStepChange = () => {};
    export let onSaved = () => {};
    export let onCreateRevision = () => {};
    export let onStageChange = () => {};
    export let resetSignal = 0;
    let resetSeen = 0;
    let working = null;
    let workingPlan = null;
    let baseline = '';
    let jsonDraft = '';
    let validity = '';
    let error = '';
    let busy = false;
    let catalogue = null;
    $: plan = planState?.plan || null;
    $: stages = plan?.stages || [];
    $: singleMode = plan?.mode === 'single';
    $: if (!selectedStageId && stages.length) selectedStageId = stages[0].component_id;
    $: stageIndex = stages.findIndex((stage) => stage.component_id === selectedStageId);
    $: current = stageIndex >= 0 ? stages[stageIndex] : null;
    $: adoptedStarter = Boolean(plan?.adopted_model_name) && stageIndex === 0;
    $: sourceKey = JSON.stringify({
        current: current || null,
        response: plan
            ? {
                  family: plan.family,
                  link: plan.link,
                  tweedie_power: plan.tweedie_power,
                  divide_target_by_weight: plan.divide_target_by_weight,
              }
            : null,
    });
    $: if (current && sourceKey !== baseline && !busy) {
        working = structuredClone(current);
        workingPlan = structuredClone(plan);
        baseline = sourceKey;
        singleStep = 'selection';
        catalogue =
            planState?.catalogue_descriptors?.[current.component_id] ||
            (current.settings?.source_path
                ? {
                      filename: current.settings.source_path.split('/').at(-1),
                      rows: 'saved',
                      columns: [],
                  }
                : null);
        jsonDraft = '';
        validity = '';
    }
    $: dirty = Boolean(
        working &&
        (JSON.stringify({
            current: working,
            response: workingPlan
                ? {
                      family: workingPlan.family,
                      link: workingPlan.link,
                      tweedie_power: workingPlan.tweedie_power,
                      divide_target_by_weight: workingPlan.divide_target_by_weight,
                  }
                : null,
        }) !== baseline ||
            jsonDraft),
    );
    $: onDirty(dirty);
    $: isFitted = Boolean(
        planState?.stage_statuses?.find((item) => item.stage_id === current?.component_id)
            ?.state === 'fitted',
    );
    $: locked = adoptedStarter || (isFitted && !revisionIntent);
    $: sourceColumns =
        planState?.safe_column_details?.[stageIndex] ||
        (planState?.safe_columns?.[stageIndex] || []).map(
            (name) => state?.columns?.find((entry) => entry.name === name) || { name, dtype: '' },
        );
    function methodLabel(kind) {
        return (
            {
                glm: 'GLM',
                catboost: 'CatBoost',
                interaction: 'Interaction table',
                catalogue: 'Catalogue',
            }[kind] || kind
        );
    }
    function typedColumn(column) {
        const name = column.name || column;
        return {
            ...column,
            name,
            numeric: /int|float|decimal/.test(
                String(column.dtype || column.type || '').toLowerCase(),
            ),
        };
    }
    $: columns =
        current?.kind === 'catalogue' && catalogue
            ? (catalogue.attribute_columns?.length
                  ? catalogue.attribute_columns
                  : catalogue.columns?.length
                    ? catalogue.columns
                    : (working?.factors || []).map((name) => ({ name, dtype: 'string' }))
              )
                  .filter((column) => (column.name || column) !== working?.settings?.key)
                  .map(typedColumn)
            : sourceColumns.map(typedColumn);
    function setSingleStep(next) {
        singleStep = next;
        onSingleStepChange(next);
    }
    function change(next) {
        working = { ...working, ...next };
    }
    function changePlan(next) {
        workingPlan = { ...workingPlan, ...next };
    }
    function changeMethod(kind) {
        const catBoostSettings = {
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
        };
        const settings =
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
                  : kind === 'catalogue'
                    ? {
                          ...catBoostSettings,
                          ...(working.kind === 'catalogue' ? working.settings : {}),
                          source_path:
                              working.kind === 'catalogue'
                                  ? working.settings?.source_path || ''
                                  : '',
                          source_sha256:
                              working.kind === 'catalogue'
                                  ? working.settings?.source_sha256 || ''
                                  : '',
                          key: working.kind === 'catalogue' ? working.settings?.key || '' : '',
                      }
                    : catBoostSettings;
        working = {
            ...working,
            kind,
            settings,
            ...(kind === 'catalogue' ? { factors: [] } : {}),
        };
    }
    async function upload(event) {
        const file = event.currentTarget.files?.[0];
        if (!file) return;
        busy = true;
        error = '';
        try {
            catalogue = await uploadCatalogue(file, {
                session_id: state.session_id,
                revision: state.revision,
            });
            working = {
                ...working,
                factors: [],
                settings: {
                    ...working.settings,
                    source_path: catalogue.source_path,
                    source_sha256: catalogue.source_sha256,
                    key: '',
                },
            };
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function setCatalogueKey(key) {
        working = { ...working, factors: [], settings: { ...working.settings, key } };
    }
    async function save() {
        if (!plan || !working || validity || locked) return;
        busy = true;
        error = '';
        const stagesNext = stages.map((item) =>
            item.component_id === working.component_id ? working : item,
        );
        try {
            const result = await api('model-plan', {
                session_id: state.session_id,
                revision: state.revision,
                operation: isFitted ? 'revise' : 'save',
                plan: {
                    ...(singleMode ? workingPlan : plan),
                    stages: stagesNext,
                },
            });
            const savedPlan = result.plan || workingPlan || plan;
            const savedStage =
                savedPlan.stages?.find((item) => item.component_id === working.component_id) ||
                working;
            working = structuredClone(savedStage);
            workingPlan = structuredClone(savedPlan);
            baseline = JSON.stringify({
                current: savedStage,
                response: {
                    family: savedPlan.family,
                    link: savedPlan.link,
                    tweedie_power: savedPlan.tweedie_power,
                    divide_target_by_weight: savedPlan.divide_target_by_weight,
                },
            });
            jsonDraft = '';
            onChanged(result);
            // Let the parent receive the cleared dirty state before it navigates to Fit.
            await tick();
            onDirty(false);
            await tick();
            onSaved(result, { stageIndex, stageId: working.component_id });
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function discard() {
        working = current ? structuredClone(current) : null;
        workingPlan = plan ? structuredClone(plan) : null;
        baseline = sourceKey;
        jsonDraft = '';
        validity = '';
        error = '';
    }
    $: if (resetSignal !== resetSeen) {
        resetSeen = resetSignal;
        discard();
    }
</script>

<section
    class="plan-variables"
    class:workflow-page={!embedded}
    class:embedded
    id={embedded ? 'stage-variables' : undefined}
    aria-label={singleMode ? 'Variables for this model' : 'Stage variables'}
>
    {#if embedded}
        <div class="embedded-heading">
            <div>
                <h2>
                    {singleMode
                        ? 'Variables for this model'
                        : adoptedStarter
                          ? `Saved variables for ${plan.adopted_model_name}`
                          : `Variables for ${current?.name || 'this stage'}`}
                </h2>
                <p class="model-variable-context">
                    {#if adoptedStarter}
                        Model {stageIndex + 1} of {stages.length} · Saved fitted model · {plan.adopted_model_name}
                    {:else}
                        Model {stageIndex + 1} of {stages.length} · {working?.name || current?.name || 'Unnamed model'} · {methodLabel(working?.kind || current?.kind)}
                    {/if}
                </p>
                <p>
                    {singleMode
                        ? 'Choose the factors and their treatment for this model.'
                        : adoptedStarter
                          ? 'The complete saved scorer and its variable definitions are preserved.'
                          : stageIndex
                            ? `This stage uses the complete score through ${stages[stageIndex - 1]?.name} as its offset.`
                            : 'This first stage starts from the shared data.'}
                </p>
            </div>
            {#if !adoptedStarter}<strong>{working?.factors?.length || 0} selected</strong>{/if}
        </div>
    {:else}
        <div class="heading">
            <div>
                <div class="eyebrow">MODEL PLAN</div>
                <h1>Set up a stage</h1>
                <p>
                    Choose variables for this stage. Data roles, the target and the split are shared
                    and cannot be changed here.
                </p>
            </div>
        </div>
    {/if}
    {#if !plan}
        <section class="model-card">
            <p>Start a model plan in Project & data before choosing stage variables.</p>
        </section>
    {:else}
        {#if !singleMode}
            <section
                class:embedded-picker={embedded}
                class:model-card={!embedded}
                class="stage-picker"
            >
                <label>
                    Model
                    <select
                        aria-label="Model stage"
                        value={selectedStageId}
                        onchange={(event) => {
                            if (dirty && !window.confirm('Discard unsaved stage setup changes?')) {
                                event.currentTarget.value = selectedStageId;
                                return;
                            }
                            selectedStageId = event.currentTarget.value;
                            onStageChange(selectedStageId);
                        }}
                    >
                        {#each stages as item, index}
                            <option value={item.component_id}
                                >Model {index + 1} of {stages.length} · {index === 0 && plan.adopted_model_name
                                    ? plan.adopted_model_name
                                    : item.name}</option
                            >
                        {/each}
                    </select>
                </label>
                {#if current && stageIndex > 0}
                    <p class="help-text">
                        This stage automatically offsets the complete score through {stages[
                            stageIndex - 1
                        ].name}.
                    </p>
                {/if}
            </section>
        {/if}
        {#if !embedded}
            <button class="quiet" onclick={onSharedSetup}>Change shared data setup</button>
        {/if}
        {#if error}<div class="message error" role="alert">{error}</div>{/if}
        {#if adoptedStarter}
            <div class="message">
                This saved starting model is locked. Add a model after it to extend the chain. Start
                a new plan to replace it.
            </div>
        {:else if locked}
            <div class="message">
                This stage is already fitted. Create a revised plan to change it while retaining
                the fitted version.
                <button onclick={onCreateRevision}>Create revised plan</button>
            </div>
        {/if}
        {#if working && working.kind === 'catalogue'}
            <section class="model-card catalogue-setup">
                <h2>Full catalogue</h2>
                <p>
                    Upload the complete CSV or Parquet file, including keys not present in this
                    sample.
                </p>
                <input
                    type="file"
                    accept=".csv,.parquet"
                    aria-label="Upload catalogue"
                    onchange={upload}
                    disabled={busy || locked}
                />
                {#if catalogue}
                    <p>{catalogue.filename} · {catalogue.rows} rows</p>
                    <label>
                        Unique key
                        <select
                            aria-label="Catalogue key"
                            value={working.settings?.key || ''}
                            onchange={(event) => setCatalogueKey(event.currentTarget.value)}
                        >
                            <option value="">Choose key</option>
                            {#each catalogue.key_columns?.length ? catalogue.key_columns : catalogue.columns || [] as column}
                                <option value={column.name || column}
                                    >{column.name || column}</option
                                >
                            {/each}
                        </select>
                    </label>
                    <p>Select attributes below. The key is not a model variable.</p>
                {/if}
            </section>
        {/if}
        {#if working && !adoptedStarter}
            <fieldset disabled={locked}>
                {#if singleMode && embedded && singleStep === 'selection'}
                    <ModelVariables
                        {columns}
                        factors={working.factors || []}
                        design={working.design}
                        inheritedDesign={{ defaults: planState?.design_defaults, variables: {} }}
                        method={working.kind === 'catalogue' ? 'catboost' : working.kind}
                        selectionOnly={true}
                        showHeading={false}
                        onChange={change}
                        onValidity={(value) => (validity = value)}
                        onDraftText={(value) => (jsonDraft = value)}
                    />
                    <div class="step-actions">
                        <button class="primary" onclick={() => setSingleStep('details')}
                            >Continue to model details</button
                        >
                    </div>
                {:else if singleMode && embedded && singleStep === 'details'}
                    <section class="single-details" aria-label="Model details">
                        <h2>Model details</h2>
                        <p>Choose how the selected factors will be fitted.</p>
                        <div class="two-column-grid">
                            <label
                                ><span>Model name</span><input
                                    aria-label="Model name"
                                    bind:value={working.name}
                                /></label
                            >
                            <label
                                ><span>Fitting method</span><select
                                    aria-label="Fitting method"
                                    value={working.kind}
                                    onchange={(event) => changeMethod(event.currentTarget.value)}
                                    ><option value="glm">GLM</option><option value="catboost"
                                        >CatBoost</option
                                    ><option value="interaction">Interaction</option><option
                                        value="catalogue">Catalogue</option
                                    ></select
                                ></label
                            >
                            <label
                                ><span>Response family</span><select
                                    aria-label="Model family"
                                    value={workingPlan.family}
                                    onchange={(event) =>
                                        changePlan({
                                            family: event.currentTarget.value,
                                            link:
                                                event.currentTarget.value === 'binomial'
                                                    ? 'logit'
                                                    : null,
                                        })}
                                    ><option value="poisson">Poisson</option><option value="tweedie"
                                        >Tweedie</option
                                    ><option value="gaussian">Gaussian</option><option value="gamma"
                                        >Gamma</option
                                    ><option value="inverse_gaussian">Inverse Gaussian</option
                                    ><option value="binomial">Binomial</option></select
                                ></label
                            >
                            <label
                                ><span>Response link</span><select
                                    aria-label="Model link"
                                    value={workingPlan.link || ''}
                                    onchange={(event) =>
                                        changePlan({ link: event.currentTarget.value || null })}
                                    ><option value="">Family default</option><option value="log"
                                        >Log</option
                                    ><option value="logit">Logit</option></select
                                ></label
                            >
                        </div>
                        {#if workingPlan.family === 'tweedie'}<label class="single-field"
                                ><span>Tweedie power</span><input
                                    aria-label="Tweedie power"
                                    type="number"
                                    min="1.01"
                                    max="1.99"
                                    step="0.01"
                                    value={workingPlan.tweedie_power}
                                    onchange={(event) =>
                                        changePlan({
                                            tweedie_power: Number(event.currentTarget.value),
                                        })}
                                /></label
                            >{/if}
                        <label class="checkbox-field"
                            ><input
                                type="checkbox"
                                checked={workingPlan.divide_target_by_weight}
                                onchange={(event) =>
                                    changePlan({
                                        divide_target_by_weight: event.currentTarget.checked,
                                    })}
                            /><span
                                >Divide target by weight <span class="help-text"
                                    >(use a unit response)</span
                                ></span
                            ></label
                        >
                        <div class="step-actions">
                            <button onclick={() => setSingleStep('selection')}
                                >Back to factor selection</button
                            ><button class="primary" onclick={() => setSingleStep('treatment')}
                                >Continue to variable treatment</button
                            >
                        </div>
                    </section>
                {:else}
                    {#if singleMode && embedded}<div class="step-summary">
                            <button class="text-button" onclick={() => setSingleStep('selection')}
                                >Change selected factors</button
                            ><button class="text-button" onclick={() => setSingleStep('details')}
                                >Change model details</button
                            >
                        </div>{/if}
                    <div class:single-treatment={singleMode && embedded && singleStep === 'treatment'}>
                        {#key singleMode && embedded ? singleStep : working.kind}
                            <ModelVariables
                                {columns}
                                factors={working.factors || []}
                                design={working.design}
                                inheritedDesign={{ defaults: planState?.design_defaults, variables: {} }}
                                method={working.kind === 'catalogue' ? 'catboost' : working.kind}
                                treatmentOnly={singleMode && embedded && singleStep === 'treatment'}
                                showHeading={false}
                                onChange={change}
                                onValidity={(value) => (validity = value)}
                                onDraftText={(value) => (jsonDraft = value)}
                            />
                        {/key}
                    </div>
                {/if}
            </fieldset>
            {#if !(singleMode && embedded && singleStep !== 'treatment')}<div class="plan-save">
                    <button onclick={discard} disabled={!dirty || busy}>Discard changes</button>
                    <button class="primary" onclick={save} disabled={!dirty || busy || !!validity}
                        >{busy
                            ? 'Saving…'
                            : singleMode && embedded
                              ? `Save ${working?.name || 'model'} setup and open Fit`
                              : stageIndex < stages.length - 1
                                ? `Save variables for ${working?.name || 'this model'} and continue`
                                : `Save variables for ${working?.name || 'this model'} and open Fit`}</button
                    >
                </div>{/if}
        {/if}
    {/if}
</section>

<style>
    .embedded {
        margin-top: 24px;
    }
    .embedded-heading {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 16px;
        margin-bottom: 16px;
    }
    .embedded-heading h2 {
        margin: 0;
    }
    .embedded-heading p {
        margin: 8px 0 0;
    }
    .embedded-heading .model-variable-context {
        color: #45695a;
        font-size: 13px;
        font-weight: 600;
    }
    .embedded-heading strong {
        white-space: nowrap;
        font-size: 13px;
    }
    .stage-picker {
        margin-bottom: 16px;
    }
    .embedded-picker {
        padding: 0;
        border: 0;
        background: transparent;
    }
    .plan-save {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin: 16px 0 32px;
    }
    .single-details {
        margin-top: 18px;
        border-top: 1px solid #dae3db;
        padding-top: 18px;
    }
    .single-details h2,
    .single-details p {
        margin-top: 0;
    }
    .two-column-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 16px;
        margin-top: 16px;
    }
    .two-column-grid label,
    .single-field {
        display: grid;
        gap: 8px;
        min-width: 0;
    }
    .two-column-grid input,
    .two-column-grid select,
    .single-field input {
        box-sizing: border-box;
        width: 100%;
        min-width: 0;
        height: 38px;
    }
    .single-field,
    .checkbox-field,
    .step-actions {
        margin-top: 16px;
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
    .step-actions,
    .step-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .step-summary {
        justify-content: flex-end;
        margin-top: 8px;
    }
    .text-button {
        border: 0;
        background: transparent;
        color: #154b35;
        padding: 0 3px;
        text-decoration: underline;
    }
    @media (max-width: 620px) {
        .two-column-grid {
            grid-template-columns: 1fr;
        }
        .plan-save {
            justify-content: stretch;
        }
        .plan-save button {
            flex: 1;
        }
    }
</style>
