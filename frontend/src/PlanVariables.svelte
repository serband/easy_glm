<script>
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
    let working = null;
    let baseline = '';
    let jsonDraft = '';
    let validity = '';
    let error = '';
    let busy = false;
    let catalogue = null;
    $: plan = planState?.plan || null;
    $: stages = plan?.stages || [];
    $: if (!selectedStageId && stages.length) selectedStageId = stages[0].component_id;
    $: stageIndex = stages.findIndex((stage) => stage.component_id === selectedStageId);
    $: current = stageIndex >= 0 ? stages[stageIndex] : null;
    $: sourceKey = JSON.stringify(current || null);
    $: if (current && sourceKey !== baseline && !busy) {
        working = structuredClone(current);
        baseline = sourceKey;
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
    $: dirty = Boolean(working && (JSON.stringify(working) !== baseline || jsonDraft));
    $: onDirty(dirty);
    $: isFitted = Boolean(
        planState?.stage_statuses?.find((item) => item.stage_id === current?.component_id)
            ?.state === 'fitted',
    );
    $: locked =
        (Boolean(plan?.adopted_model_name) && stageIndex === 0) || (isFitted && !revisionIntent);
    $: sourceColumns =
        planState?.safe_column_details?.[stageIndex] ||
        (planState?.safe_columns?.[stageIndex] || []).map(
            (name) => state?.columns?.find((entry) => entry.name === name) || { name, dtype: '' },
        );
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
    function change(next) {
        working = { ...working, ...next };
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
                plan: { ...plan, stages: stagesNext },
            });
            baseline = JSON.stringify(
                result.plan?.stages?.find((item) => item.component_id === working.component_id) ||
                    working,
            );
            jsonDraft = '';
            onChanged(result);
        } catch (e) {
            error = e.message;
        } finally {
            busy = false;
        }
    }
    function discard() {
        working = current ? structuredClone(current) : null;
        baseline = JSON.stringify(current || null);
        jsonDraft = '';
        validity = '';
    }
</script>

<section class="plan-variables workflow-page" aria-label="Stage variables">
    <div class="heading">
        <div>
            <div class="eyebrow">MODEL PLAN</div>
            <h1>Set up a stage</h1>
            <p>
                Choose variables for this stage. Data roles, the target and the split are shared and
                cannot be changed here.
            </p>
        </div>
    </div>
    {#if !plan}<section class="model-card">
            <p>Start a model plan in Project & data before choosing stage variables.</p>
        </section>{:else}<section class="model-card stage-picker">
            <label
                >Model stage<select
                    aria-label="Model stage"
                    value={selectedStageId}
                    onchange={(event) => {
                        if (dirty && !window.confirm('Discard unsaved stage setup changes?')) {
                            event.currentTarget.value = selectedStageId;
                            return;
                        }
                        selectedStageId = event.currentTarget.value;
                    }}
                    >{#each stages as item, index}<option value={item.component_id}
                            >{index + 1}. {item.name}</option
                        >{/each}</select
                ></label
            >{#if current && stageIndex > 0}<p class="help-text">
                    This stage automatically offsets the complete score through {stages[
                        stageIndex - 1
                    ].name}.
                </p>{/if}
        </section>
        <button class="quiet" onclick={onSharedSetup}>Change shared data setup</button
        >{#if error}<div class="message error" role="alert">{error}</div>{/if}{#if locked}<div
                class="message"
            >
                This stage is already fitted. Go to Project &amp; data and choose Create revised
                plan before changing it.
            </div>{/if}{#if working && working.kind === 'catalogue'}<section class="model-card">
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
                />{#if catalogue}<p>{catalogue.filename} · {catalogue.rows} rows</p>
                    <label
                        >Unique key<select
                            aria-label="Catalogue key"
                            value={working.settings?.key || ''}
                            onchange={(event) => setCatalogueKey(event.currentTarget.value)}
                            ><option value="">Choose key</option
                            >{#each catalogue.key_columns?.length ? catalogue.key_columns : catalogue.columns || [] as column}<option
                                    value={column.name || column}>{column.name || column}</option
                                >{/each}</select
                        ></label
                    >
                    <p>Select attributes below. The key is not a model variable.</p>{/if}
            </section>{/if}{#if working}<fieldset disabled={locked}>
                <ModelVariables
                    {columns}
                    factors={working.factors || []}
                    design={working.design}
                    inheritedDesign={{ defaults: planState?.design_defaults, variables: {} }}
                    method={working.kind === 'catalogue' ? 'catboost' : working.kind}
                    onChange={change}
                    onValidity={(value) => (validity = value)}
                    onDraftText={(value) => (jsonDraft = value)}
                />
            </fieldset>
            <div class="plan-save">
                <button onclick={discard} disabled={!dirty || busy}>Discard changes</button><button
                    class="primary"
                    onclick={save}
                    disabled={!dirty || busy || !!validity}
                    >{busy ? 'Saving…' : 'Save stage setup'}</button
                >
            </div>{/if}{/if}
</section>

<style>
    .stage-picker {
        margin-bottom: 16px;
    }
    .plan-save {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin: 16px 0 32px;
    }
    @media (max-width: 620px) {
        .plan-save {
            justify-content: stretch;
        }
        .plan-save button {
            flex: 1;
        }
    }
</style>
