<script>
    import { onDestroy, onMount } from 'svelte';
    import RateChart from './RateChart.svelte';
    import { formatNumber as num, formatRelativity as rel } from './format.js';
    import { completeVariableDesign, normaliseDefaults } from './modelVariables.js';
    export let api, state, planState, model;
    export let onChanged = () => {};
    let detail = null,
        loading = false,
        busy = false,
        error = '',
        revisionNotice = '',
        revisionNoticeStatusKey = '',
        searchResult = null,
        searchAction = '',
        searchError = '',
        searching = false,
        left = '',
        right = '',
        amendment = null,
        amendmentPreview = null,
        amendmentBusy = false,
        tuningMode = 'automatic',
        selectedCandidateId = '',
        candidateSettings = {
            tune: true,
            cv: 3,
            trials: 8,
            time_limit_minutes: 5,
            candidate: {
                iterations: 120,
                depth: 3,
                learning_rate: 0.05,
                l2_leaf_reg: 3,
            },
            seed: 42,
            min_weight_share: 0.0025,
        },
        candidateDesign = { defaults: normaliseDefaults(), variables: {} },
        designSourceKey = '';
    let mounted = false;
    const pollingCandidates = new Set();
    let planSyncEpoch = 0;
    let detailGeneration = 0;
    let detailRequestEpoch = 0;
    let confirmedSourceSignature = '',
        sourceConfirmed = false;
    $: modelId = model?.model_id || '';
    $: owningDesign = model?.core?.design || null;
    $: nextDesignSourceKey = JSON.stringify({ modelId, design: owningDesign || {} });
    $: if (nextDesignSourceKey !== designSourceKey) {
        designSourceKey = nextDesignSourceKey;
        candidateDesign = {
            defaults: normaliseDefaults(owningDesign?.defaults),
            variables: {},
        };
    }
    $: accepted = detail?.pair_corrections || model?.pair_corrections || [];
    $: acceptedTables = detail?.accepted_tables || {};
    $: partialRef = (planState?.partial_fit_refs || []).find((item) => item.model_id === modelId);
    $: pairStatusById = Object.fromEntries(
        (detail?.pair_statuses || []).map((item) => [item.component_id, item.state]),
    );
    $: acceptedThroughComponentId =
        detail?.accepted_through_component_id || partialRef?.through_component_id || '';
    $: acceptedThroughIndex = acceptedThroughComponentId
        ? accepted.findIndex((pair) => pair.component_id === acceptedThroughComponentId)
        : accepted.length - 1;
    $: candidates = detail?.candidates || [];
    $: candidate =
        candidates.find((item) => item.candidate_id === selectedCandidateId) ||
        candidates[0] ||
        null;
    $: candidateTable = Object.values(candidate?.review?.pair_tables || {})[0] || null;
    $: candidateChart = candidateTable
        ? {
              ...candidateTable,
              kind: 'pair',
              parents: [candidate.a, candidate.b],
              display_label: pairName(candidate),
          }
        : null;
    $: incumbentTrain = candidate?.review?.incumbent_train || {};
    $: candidateTrain = candidate?.review?.candidate_train || {};
    $: trainingLossBefore = Number(incumbentTrain.mean_deviance);
    $: trainingLossAfter = Number(candidateTrain.mean_deviance);
    $: trainingLossChange =
        Number.isFinite(trainingLossBefore) && Number.isFinite(trainingLossAfter)
            ? trainingLossAfter - trainingLossBefore
            : null;
    $: tableApproximationLoss = Number(
        candidate?.review?.component_evidence?.approximation_loss ??
            candidate?.review?.component_evidence?.table_approximation_loss,
    );
    $: grouped = detail?.eligible || { main: model?.core?.factors || [], other: [] };
    $: mainChoices = Array.isArray(grouped) ? grouped : grouped.main || [];
    $: otherChoices = Array.isArray(grouped)
        ? []
        : (grouped.other || []).filter((factor) => !mainChoices.includes(factor));
    $: frozen = Boolean(model?.frozen || model?.adopted_root);
    $: logicalPlanModels = planState?.plan?.models || [];
    $: modelIndex = logicalPlanModels.findIndex((item) => item.model_id === modelId);
    $: effectiveLink =
        planState?.plan?.link ||
        (['poisson', 'tweedie', 'gaussian'].includes(planState?.plan?.family) ? 'log' : null);
    $: interactionSupported = effectiveLink === 'log';
    $: sourceSignature = JSON.stringify({
        modelId,
        sourceModels: logicalPlanModels.slice(0, modelIndex + 1).map((item) => ({
            model_id: item.model_id,
            pairs: (item.pair_corrections || []).map((pair) => pair.component_id),
        })),
        checkpoints: (planState?.stage_statuses || [])
            .filter((item) => {
                const index = logicalPlanModels.findIndex(
                    (model) => model.model_id === (item.model_id || item.stage_id),
                );
                return index >= 0 && index <= modelIndex;
            })
            .map((item) => [
                item.model_id || item.stage_id,
                item.state,
                item.fit?.fit_id,
                item.fit?.prefix_fingerprint,
                item.through_component_id,
            ]),
        completed: (planState?.fit_refs || [])
            .filter((item) => {
                const index = logicalPlanModels.findIndex(
                    (model) => model.model_id === item.model_id,
                );
                return index >= 0 && index <= modelIndex;
            })
            .map((item) => [item.model_id, item.fit_id, item.prefix_fingerprint]),
        partial: (planState?.partial_fit_refs || [])
            .filter((item) => {
                const index = logicalPlanModels.findIndex(
                    (model) => model.model_id === item.model_id,
                );
                return index >= 0 && index <= modelIndex;
            })
            .map((item) => [
                item.model_id,
                item.fit_id,
                item.prefix_fingerprint,
                item.through_component_id,
            ]),
    });
    $: if (sourceSignature !== confirmedSourceSignature) {
        confirmedSourceSignature = sourceSignature;
        sourceConfirmed = false;
        detailGeneration += 1;
        detailRequestEpoch += 1;
        if (detail) detail = { ...detail, candidates: [] };
        searchResult = null;
        searchAction = '';
        searchError = '';
        searching = false;
        selectedCandidateId = '';
        if (mounted && modelId) void refresh();
    }
    $: canKeep =
        sourceConfirmed &&
        candidate?.status === 'review' &&
        !candidate?.stale &&
        !candidate?.no_change;
    $: downstreamNeedsRefit = (planState?.stage_statuses || [])
        .filter(
            (item) =>
                item.state === 'needs_refit' &&
                logicalPlanModels.findIndex((entry) => entry.model_id === item.model_id) >
                    modelIndex,
        )
        .map((item) => item.name);
    $: downstreamStatusKey = JSON.stringify(
        (planState?.stage_statuses || [])
            .filter(
                (item) =>
                    logicalPlanModels.findIndex((entry) => entry.model_id === item.model_id) >
                    modelIndex,
            )
            .map((item) => [item.model_id, item.state]),
    );
    $: if (
        revisionNotice &&
        revisionNoticeStatusKey &&
        downstreamStatusKey !== revisionNoticeStatusKey &&
        !downstreamNeedsRefit.length
    ) {
        revisionNotice = '';
        revisionNoticeStatusKey = '';
    }
    export async function runSearch(action) {
        if (!interactionSupported || !sourceConfirmed || !modelId || searching || busy) return;
        const searchGeneration = detailGeneration;
        const searchSourceSignature = sourceSignature;
        searching = true;
        searchAction = action;
        searchError = '';
        try {
            const result = await api(`core-models/${encodeURIComponent(modelId)}/search`, {
                session_id: state.session_id,
                revision: state.revision,
                action,
                n_bins: candidateDesign.defaults.n_bins,
                min_expected: 3,
                min_cell_share: candidateSettings.min_weight_share,
                top: 20,
            });
            if (searchGeneration !== detailGeneration || searchSourceSignature !== sourceSignature)
                return;
            searchResult = result;
        } catch (cause) {
            if (searchGeneration === detailGeneration && searchSourceSignature === sourceSignature)
                searchError = cause.message;
        } finally {
            if (searchGeneration === detailGeneration && searchSourceSignature === sourceSignature)
                searching = false;
        }
    }
    function factorName(row) {
        return row?.variable || row?.name || row?.factor || 'Candidate factor';
    }
    function pairFromResult(row) {
        const first = row?.a || row?.first || row?.variable_a || row?.left;
        const second = row?.b || row?.second || row?.variable_b || row?.right;
        return typeof first === 'string' && typeof second === 'string' ? [first, second] : null;
    }
    function usePair(row) {
        const pair = pairFromResult(row);
        if (!pair) {
            searchError = 'This search result does not identify two usable source columns.';
            return;
        }
        [left, right] = pair;
        searchResult = null;
        requestAnimationFrame(() =>
            document
                .getElementById(`add-interaction-${modelId}`)
                ?.scrollIntoView({ block: 'center' }),
        );
    }
    async function refresh() {
        if (!modelId) return;
        const requestedModelId = modelId;
        const requestedGeneration = detailGeneration;
        const requestedSourceSignature = sourceSignature;
        const requestEpoch = ++detailRequestEpoch;
        const isCurrentRequest = () =>
            mounted &&
            modelId === requestedModelId &&
            requestedGeneration === detailGeneration &&
            requestedSourceSignature === sourceSignature &&
            requestEpoch === detailRequestEpoch;
        loading = true;
        error = '';
        try {
            const fresh = await api(
                `core-models/${encodeURIComponent(requestedModelId)}/interactions`,
            );
            if (!isCurrentRequest()) return;
            detail = fresh;
            sourceConfirmed = true;
            resumeFittingCandidates(fresh, requestedModelId);
        } catch (cause) {
            if (isCurrentRequest() && !accepted.length) error = cause.message;
        } finally {
            if (isCurrentRequest()) loading = false;
        }
    }
    function advanceParentRevision(result) {
        if (result?.revision == null) return;
        onChanged({
            ...(planState || {}),
            session_id: result.session_id || planState?.session_id,
            revision: result.revision,
        });
        const syncEpoch = ++planSyncEpoch;
        void api('model-plan')
            .then((fresh) => {
                if (mounted && syncEpoch === planSyncEpoch) onChanged(fresh);
            })
            .catch((cause) => {
                if (mounted && syncEpoch === planSyncEpoch)
                    error = `Interaction changes were saved, but the model plan has not refreshed. ${cause.message}`;
            });
    }
    onMount(() => {
        mounted = true;
        void refresh();
        return () => {
            mounted = false;
        };
    });
    onDestroy(() => {
        mounted = false;
    });
    $: if (modelId && detail?.model_id && detail.model_id !== modelId) void refresh();
    async function action(operation, extra = {}) {
        if (!modelId || busy) return;
        if (['move', 'remove'].includes(operation)) {
            const downstream =
                planState?.plan?.models
                    ?.slice(
                        (planState.plan.models || []).findIndex(
                            (item) => item.model_id === modelId,
                        ) + 1,
                    )
                    .map((item) => item.name) || [];
            const affected = [model.name, ...downstream].join(' and ');
            if (
                !window.confirm(
                    `This will revise ${affected} and mark those fits Needs refit. Continue?`,
                )
            )
                return;
        }
        busy = true;
        // Do not let a GET started before this mutation replace its returned
        // interaction view once the write succeeds.
        detailGeneration += 1;
        detailRequestEpoch += 1;
        // A search response from the prior source generation now deliberately
        // ignores its completion handler, so clear its visible state here.
        searching = false;
        searchResult = null;
        searchError = '';
        error = '';
        try {
            const candidateId = extra.candidate_id;
            let result;
            if (operation === 'start') {
                result = await api(`core-models/${encodeURIComponent(modelId)}/interactions`, {
                    session_id: state.session_id,
                    revision: state.revision,
                    model_id: modelId,
                    a: extra.left,
                    b: extra.right,
                    settings: extra.settings || {},
                    design: extra.design || null,
                });
            } else if (operation === 'fit') {
                result = await api(
                    `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(candidateId)}/fit`,
                    {
                        session_id: state.session_id,
                        revision: state.revision,
                        candidate_id: candidateId,
                    },
                );
            } else if (operation === 'keep') {
                result = await api(
                    `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(candidateId)}/accept`,
                    {
                        session_id: state.session_id,
                        revision: state.revision,
                        candidate_id: candidateId,
                    },
                );
            } else if (operation === 'discard') {
                result = await api(
                    `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(candidateId)}/discard`,
                    {
                        session_id: state.session_id,
                        revision: state.revision,
                        candidate_id: candidateId,
                    },
                );
            } else if (operation === 'move') {
                result = await api(
                    `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(extra.pair_id)}/move`,
                    {
                        session_id: state.session_id,
                        revision: state.revision,
                        before_component_id: extra.before_component_id,
                    },
                );
            } else {
                result = await api(
                    `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(extra.pair_id)}/remove`,
                    { session_id: state.session_id, revision: state.revision },
                );
            }
            // Candidate endpoints return interaction state and a new revision. Advance
            // the parent immediately, then refresh its descriptive plan in the
            // background. A slow plan read must never leave accepted-table actions
            // disabled after a successful Keep.
            advanceParentRevision(result);
            detail = result?.interactions || result;
            if (operation === 'start' && result?.candidate?.candidate_id)
                selectedCandidateId = result.candidate.candidate_id;
            const affected = (result?.needs_refit || []).filter((name) => name !== model.name);
            revisionNotice = affected.length
                ? `${affected.join(' → ')} need${affected.length === 1 ? 's' : ''} refitting after this accepted-score change.`
                : '';
            revisionNoticeStatusKey = downstreamStatusKey;
            if (operation === 'fit') void pollCandidate(candidateId, modelId);
            if (operation === 'discard' || operation === 'keep') {
                left = '';
                right = '';
            }
        } catch (cause) {
            error = cause.message;
        } finally {
            busy = false;
        }
    }
    function resumeFittingCandidates(view, requestedModelId) {
        for (const item of view?.candidates || []) {
            if (item.status === 'fitting') void pollCandidate(item.candidate_id, requestedModelId);
        }
    }
    async function pollCandidate(candidateId, requestedModelId = modelId) {
        const pollKey = `${requestedModelId}:${candidateId}`;
        if (pollingCandidates.has(pollKey)) return;
        pollingCandidates.add(pollKey);
        try {
            while (mounted && modelId === requestedModelId) {
                await new Promise((resolve) => setTimeout(resolve, 1000));
                if (!mounted || modelId !== requestedModelId) return;
                try {
                    const requestedGeneration = detailGeneration;
                    const requestEpoch = ++detailRequestEpoch;
                    const fresh = await api(
                        `core-models/${encodeURIComponent(requestedModelId)}/interactions`,
                    );
                    if (
                        !mounted ||
                        fresh.model_id !== requestedModelId ||
                        modelId !== requestedModelId ||
                        requestedGeneration !== detailGeneration ||
                        requestEpoch !== detailRequestEpoch
                    )
                        continue;
                    detail = fresh;
                    const current = (fresh.candidates || []).find(
                        (item) => item.candidate_id === candidateId,
                    );
                    if (!current || current.status !== 'fitting') {
                        // Fitting completes asynchronously and advances the server
                        // revision.  Bring the parent state forward before exposing
                        // Keep/Discard, otherwise their write would use the revision
                        // from the original Fit request.
                        advanceParentRevision(fresh);
                        return;
                    }
                } catch (cause) {
                    error = `Cannot refresh interaction candidate. ${cause.message}`;
                }
            }
        } finally {
            pollingCandidates.delete(pollKey);
        }
    }
    function startCandidate() {
        if (!interactionSupported || !sourceConfirmed || !left || !right || left === right) {
            error = 'Choose two different variables.';
            return;
        }
        candidateSettings = {
            ...candidateSettings,
            tune: tuningMode === 'automatic',
        };
        void action('start', {
            left,
            right,
            settings: candidateSettings,
            design: pairDesign(),
        });
    }
    function pairDesign() {
        const sourceVariables = owningDesign?.variables || {};
        return {
            defaults: normaliseDefaults(candidateDesign.defaults),
            variables: Object.fromEntries(
                [left, right].map((factor) => [
                    factor,
                    completeVariableDesign(factor, sourceVariables, candidateDesign.variables),
                ]),
            ),
        };
    }
    function usesAutomaticBands(factor) {
        return candidateDesign.variables?.[factor]?.n_bins != null;
    }
    function setAutomaticBands(factor, enabled) {
        const variables = { ...candidateDesign.variables };
        if (enabled) {
            variables[factor] = {
                ...(variables[factor] || {}),
                kind: 'step',
                knots: 'quantile',
                n_bins: candidateDesign.defaults.n_bins,
            };
        } else {
            delete variables[factor];
        }
        candidateDesign = { ...candidateDesign, variables };
    }
    function setPairBandCount(factor, value) {
        const bands = Number(value);
        if (!Number.isInteger(bands) || bands < 2 || bands > 200) return;
        candidateDesign = {
            ...candidateDesign,
            variables: {
                ...candidateDesign.variables,
                [factor]: {
                    ...(candidateDesign.variables[factor] || {}),
                    kind: 'step',
                    knots: 'quantile',
                    n_bins: bands,
                },
            },
        };
    }
    function pairName(pair) {
        return (
            pair?.name ||
            pair?.label ||
            pair?.factors?.join(' × ') ||
            [pair?.a, pair?.b].filter(Boolean).join(' × ') ||
            'Interaction table'
        );
    }
    function sourceLabel() {
        if (candidate?.source_label) return candidate.source_label;
        const source = detail?.source?.model_name;
        const prior = accepted.map(pairName);
        const base = source
            ? `Complete accepted score through ${source}`
            : 'Complete accepted score';
        return `${base}${prior.length ? `, including ${prior.join(' + ')}` : ''}`;
    }
    function movePair(index, direction) {
        const target = index + direction;
        if (target < 0 || target >= accepted.length) return;
        const before =
            direction < 0
                ? accepted[target].component_id
                : accepted[target + 1]?.component_id || null;
        void action('move', { pair_id: accepted[index].component_id, before_component_id: before });
    }
    $: acceptedPairStates = accepted.map((pair, index) => {
        const state =
            pairStatusById[pair.component_id] ||
            (partialRef && index > acceptedThroughIndex
                ? 'needs_refit'
                : pair.status || 'accepted');
        // The API describes a frozen, accepted checkpoint as `fitted`; this UI's
        // accepted-table actions use `accepted` as their stable display state.
        return state === 'fitted' ? 'accepted' : state;
    });
    $: acceptedCurrentCount = acceptedPairStates.filter((status) => status === 'accepted').length;
    function openAmendment(pair) {
        amendment = {
            component_id: pair.component_id || pair.pair_id,
            name: pairName(pair),
            operation: 'cap',
            lower: '',
            upper: '',
            decimals: 2,
            new_name: `${pairName(pair)} amended`,
        };
        amendmentPreview = null;
        error = '';
    }
    function amendmentPayload(includeName = false) {
        const payload = {
            session_id: state.session_id,
            revision: state.revision,
            operation: amendment.operation,
        };
        if (amendment.operation === 'cap') {
            if (amendment.lower !== '') payload.lower = Number(amendment.lower);
            if (amendment.upper !== '') payload.upper = Number(amendment.upper);
        } else {
            payload.decimals = Number(amendment.decimals);
        }
        if (includeName) payload.new_name = amendment.new_name.trim();
        return payload;
    }
    async function previewAmendment() {
        if (!amendment || amendmentBusy) return;
        amendmentBusy = true;
        error = '';
        try {
            amendmentPreview = await api(
                `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(amendment.component_id)}/preview`,
                amendmentPayload(),
            );
        } catch (cause) {
            error = cause.message;
        } finally {
            amendmentBusy = false;
        }
    }
    async function applyAmendment() {
        if (!amendment || amendmentBusy || !amendmentPreview || !amendment.new_name.trim()) return;
        if (
            !window.confirm(
                `Apply this table amendment as ${amendment.new_name.trim()}? This revises ${model.name} and later models will need refitting.`,
            )
        )
            return;
        amendmentBusy = true;
        error = '';
        try {
            const result = await api(
                `core-models/${encodeURIComponent(modelId)}/interactions/${encodeURIComponent(amendment.component_id)}/apply`,
                amendmentPayload(true),
            );
            onChanged(await api('model-plan'));
            detail = result.interactions || result;
            const affected = (result?.needs_refit || []).filter((name) => name !== model.name);
            revisionNotice = affected.length
                ? `${affected.join(' → ')} need${affected.length === 1 ? 's' : ''} refitting after this table amendment.`
                : '';
            revisionNoticeStatusKey = downstreamStatusKey;
            amendment = null;
            amendmentPreview = null;
        } catch (cause) {
            error = cause.message;
        } finally {
            amendmentBusy = false;
        }
    }
</script>

<section class="core-interactions model-card" aria-label="Interactions">
    <div class="section-heading">
        <div>
            <h2>Interactions</h2>
            <p>
                Accepted tables are part of this GLM’s score. A candidate stays separate until you
                review and keep it.
            </p>
        </div>
        <span>{acceptedCurrentCount} accepted</span>
    </div>
    {#if error}<p class="message error" role="alert">{error}</p>{/if}
    {#if revisionNotice}<p class="message" role="status">{revisionNotice}</p>{/if}
    {#if searchError}<p class="message error" role="alert">{searchError}</p>{/if}
    {#if searching}<p class="message" role="status">Searching training rows…</p>{/if}
    {#if searchResult}<section class="search-results" aria-label="Interaction discovery results">
            <h3>
                {searchAction === 'factors'
                    ? 'Missing-variable candidates'
                    : 'Interaction candidates'}
            </h3>
            <p>
                <strong>Training rows only.</strong>
                {searchResult.source || 'Complete accepted scorer.'}
                {#if searchResult.note}
                    {searchResult.note}{/if}
            </p>
            {#if searchResult.rows?.length}<div class="result-table">
                    <table>
                        <thead
                            ><tr>
                                <th>{searchAction === 'factors' ? 'Variable' : 'Pair'}</th>
                                <th>Evidence</th>
                                {#if searchAction === 'interactions'}<th></th>{/if}
                            </tr></thead
                        >
                        <tbody
                            >{#each searchResult.rows as row}<tr>
                                    <td
                                        >{searchAction === 'factors'
                                            ? factorName(row)
                                            : pairName(row)}</td
                                    >
                                    <td>
                                        Signal {num(row.signal)}
                                        {#if row.n_bands != null}
                                            · {row.n_bands} bands
                                        {/if}
                                        {#if row.n_cells != null}
                                            · {row.n_cells} supported cells
                                        {/if}
                                        {#if row.sd_log_ae != null}
                                            · residual spread {num(row.sd_log_ae)}
                                        {/if}
                                        {#if row.max_abs_log_ae != null}
                                            · maximum residual {num(row.max_abs_log_ae)}
                                        {/if}
                                        {#if row.worst_cell}
                                            · largest cell {row.worst_cell}
                                        {/if}
                                    </td>
                                    {#if searchAction === 'interactions'}<td
                                            ><button
                                                onclick={() => usePair(row)}
                                                disabled={Boolean(candidate)}
                                                >Use for interaction</button
                                            ></td
                                        >{/if}
                                </tr>{/each}</tbody
                        >
                    </table>
                </div>
            {:else}<p>No candidates met the current training-data threshold.</p>{/if}
        </section>{/if}
    {#if frozen}<p class="message">
            This saved scorer is frozen, including its interaction tables.
        </p>
    {:else}
        {#if accepted.length}<ol class="interaction-list" aria-label="Accepted interaction tables">
                {#each accepted as pair, index}
                    {@const acceptedTable = acceptedTables[pair.component_id || pair.pair_id]}
                    <li>
                        <div class="interaction-row-header">
                            <div>
                                <strong>{index + 1}. {pairName(pair)}</strong><small
                                    >{acceptedPairStates[index] === 'needs_refit'
                                        ? 'Needs refit'
                                        : 'Accepted table'}</small
                                >
                            </div>
                            <div class="row-actions">
                                <button
                                    disabled={busy || acceptedPairStates[index] !== 'accepted'}
                                    onclick={() => movePair(index, -1)}>Up</button
                                ><button
                                    disabled={busy || acceptedPairStates[index] !== 'accepted'}
                                    onclick={() => movePair(index, 1)}>Down</button
                                ><button
                                    disabled={busy || acceptedPairStates[index] !== 'accepted'}
                                    onclick={() =>
                                        action('remove', {
                                            pair_id: pair.component_id || pair.pair_id,
                                        })}>Remove</button
                                ><button
                                    disabled={busy || acceptedPairStates[index] !== 'accepted'}
                                    onclick={() => openAmendment(pair)}>Amend table</button
                                >
                            </div>
                        </div>
                        {#if acceptedTable}<RateChart
                                table={{
                                    ...acceptedTable,
                                    kind: 'pair',
                                    parents: pair.factors || [pair.a, pair.b].filter(Boolean),
                                    display_label: pairName(pair),
                                }}
                                variable={pairName(pair)}
                                label="accepted interaction correction"
                                fittedLabel="Accepted"
                            />
                            <details class="candidate-cells">
                                <summary>Accepted interaction table cells</summary>
                                <div class="cell-table">
                                    <table>
                                        <thead
                                            ><tr
                                                ><th>First factor</th><th>Second factor</th><th
                                                    >Correction</th
                                                ><th>Fitting weight</th></tr
                                            ></thead
                                        >
                                        <tbody
                                            >{#each acceptedTable.rows || [] as row}<tr
                                                    ><td>{row.label_a ?? '—'}</td><td
                                                        >{row.label_b ?? '—'}</td
                                                    ><td>{rel(row.relativity ?? row.fitted)}</td><td
                                                        >{num(row.fitting_weight)}</td
                                                    ></tr
                                                >{/each}</tbody
                                        >
                                    </table>
                                </div>
                            </details>{/if}
                    </li>{/each}
            </ol>
        {:else}<p class="empty">
                No interaction tables have been accepted. You can continue to the next model without
                one.
            </p>{/if}
        {#if amendment}<section class="table-amendment" aria-label="Amend interaction table">
                <h3>Amend {amendment.name}</h3>
                <p>
                    Preview the table change before publishing a new scorer. Later models will need
                    refitting.
                </p>
                <label
                    >Amendment<select
                        aria-label="Table amendment"
                        bind:value={amendment.operation}
                        onchange={() => (amendmentPreview = null)}
                        ><option value="cap">Cap correction values</option><option value="round"
                            >Round correction values</option
                        ></select
                    ></label
                >
                {#if amendment.operation === 'cap'}<div class="pair-fields">
                        <label
                            >Lower cap<input
                                aria-label="Lower cap"
                                type="number"
                                step="0.01"
                                bind:value={amendment.lower}
                                oninput={() => (amendmentPreview = null)}
                            /></label
                        ><label
                            >Upper cap<input
                                aria-label="Upper cap"
                                type="number"
                                step="0.01"
                                bind:value={amendment.upper}
                                oninput={() => (amendmentPreview = null)}
                            /></label
                        >
                    </div>{:else}<label
                        >Decimal places<input
                            aria-label="Decimal places"
                            type="number"
                            min="0"
                            step="1"
                            bind:value={amendment.decimals}
                            oninput={() => (amendmentPreview = null)}
                        /></label
                    >{/if}
                <div class="candidate-actions">
                    <button onclick={previewAmendment} disabled={amendmentBusy}
                        >Preview table amendment</button
                    >
                    {#if amendmentPreview}<label
                            >New fitted model name<input
                                aria-label="Amended model name"
                                bind:value={amendment.new_name}
                            /></label
                        ><button
                            class="primary"
                            onclick={applyAmendment}
                            disabled={amendmentBusy || !amendment.new_name.trim()}
                            >Apply table amendment</button
                        >{/if}<button
                        onclick={() => {
                            amendment = null;
                            amendmentPreview = null;
                        }}
                        disabled={amendmentBusy}>Cancel</button
                    >
                </div>
                {#if amendmentPreview}<p class="message" role="status">
                        Preview ready. Applying creates a new scorer; the earlier one is retained.
                    </p>{/if}
            </section>{/if}
        {#if candidate}<section class="candidate" aria-label="Interaction candidate">
                <h3>Interaction candidate</h3>
                {#if candidates.length > 1}<label class="candidate-choice"
                        >Interaction candidate<select
                            aria-label="Interaction candidate"
                            bind:value={selectedCandidateId}
                            >{#each candidates as item}<option value={item.candidate_id}
                                    >{pairName(item)} · {item.status}</option
                                >{/each}</select
                        ></label
                    >{/if}
                <p><strong>Not included in score.</strong> Offset: {sourceLabel()}.</p>
                {#if candidate.status === 'review'}<p>
                        Candidate fitted on training data. Review the table and fit comparison
                        before keeping it.
                    </p>
                    <dl class="metrics" aria-label="Candidate fit comparison">
                        <div>
                            <dt>Training loss before</dt>
                            <dd>{num(trainingLossBefore)}</dd>
                        </div>
                        <div>
                            <dt>Training loss after</dt>
                            <dd>{num(trainingLossAfter)}</dd>
                        </div>
                        <div>
                            <dt>Training loss change</dt>
                            <dd>{num(trainingLossChange)}</dd>
                        </div>
                        {#if Number.isFinite(tableApproximationLoss)}<div>
                                <dt>Table approximation loss</dt>
                                <dd>{num(tableApproximationLoss)}</dd>
                            </div>{/if}
                    </dl>
                    {#if candidateChart}<RateChart
                            table={candidateChart}
                            variable={pairName(candidate)}
                            label="interaction correction"
                            fittedLabel="Candidate"
                        />
                        <details class="candidate-cells">
                            <summary>Interaction table cells</summary>
                            <div class="cell-table">
                                <table>
                                    <thead
                                        ><tr>
                                            <th>First factor</th><th>Second factor</th>
                                            <th>Correction</th><th>Fitting weight</th>
                                        </tr></thead
                                    >
                                    <tbody
                                        >{#each candidateChart.rows || [] as row}<tr>
                                                <td>{row.label_a ?? '—'}</td>
                                                <td>{row.label_b ?? '—'}</td>
                                                <td>{rel(row.relativity ?? row.fitted)}</td>
                                                <td>{num(row.fitting_weight)}</td>
                                            </tr>{/each}</tbody
                                    >
                                </table>
                            </div>
                        </details>{/if}
                {:else if candidate.status === 'no_change'}<p>
                        No useful table was produced. Keeping it would not change the score.
                    </p>
                {:else if candidate.stale || ['stale', 'needs_rebuild'].includes(candidate.status)}<p
                    >
                        This candidate needs rebuild because its preceding accepted score has
                        changed. It cannot be kept.
                    </p>
                {:else if candidate.status === 'fitting'}<p>Fitting on training data…</p>
                {:else}<p>
                        Fit this candidate on training data. It does not affect A/E, discovery, or
                        downstream offsets yet.
                    </p>{/if}
                <div class="candidate-actions">
                    {#if candidate.status === 'draft' || candidate.status === 'failed'}<button
                            class="primary"
                            onclick={() => action('fit', { candidate_id: candidate.candidate_id })}
                            disabled={busy}>Fit interaction</button
                        >{/if}<button
                        class="primary"
                        onclick={() => action('keep', { candidate_id: candidate.candidate_id })}
                        disabled={busy || !canKeep}>Keep interaction</button
                    ><button
                        onclick={() => action('discard', { candidate_id: candidate.candidate_id })}
                        disabled={busy}>Discard interaction</button
                    >
                </div>
            </section>
        {:else}<section
                id={`add-interaction-${modelId}`}
                class="candidate-picker"
                aria-label="Add interaction"
            >
                <h3>Add interaction</h3>
                <p>
                    Pairs can use a main-effect factor or another eligible predictor. Pair-only
                    factors do not become main effects.
                </p>
                <div class="pair-fields">
                    <label
                        >First variable<select
                            aria-label="Pair first column"
                            bind:value={left}
                            disabled={busy || loading}
                            ><option value="">Choose variable</option
                            >{#if mainChoices.length}<optgroup label="Used in main effects"
                                    >{#each mainChoices as name}<option value={name}>{name}</option
                                        >{/each}</optgroup
                                >{/if}{#if otherChoices.length}<optgroup
                                    label="Other eligible columns"
                                    >{#each otherChoices as name}<option value={name}>{name}</option
                                        >{/each}</optgroup
                                >{/if}</select
                        ></label
                    ><label
                        >Second variable<select
                            aria-label="Pair second column"
                            bind:value={right}
                            disabled={busy || loading}
                            ><option value="">Choose variable</option
                            >{#if mainChoices.length}<optgroup label="Used in main effects"
                                    >{#each mainChoices as name}<option value={name}>{name}</option
                                        >{/each}</optgroup
                                >{/if}{#if otherChoices.length}<optgroup
                                    label="Other eligible columns"
                                    >{#each otherChoices as name}<option value={name}>{name}</option
                                        >{/each}</optgroup
                                >{/if}</select
                        ></label
                    ><button
                        class="primary"
                        onclick={startCandidate}
                        disabled={busy || loading || !sourceConfirmed}>Add interaction</button
                    >
                </div>
                <details class="interaction-settings">
                    <summary>Interaction fitting and table bands</summary>
                    <p>
                        Automatic tuning uses training folds and stops when either cap is reached.
                        Manual settings use the values below. Table bands describe the scoring-table
                        axes; they do not change the raw tree inputs.
                    </p>
                    <div class="settings-grid">
                        <label
                            >Fitting approach<select bind:value={tuningMode}>
                                <option value="automatic">Automatic tuning</option>
                                <option value="manual">Manual tree settings</option>
                            </select></label
                        >
                        {#if tuningMode === 'automatic'}<label
                                >Cross-validation folds<input
                                    type="number"
                                    min="2"
                                    bind:value={candidateSettings.cv}
                                /></label
                            ><label
                                >Maximum candidate trials<input
                                    type="number"
                                    min="1"
                                    bind:value={candidateSettings.trials}
                                /></label
                            ><label
                                >Total tuning time limit (minutes)<input
                                    type="number"
                                    min="1"
                                    bind:value={candidateSettings.time_limit_minutes}
                                /></label
                            >{:else}<label
                                >Iterations<input
                                    type="number"
                                    min="1"
                                    bind:value={candidateSettings.candidate.iterations}
                                /></label
                            ><label
                                >Depth<input
                                    type="number"
                                    min="1"
                                    bind:value={candidateSettings.candidate.depth}
                                /></label
                            ><label
                                >Learning rate<input
                                    type="number"
                                    min="0.001"
                                    step="0.001"
                                    bind:value={candidateSettings.candidate.learning_rate}
                                /></label
                            ><label
                                >L2 regularisation<input
                                    type="number"
                                    min="0"
                                    step="0.1"
                                    bind:value={candidateSettings.candidate.l2_leaf_reg}
                                /></label
                            >{/if}<label
                            >Minimum support share<input
                                type="number"
                                min="0"
                                step="0.0001"
                                bind:value={candidateSettings.min_weight_share}
                            /></label
                        ><label
                            >Table bands per numeric factor<input
                                type="number"
                                min="2"
                                bind:value={candidateDesign.defaults.n_bins}
                            /></label
                        >
                        {#each [left, right].filter(Boolean) as factor}<label
                                class="pair-band-override"
                                ><span
                                    ><input
                                        type="checkbox"
                                        checked={usesAutomaticBands(factor)}
                                        onchange={(event) =>
                                            setAutomaticBands(factor, event.currentTarget.checked)}
                                    />
                                    Override {factor} table bands</span
                                >{#if usesAutomaticBands(factor)}<input
                                        aria-label={`${factor} interaction table bands`}
                                        type="number"
                                        min="2"
                                        value={candidateDesign.variables[factor].n_bins}
                                        onchange={(event) =>
                                            setPairBandCount(factor, event.currentTarget.value)}
                                    />{:else}<small
                                        >Uses the model default bands unless this factor already has
                                        saved cuts.</small
                                    >{/if}</label
                            >{/each}
                    </div>
                </details>
            </section>{/if}
    {/if}
</section>

<style>
    .section-heading {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: start;
    }
    .section-heading h2,
    .candidate h3,
    .candidate-picker h3 {
        margin: 0;
    }
    .section-heading p {
        margin: 8px 0 0;
    }
    .section-heading > span,
    small {
        color: #526e5d;
        font-size: 13px;
    }
    .interaction-list {
        list-style: none;
        padding: 0;
        margin: 18px 0;
        border-top: 1px solid var(--border, #d8e3dc);
    }
    .interaction-list li {
        display: block;
        padding: 12px 0;
        border-bottom: 1px solid var(--border, #d8e3dc);
    }
    .interaction-row-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 16px;
    }
    .interaction-list small {
        display: block;
        margin-top: 4px;
    }
    .row-actions,
    .candidate-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    .candidate,
    .candidate-picker {
        margin-top: 20px;
        padding-top: 18px;
        border-top: 1px solid var(--border, #d8e3dc);
    }
    .candidate > p,
    .candidate-picker > p {
        margin: 8px 0 14px;
    }
    .pair-fields {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) auto;
        align-items: end;
        gap: 12px;
    }
    label {
        display: grid;
        gap: 7px;
        min-width: 0;
        color: #516b60;
        font-size: 13px;
    }
    select {
        width: 100%;
        min-width: 0;
    }
    .metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 16px;
        margin: 14px 0;
    }
    .metrics div {
        min-width: 120px;
    }
    .metrics dt {
        color: #526e5d;
        font-size: 12px;
    }
    .metrics dd {
        margin: 3px 0 0;
        font-weight: 650;
    }
    .interaction-settings {
        margin-top: 16px;
    }
    .interaction-settings p {
        margin: 10px 0 0;
        color: #526e5d;
    }
    .settings-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin-top: 12px;
    }
    .pair-band-override span {
        display: flex;
        align-items: center;
        gap: 7px;
    }
    .pair-band-override small {
        color: #526e5d;
        font-weight: 400;
    }
    .candidate-choice {
        margin-top: 12px;
        max-width: 320px;
    }
    .candidate-cells {
        margin-top: 14px;
    }
    .cell-table {
        overflow: auto;
        margin-top: 10px;
    }
    .cell-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .cell-table th,
    .cell-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid var(--border, #d8e3dc);
    }
    .search-results {
        margin-top: 18px;
        padding: 14px;
        border: 1px solid var(--border, #d8e3dc);
        background: #f7faf7;
    }
    .search-results h3,
    .search-results p {
        margin: 0;
    }
    .search-results p {
        margin-top: 8px;
        color: #526e5d;
    }
    .result-table {
        overflow: auto;
        margin-top: 12px;
    }
    .result-table table {
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
    }
    .result-table th,
    .result-table td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid var(--border, #d8e3dc);
    }
    .empty {
        color: #526e5d;
    }
    @media (max-width: 900px) {
        .pair-fields,
        .settings-grid {
            grid-template-columns: 1fr;
        }
        .interaction-row-header {
            align-items: start;
            flex-direction: column;
        }
    }
</style>
