<script>
    import { onDestroy, tick } from 'svelte';
    import { formatNumber as num } from './format.js';

    export let api, state, setup, context, prepare, onChange;
    export let disabled = false;

    const families = ['poisson', 'gamma', 'gaussian', 'binomial', 'tweedie'];
    const links = {
        poisson: ['log', 'identity'],
        gamma: ['log', 'identity'],
        gaussian: ['identity', 'log'],
        binomial: ['logit'],
        tweedie: ['log', 'identity'],
    };
    const pageSize = 10;
    let family = 'poisson',
        link = '',
        tweediePower = 1.5,
        l1Ratio = 1,
        pathLength = 20,
        repeats = 5,
        seed = 42,
        includeUnassigned = true,
        divideTarget = false,
        divideTouched = false;
    let job = null,
        result = null,
        pending = false,
        starting = false,
        error = '',
        note = '',
        query = '',
        statusFilter = 'all',
        page = 0,
        selected = new Set(),
        expanded = new Set();
    let sequence = 0,
        timer = null,
        alive = true,
        seenScope = '',
        reportScope = '';

    $: weight = setup?.assignments.weight;
    $: if (!divideTouched) divideTarget = Boolean(weight);
    $: options = {
        family,
        link: link || null,
        tweedie_power: Number(tweediePower),
        divide_target_by_weight: Boolean(weight && divideTarget),
        l1_ratio: Number(l1Ratio),
        n_alphas: Number(pathLength),
        repeats: Number(repeats),
        seed: Number(seed),
        include_unassigned: includeUnassigned,
    };
    $: scope = JSON.stringify([context, options]);
    $: if (scope !== seenScope) invalidate(scope);
    $: candidates =
        state?.columns.filter(
            (column) =>
                (setup?.roles.predictor || []).includes(column.name) ||
                (includeUnassigned && (setup?.roles.unassigned || []).includes(column.name)),
        ) || [];
    $: validOptions =
        families.includes(family) &&
        (!link || links[family].includes(link)) &&
        (family !== 'tweedie' ||
            (Number.isFinite(options.tweedie_power) &&
                options.tweedie_power > 1 &&
                options.tweedie_power < 2)) &&
        Number.isFinite(options.l1_ratio) &&
        options.l1_ratio > 0 &&
        options.l1_ratio <= 1 &&
        Number.isInteger(options.n_alphas) &&
        options.n_alphas >= 2 &&
        options.n_alphas <= 100 &&
        Number.isInteger(options.repeats) &&
        options.repeats >= 1 &&
        options.repeats <= 20 &&
        Number.isInteger(options.seed) &&
        options.seed >= 0 &&
        options.seed <= 4294967295;
    $: filtered = (result?.rows || []).filter(
        (row) =>
            (!query ||
                [row.variable, row.raw_name].some((value) =>
                    value?.toLowerCase().includes(query.trim().toLowerCase()),
                )) &&
            (statusFilter === 'all' || row.status === statusFilter),
    );
    $: actionable = new Set(
        (result?.rows || [])
            .filter(
                (row) =>
                    ['signal', 'no_signal'].includes(row.status) &&
                    typeof row.raw_name === 'string',
            )
            .map((row) => row.raw_name),
    );
    $: effectiveOffset =
        setup?.assignments.offset ||
        (setup?.assignments.current_premium
            ? 'log_' +
              (setup.renames[setup.assignments.current_premium] ||
                  setup.assignments.current_premium) +
              ' (from current premium)'
            : null);

    function currentScope() {
        return JSON.stringify([context, options]);
    }
    function clearTimer() {
        if (timer) clearTimeout(timer);
        timer = null;
    }
    function cancelJob(packet) {
        if (packet && ['queued', 'running'].includes(packet.status))
            void api('feature-selections/' + encodeURIComponent(packet.id) + '/cancel', {
                session_id: state.session_id,
                revision: state.revision,
            }).catch(() => {});
    }
    function invalidate(nextScope, message = 'Settings changed. Run feature selection again.') {
        const hadReport = pending || job || result;
        sequence += 1;
        clearTimer();
        cancelJob(job);
        seenScope = nextScope;
        reportScope = '';
        job = null;
        result = null;
        pending = false;
        selected = new Set();
        expanded = new Set();
        page = 0;
        error = '';
        note = hadReport ? message : '';
    }
    function isCurrent(request, requestedScope) {
        return alive && request === sequence && requestedScope === currentScope();
    }
    function sameGeneration(packet) {
        return (
            packet.session_id === state.session_id &&
            packet.project_id === state.project_id &&
            packet.revision === state.revision
        );
    }
    function accept(packet, request, requestedScope) {
        if (!isCurrent(request, requestedScope)) {
            cancelJob(packet);
            return;
        }
        if (
            !sameGeneration(packet) ||
            (job && (packet.id !== job.id || packet.fingerprint !== job.fingerprint))
        ) {
            cancelJob(packet);
            invalidate(currentScope());
            return;
        }
        job = packet;
        pending = ['queued', 'running'].includes(packet.status);
        if (packet.status === 'complete') {
            if (!packet.result) {
                error = 'The selection run returned no results. Run it again.';
                return;
            }
            result = packet.result;
            reportScope = requestedScope;
            note = '';
        } else if (packet.status === 'failed')
            error = packet.message || 'Feature selection failed.';
        else if (packet.status === 'stale') note = 'Settings changed. Run feature selection again.';
        else if (packet.status === 'cancelled') note = 'Feature selection cancelled.';
        else if (pending) timer = setTimeout(() => poll(packet, request, requestedScope), 500);
        else error = 'The selection run returned an unknown status. Run it again.';
    }
    async function poll(packet, request, requestedScope) {
        if (!isCurrent(request, requestedScope)) return;
        try {
            accept(
                await api('feature-selections/' + encodeURIComponent(packet.id)),
                request,
                requestedScope,
            );
        } catch (failure) {
            if (isCurrent(request, requestedScope)) {
                pending = false;
                error = failure.message;
            }
        }
    }
    async function run() {
        if (
            disabled ||
            pending ||
            starting ||
            !validOptions ||
            !candidates.length ||
            !setup?.assignments.target
        )
            return;
        starting = true;
        try {
            if (!prepare()) return;
            await tick();
            if (!alive || disabled) return;
            invalidate(currentScope(), '');
            const requestedScope = currentScope(),
                request = ++sequence;
            pending = true;
            note = '';
            try {
                const packet = await api('variables/feature-selection', {
                    session_id: state.session_id,
                    revision: state.revision,
                    setup: structuredClone(setup),
                    options: structuredClone(options),
                });
                accept(packet, request, requestedScope);
            } catch (failure) {
                if (isCurrent(request, requestedScope)) {
                    pending = false;
                    error = failure.message;
                }
            }
        } finally {
            starting = false;
        }
    }
    function choose(raw, checked) {
        if (!actionable.has(raw)) return;
        const next = new Set(selected);
        if (checked) next.add(raw);
        else next.delete(raw);
        selected = next;
    }
    function chooseGroup(status) {
        selected = new Set(
            filtered
                .filter((row) => row.status === status && actionable.has(row.raw_name))
                .map((row) => row.raw_name),
        );
    }
    function toggleDetails(raw) {
        const next = new Set(expanded);
        if (next.has(raw)) next.delete(raw);
        else next.add(raw);
        expanded = next;
    }
    async function stage(action) {
        if (disabled || !selected.size || reportScope !== currentScope()) return;
        try {
            await onChange(action, [...selected], context);
        } catch (failure) {
            error = failure.message;
        }
    }
    function changeFamily(next) {
        family = next;
        link = '';
        if (next !== 'tweedie') tweediePower = 1.5;
    }
    function statusLabel(status) {
        return (
            {
                signal: 'Signal detected',
                no_signal: 'No signal detected',
                skipped: 'Skipped',
                failed: 'Failed',
            }[status] || status
        );
    }
    onDestroy(() => {
        alive = false;
        sequence += 1;
        clearTimer();
        cancelJob(job);
    });
</script>

<details class="selection-card" aria-label="One-way feature selection">
    <summary
        >One-way feature selection <span>optional · {candidates.length} candidates</span></summary
    >
    <div class="selection-body">
        <p>
            Looks for one-way signal. Variables can still matter through interactions. No signal
            detected is not proof of no effect.
        </p>
        <p>
            Each candidate competes with four shuffled copies and independent random noise.
            Five-fold CV chooses a penalty; full training rows supply the importance scores. This is
            a screening guide, not a p-value.
        </p>
        <p class="selection-cost">
            {num(candidates.length)} candidate CV fits · five folds and {num(pathLength)} penalty values
            each · repeated importance checks. No automatic sampling; this may take time.
        </p>
        <div class="selection-role-note">
            Target: <b>{setup?.assignments.target || 'Choose a target above'}</b> · Weight:
            <b>{weight || 'None'}</b>
            · Offset: <b>{effectiveOffset || 'None'}</b>
        </div>
        <div class="selection-options">
            <label
                >Family<select
                    aria-label="Feature selection family"
                    value={family}
                    disabled={disabled || pending}
                    onchange={(event) => changeFamily(event.currentTarget.value)}
                >
                    <option value="poisson">Poisson</option><option value="gamma">Gamma</option
                    ><option value="gaussian">Gaussian</option><option value="binomial"
                        >Binomial</option
                    ><option value="tweedie">Tweedie</option>
                </select></label
            >
            <label
                >Link<select
                    aria-label="Feature selection link"
                    value={link}
                    disabled={disabled || pending}
                    onchange={(event) => (link = event.currentTarget.value)}
                >
                    <option value="">Family default</option
                    >{#each links[family] as available}<option value={available}>{available}</option
                        >{/each}
                </select></label
            >
            {#if family === 'tweedie'}<label
                    >Tweedie power<input
                        aria-label="Feature selection Tweedie power"
                        type="number"
                        min="1.01"
                        max="1.99"
                        step=".05"
                        bind:value={tweediePower}
                        disabled={disabled || pending}
                    /></label
                >{/if}
        </div>
        <div class="selection-toggles">
            {#if weight}<label
                    ><input
                        type="checkbox"
                        aria-label="Divide target by weight for feature selection"
                        checked={divideTarget}
                        disabled={disabled || pending}
                        onchange={(event) => {
                            divideTouched = true;
                            divideTarget = event.currentTarget.checked;
                        }}
                    />Divide target by weight</label
                >{/if}
            <label
                ><input
                    type="checkbox"
                    aria-label="Include unassigned candidates"
                    bind:checked={includeUnassigned}
                    disabled={disabled || pending}
                />Include unassigned candidates</label
            >
        </div>
        <details class="selection-advanced">
            <summary>Advanced settings</summary>
            <div class="selection-options">
                <label
                    >L1 ratio<input
                        aria-label="Feature selection L1 ratio"
                        type="number"
                        min="0.01"
                        max="1"
                        step=".05"
                        bind:value={l1Ratio}
                        disabled={disabled || pending}
                    /></label
                >
                <label
                    >Penalty path length<input
                        aria-label="Feature selection penalty path length"
                        type="number"
                        min="2"
                        max="100"
                        step="1"
                        bind:value={pathLength}
                        disabled={disabled || pending}
                    /></label
                >
                <label
                    >Importance repeats<input
                        aria-label="Feature selection repeats"
                        type="number"
                        min="1"
                        max="20"
                        step="1"
                        bind:value={repeats}
                        disabled={disabled || pending}
                    /></label
                >
                <label
                    >Random seed<input
                        aria-label="Feature selection seed"
                        type="number"
                        min="0"
                        max="4294967295"
                        step="1"
                        bind:value={seed}
                        disabled={disabled || pending}
                    /></label
                >
            </div>
        </details>
        {#if !validOptions}<p role="alert" class="selection-error">
                Check the family/link, Tweedie power (1–2), L1 ratio (above 0 through 1), path
                length (2–100), repeats (1–20) and whole seed (0–4,294,967,295).
            </p>{/if}
        <div class="selection-actions">
            <button
                class="primary"
                onclick={run}
                disabled={disabled ||
                    pending ||
                    starting ||
                    !validOptions ||
                    !candidates.length ||
                    !setup?.assignments.target}>Run one-way feature selection</button
            >
            {#if pending}<button
                    onclick={() => invalidate(currentScope(), 'Feature selection cancelled.')}
                    >Cancel feature selection</button
                >{/if}
        </div>
        {#if pending}<div class="selection-progress" role="status">
                <span
                    >{job?.progress?.message ||
                        job?.message ||
                        'Starting feature selection…'}{#if job?.progress?.current_variable}
                        · {job.progress.current_variable}{/if}</span
                >
                <progress
                    aria-label="Feature selection progress"
                    max={job?.progress?.total || 1}
                    value={job?.progress?.total ? job.progress.completed : undefined}
                ></progress>
                <span
                    >{num(job?.progress?.completed || 0)} of {num(
                        job?.progress?.total || candidates.length,
                    )} candidates · {num(job?.elapsed || 0)} seconds</span
                >
            </div>{/if}
        {#if error}<p role="alert" class="selection-error">{error}</p>{/if}
        {#if note}<p class="selection-note">{note}</p>{/if}
        {#if result}<div class="selection-results">
                <div class="selection-summary">
                    <b
                        >{num(result.tested_count)} tested of {num(result.candidate_count)} candidates</b
                    ><span
                        >{num(result.training_rows)} training rows · {num(result.cv_folds || 5)} CV folds
                        · {num(result.repeats ?? repeats)} importance repeats</span
                    >
                </div>
                {#if result.random_definition}<p class="selection-method">
                        Random control: {result.random_definition}.
                    </p>{/if}
                <div class="selection-controls">
                    <input
                        aria-label="Search feature selection results"
                        placeholder="Find a variable…"
                        bind:value={query}
                        oninput={() => (page = 0)}
                    />
                    <select
                        aria-label="Filter feature selection status"
                        bind:value={statusFilter}
                        onchange={() => (page = 0)}
                        ><option value="all">All outcomes</option><option value="signal"
                            >Signal</option
                        ><option value="no_signal">No signal detected</option><option
                            value="skipped">Skipped</option
                        ><option value="failed">Failed</option></select
                    >
                    <span>{num(filtered.length)} results</span>
                </div>
                <div class="selection-table-wrap">
                    <table>
                        <thead
                            ><tr
                                ><th>Select</th><th>Variable</th><th>Outcome</th><th
                                    >Real importance</th
                                ><th>Strongest control</th><th>Margin</th><th>Details</th></tr
                            ></thead
                        ><tbody>
                            {#each filtered.slice(page * pageSize, (page + 1) * pageSize) as row (row.variable)}
                                <tr
                                    ><td
                                        ><input
                                            type="checkbox"
                                            aria-label={'Select feature ' + row.variable}
                                            checked={selected.has(row.raw_name)}
                                            disabled={!actionable.has(row.raw_name) || disabled}
                                            onchange={(event) =>
                                                choose(row.raw_name, event.currentTarget.checked)}
                                        /></td
                                    >
                                    <td><strong title={row.raw_name}>{row.variable}</strong></td><td
                                        >{statusLabel(row.status)}</td
                                    >
                                    <td>{num(row.importance)}</td><td>{num(row.threshold)}</td><td
                                        >{num(row.margin)}</td
                                    >
                                    <td
                                        ><button
                                            aria-label={'Details for ' + row.variable}
                                            onclick={() => toggleDetails(row.variable)}
                                            >{expanded.has(row.variable) ? 'Hide' : 'Show'}</button
                                        ></td
                                    ></tr
                                >
                                {#if expanded.has(row.variable)}<tr class="selection-detail"
                                        ><td colspan="7"
                                            ><p>
                                                {row.reason ||
                                                    'Training permutation importance: mean weighted deviance increase after shuffling.'}
                                            </p>
                                            <div class="selection-detail-grid">
                                                <span
                                                    >Real: {num(row.importance)} ± {num(
                                                        row.std,
                                                    )}</span
                                                >
                                                {#each row.shadows || [] as score, index}<span
                                                        >Shadow {index + 1}: {num(score)} ± {num(
                                                            row.shadow_stds?.[index],
                                                        )}</span
                                                    >{/each}
                                                <span
                                                    >Random: {num(row.random_importance)} ± {num(
                                                        row.random_std,
                                                    )}</span
                                                ><span>Selected alpha: {num(row.alpha)}</span><span
                                                    >Design columns: {num(row.design_columns)}</span
                                                >{#if row.monotone}<span
                                                        >Monotone constraint: {row.monotone}</span
                                                    >{/if}
                                            </div>
                                        </td></tr
                                    >{/if}
                            {/each}
                        </tbody>
                    </table>
                </div>
                {#if filtered.length > pageSize}<div class="selection-pagination">
                        <span
                            >{page * pageSize + 1}–{Math.min(
                                filtered.length,
                                (page + 1) * pageSize,
                            )} of {num(filtered.length)}</span
                        >
                        <button
                            aria-label="Previous feature results"
                            disabled={!page}
                            onclick={() => (page -= 1)}>Previous</button
                        ><button
                            aria-label="Next feature results"
                            disabled={(page + 1) * pageSize >= filtered.length}
                            onclick={() => (page += 1)}>Next</button
                        >
                    </div>{/if}
                <div class="selection-stage">
                    <button
                        onclick={() => chooseGroup('no_signal')}
                        disabled={!filtered.some((row) => row.status === 'no_signal')}
                        >Select no-signal variables</button
                    ><button
                        onclick={() => chooseGroup('signal')}
                        disabled={!filtered.some((row) => row.status === 'signal')}
                        >Select signal variables</button
                    ><button onclick={() => (selected = new Set())} disabled={!selected.size}
                        >Clear selection</button
                    >
                    <span>{selected.size} selected</span><button
                        class="primary"
                        onclick={() => stage('ignore')}
                        disabled={disabled || !selected.size || reportScope !== scope}
                        >Ignore selected</button
                    ><button
                        class="primary"
                        onclick={() => stage('predictor')}
                        disabled={disabled || !selected.size || reportScope !== scope}
                        >Make selected predictors</button
                    >
                </div>
                <p class="selection-note">
                    Selections only stage role changes. Review the Variables preview, then Apply
                    changes to update the project.
                </p>
            </div>{/if}
    </div>
</details>

<style>
    .selection-card {
        margin-top: 20px;
        border: 1px solid var(--border);
        border-radius: 7px;
        background: white;
    }
    .selection-card > summary {
        cursor: pointer;
        padding: 16px 20px;
        font-size: 16px;
        font-weight: 600;
    }
    .selection-card > summary span {
        color: var(--muted);
        font-size: 11px;
        font-weight: 400;
        margin-left: 10px;
    }
    .selection-body {
        border-top: 1px solid var(--border);
        padding: 16px 20px 20px;
    }
    .selection-body p {
        font-size: 11px;
        line-height: 1.5;
        margin: 0 0 10px;
    }
    .selection-role-note,
    .selection-method,
    .selection-note {
        color: var(--muted);
        font-size: 11px;
    }
    .selection-role-note {
        margin: 13px 0;
    }
    .selection-cost {
        color: var(--muted);
    }
    .selection-options {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 12px 0;
    }
    .selection-advanced {
        margin: 12px 0;
        font-size: 11px;
    }
    .selection-advanced > summary {
        cursor: pointer;
        color: #345b51;
    }
    .selection-options label {
        font-size: 11px;
        min-width: 0;
    }
    .selection-options input,
    .selection-options select {
        display: block;
        width: 100%;
        margin-top: 5px;
    }
    .selection-toggles,
    .selection-actions,
    .selection-controls,
    .selection-stage,
    .selection-pagination {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .selection-toggles {
        margin: 12px 0;
        font-size: 11px;
    }
    .selection-toggles label {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .selection-actions {
        margin-top: 15px;
    }
    .selection-progress {
        display: grid;
        gap: 6px;
        margin-top: 13px;
        font-size: 11px;
        color: var(--muted);
    }
    .selection-progress progress {
        width: 100%;
        height: 7px;
        accent-color: #287762;
    }
    .selection-error {
        color: #994a39;
        margin-top: 12px !important;
    }
    .selection-note {
        margin-top: 11px !important;
    }
    .selection-results {
        margin-top: 20px;
        border-top: 1px solid var(--border);
        padding-top: 15px;
    }
    .selection-summary {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: baseline;
        font-size: 11px;
        margin-bottom: 10px;
    }
    .selection-summary span,
    .selection-controls span {
        color: var(--muted);
    }
    .selection-controls {
        margin-bottom: 9px;
    }
    .selection-controls input {
        width: min(270px, 100%);
    }
    .selection-table-wrap {
        overflow-x: auto;
        max-height: 400px;
    }
    .selection-table-wrap table {
        border-collapse: collapse;
        width: 100%;
        min-width: 850px;
        font-size: 11px;
    }
    .selection-table-wrap th,
    .selection-table-wrap td {
        border-bottom: 1px solid #e9edeb;
        padding: 8px 7px;
        text-align: left;
    }
    .selection-table-wrap th {
        position: sticky;
        top: 0;
        background: #f5f7f6;
        color: var(--muted);
        line-height: 1.25;
        white-space: normal;
        z-index: 2;
    }
    .selection-table-wrap th:first-child,
    .selection-table-wrap tbody tr:not(.selection-detail) td:first-child {
        position: sticky;
        left: 0;
        width: 62px;
        min-width: 62px;
        max-width: 62px;
        background: #fff;
        z-index: 3;
    }
    .selection-table-wrap th:nth-child(2),
    .selection-table-wrap tbody tr:not(.selection-detail) td:nth-child(2) {
        position: sticky;
        left: 62px;
        width: 150px;
        min-width: 150px;
        max-width: 150px;
        background: #fff;
        box-shadow: 1px 0 #dfe6e2;
        z-index: 3;
    }
    .selection-table-wrap th:first-child,
    .selection-table-wrap th:nth-child(2) {
        background: #f5f7f6;
        z-index: 4;
    }
    .selection-table-wrap tbody tr:not(.selection-detail) td:nth-child(2) strong {
        display: block;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .selection-table-wrap td button {
        padding: 4px 7px;
        font-size: 10px;
    }
    .selection-detail {
        background: #f8faf9;
    }
    .selection-detail-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 7px 15px;
    }
    .selection-pagination {
        margin: 10px 0;
        font-size: 11px;
    }
    .selection-stage {
        border-top: 1px solid var(--border);
        padding-top: 13px;
        margin-top: 12px;
        font-size: 11px;
    }
    .selection-stage span {
        color: var(--muted);
    }
    @media (max-width: 1000px) {
        .selection-options {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 650px) {
        .selection-options {
            grid-template-columns: 1fr;
        }
        .selection-card > summary span {
            display: block;
            margin: 5px 0 0;
        }
    }
</style>
