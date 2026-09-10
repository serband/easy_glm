<script>
    import { onDestroy, tick } from 'svelte';
    import { formatNumber as num } from './format.js';

    export let api, state, setup, context, prepare, onRemove;
    export let disabled = false;
    let sampleRows = 10000,
        missingPercent = 70,
        correlationThreshold = 0.95,
        leakageThreshold = 0.9,
        divideTarget = true;
    let job = null,
        result = null,
        pending = false,
        starting = false,
        error = '',
        note = '',
        query = '',
        selected = new Set(),
        pages = { leakage: 0, correlated: 0, missing: 0 };
    let sequence = 0,
        timer = null,
        alive = true,
        seenScope = '',
        reportScope = '';
    const pageSize = 8;
    $: options = {
        sample_rows: sampleRows,
        seed: 42,
        missing_threshold: missingPercent / 100,
        correlation_threshold: correlationThreshold,
        leakage_threshold: leakageThreshold,
        divide_target_by_weight: divideTarget,
    };
    $: scope = JSON.stringify([context, options]);
    $: if (scope !== seenScope) invalidate(scope);
    $: predictors = new Set(setup?.roles.predictor || []);
    $: weight = setup?.assignments.weight;
    $: validOptions =
        Number.isInteger(sampleRows) &&
        sampleRows >= 500 &&
        sampleRows <= 20000 &&
        Number.isFinite(missingPercent) &&
        missingPercent >= 0 &&
        missingPercent <= 100 &&
        Number.isFinite(correlationThreshold) &&
        correlationThreshold >= 0 &&
        correlationThreshold <= 1 &&
        Number.isFinite(leakageThreshold) &&
        leakageThreshold >= 0 &&
        leakageThreshold <= 1;
    $: filtered = {
        leakage: (result?.leakage || []).filter((row) => matches(row, query)),
        correlated: (result?.correlated || []).filter((row) => matches(row, query)),
        missing: (result?.missing || []).filter((row) => matches(row, query)),
    };
    $: conflicts = (result?.correlated || []).filter(
        (row) => selected.has(row.first_raw) && selected.has(row.second_raw),
    ).length;

    function currentScope() {
        return JSON.stringify([
            context,
            {
                sample_rows: sampleRows,
                seed: 42,
                missing_threshold: missingPercent / 100,
                correlation_threshold: correlationThreshold,
                leakage_threshold: leakageThreshold,
                divide_target_by_weight: divideTarget,
            },
        ]);
    }
    function matches(row, search) {
        const wanted = search.trim().toLowerCase();
        return (
            !wanted ||
            [row.variable, row.raw_name, row.first, row.second, row.first_raw, row.second_raw].some(
                (value) => value?.toLowerCase().includes(wanted),
            )
        );
    }
    function canRemove(raw) {
        return typeof raw === 'string' && predictors.has(raw);
    }
    function clearTimer() {
        if (timer) clearTimeout(timer);
        timer = null;
    }
    function cancelJob(packet) {
        if (packet && ['queued', 'running'].includes(packet.status))
            void api('screenings/' + encodeURIComponent(packet.id) + '/cancel', {
                session_id: state.session_id,
                revision: state.revision,
            }).catch(() => {});
    }
    function invalidate(nextScope, message = 'Settings changed. Check predictors again.') {
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
        pages = { leakage: 0, correlated: 0, missing: 0 };
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
            invalidate(currentScope(), 'Settings changed. Check predictors again.');
            return;
        }
        job = packet;
        pending = ['queued', 'running'].includes(packet.status);
        if (packet.status === 'complete') {
            if (!packet.result) {
                error = 'The scan returned no results. Check predictors again.';
                return;
            }
            result = packet.result;
            reportScope = requestedScope;
            note = '';
        } else if (packet.status === 'failed') error = packet.message || 'The scan failed.';
        else if (packet.status === 'stale') note = 'Settings changed. Check predictors again.';
        else if (packet.status === 'cancelled') note = 'Scan cancelled.';
        else if (pending) timer = setTimeout(() => poll(packet, request, requestedScope), 500);
        else error = 'The scan returned an unknown status. Check predictors again.';
    }
    async function poll(packet, request, requestedScope) {
        if (!isCurrent(request, requestedScope)) return;
        try {
            accept(
                await api('screenings/' + encodeURIComponent(packet.id)),
                request,
                requestedScope,
            );
        } catch (e) {
            if (isCurrent(request, requestedScope)) {
                pending = false;
                error = e.message;
            }
        }
    }
    async function check() {
        if (disabled || pending || starting || !validOptions) return;
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
                const packet = await api('variables/screen', {
                    session_id: state.session_id,
                    revision: state.revision,
                    setup: structuredClone(setup),
                    options: structuredClone(options),
                });
                accept(packet, request, requestedScope);
            } catch (e) {
                if (isCurrent(request, requestedScope)) {
                    pending = false;
                    error = e.message;
                }
            }
        } finally {
            starting = false;
        }
    }
    function select(raw, checked) {
        if (!canRemove(raw)) return;
        const next = new Set(selected);
        if (checked) next.add(raw);
        else next.delete(raw);
        selected = next;
    }
    function selectGroup(rows, checked) {
        const next = new Set(selected);
        for (const row of rows)
            if (canRemove(row.raw_name)) {
                if (checked) next.add(row.raw_name);
                else next.delete(row.raw_name);
            }
        selected = next;
    }
    function pairChoice(row, choices) {
        if (choices.has(row.first_raw) && choices.has(row.second_raw)) return '__both__';
        if (choices.has(row.first_raw)) return 'first';
        if (choices.has(row.second_raw)) return 'second';
        return '';
    }
    function selectPair(row, choice) {
        const next = new Set(selected);
        next.delete(row.first_raw);
        next.delete(row.second_raw);
        const raw =
            choice === 'first' ? row.first_raw : choice === 'second' ? row.second_raw : null;
        if (canRemove(raw)) next.add(raw);
        selected = next;
    }
    async function remove() {
        if (disabled || !selected.size || reportScope !== currentScope()) return;
        try {
            await onRemove([...selected], context);
        } catch (e) {
            error = e.message;
        }
    }
    onDestroy(() => {
        alive = false;
        sequence += 1;
        clearTimer();
        cancelJob(job);
    });
</script>

<section class="screening-card" aria-label="Predictor screening">
    <div class="screening-heading">
        <div>
            <h2>Check predictors</h2>
            <p>Possible leakage, related predictors and missing data.</p>
        </div>
        <div class="screening-actions">
            <button
                class="primary"
                onclick={check}
                disabled={disabled || pending || starting || !validOptions}
                >Check selected predictors</button
            >
            {#if pending}<button onclick={() => invalidate(currentScope(), 'Scan cancelled.')}
                    >Cancel scan</button
                >{/if}
        </div>
    </div>
    <details class="screening-settings">
        <summary
            >Settings · missing {num(missingPercent)}% · correlation {num(correlationThreshold)} · target
            association {num(leakageThreshold)}</summary
        >
        <div class="screening-options">
            <label
                >Missing threshold (%)<input
                    aria-label="Screening missing threshold"
                    type="number"
                    min="0"
                    max="100"
                    step="1"
                    bind:value={missingPercent}
                /></label
            >
            <label
                >Pair association threshold<input
                    aria-label="Screening correlation threshold"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={correlationThreshold}
                /></label
            >
            <label
                >Target association threshold<input
                    aria-label="Screening leakage threshold"
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    bind:value={leakageThreshold}
                /></label
            >
            <label
                >Training sample rows<input
                    aria-label="Screening sample rows"
                    type="number"
                    min="500"
                    max="20000"
                    step="500"
                    bind:value={sampleRows}
                /></label
            >
        </div>
        {#if weight}<label class="screening-toggle"
                ><input
                    type="checkbox"
                    aria-label="Divide target by weight for screening"
                    bind:checked={divideTarget}
                />Divide target by weight for screening</label
            >{/if}
        {#if !validOptions}<p class="screening-error">
                Use 500–20,000 rows, a missing percentage from 0–100, and association thresholds
                from 0–1.
            </p>{/if}
    </details>
    {#if pending}<div class="scan-progress" role="status">
            <span>{job?.progress?.message || job?.message || 'Starting scan…'}</span>
            <progress
                aria-label="Screening progress"
                max={job?.progress?.total || 1}
                value={job?.progress?.total ? job.progress.completed : undefined}
            ></progress>
        </div>{/if}
    {#if error}<p class="screening-error" role="alert">{error}</p>{/if}
    {#if note}<p class="screening-note">{note}</p>{/if}
    {#if result}
        <div class="screening-summary">
            <span
                >{num(result.rows)} of {num(result.training_rows)} training rows · {num(
                    result.predictor_count,
                )} predictors</span
            >
            {#if result.target}<span
                    >Target: {result.target}{result.divide_target_by_weight && result.weight
                        ? ' / ' + result.weight
                        : ''}{result.weight ? ' · Weight: ' + result.weight : ''}</span
                >{/if}
        </div>
        <p class="screening-caution">
            Flags need review. Strong association alone does not prove leakage.
        </p>
        <input
            class="screening-search"
            aria-label="Search screening results"
            placeholder="Find a flagged predictor…"
            bind:value={query}
            oninput={() => (pages = { leakage: 0, correlated: 0, missing: 0 })}
        />
        {#each [{ key: 'leakage', title: 'Possible target leakage' }, { key: 'correlated', title: 'Highly related predictors' }, { key: 'missing', title: 'Mostly missing' }] as group}
            {@const rows = filtered[group.key]}
            <section class="screening-group" aria-label={group.title}>
                <div class="screening-group-heading">
                    <h3>
                        {group.title}
                        <span
                            >{num(
                                group.key === 'correlated'
                                    ? (result.correlated_count ?? result.correlated?.length ?? 0)
                                    : result[group.key]?.length || 0,
                            )}</span
                        >
                    </h3>
                    {#if group.key !== 'correlated' && rows.some((row) => canRemove(row.raw_name))}
                        <button
                            aria-label={'Select all ' + group.title.toLowerCase()}
                            onclick={() => selectGroup(rows, true)}
                            >Select all{query ? ' matches' : ''}</button
                        >
                        <button
                            aria-label={'Clear ' + group.title.toLowerCase()}
                            onclick={() => selectGroup(rows, false)}>Clear</button
                        >
                    {/if}
                </div>
                {#if group.key === 'correlated' && result.correlated_count > result.correlated.length}<p
                        class="screening-caution"
                    >
                        Showing the strongest {num(result.correlated.length)} of {num(
                            result.correlated_count,
                        )} flagged pairs.
                    </p>{/if}
                {#if !rows.length}<p class="screening-empty">
                        {query ? 'No matching predictors.' : 'None flagged.'}
                    </p>
                {:else}
                    <div class="screening-rows">
                        {#each rows.slice(pages[group.key] * pageSize, (pages[group.key] + 1) * pageSize) as row}
                            {#if group.key === 'correlated'}
                                <div class="screening-row screening-pair">
                                    <div class="screening-variable">
                                        <strong title={row.first + ' · ' + row.second}
                                            >{row.first} <span>↔</span> {row.second}</strong
                                        ><span>{row.method} · {num(row.observations)} rows</span>
                                    </div>
                                    <b class="screening-value">{num(row.association)}</b>
                                    <select
                                        aria-label={'Removal choice for ' +
                                            row.first +
                                            ' and ' +
                                            row.second}
                                        value={pairChoice(row, selected)}
                                        onchange={(e) => selectPair(row, e.currentTarget.value)}
                                    >
                                        <option value="">Keep both</option>
                                        {#if pairChoice(row, selected) === '__both__'}<option
                                                value="__both__"
                                                disabled>Both selected</option
                                            >{/if}
                                        <option value="first" disabled={!canRemove(row.first_raw)}
                                            >Remove {row.first}</option
                                        >
                                        <option value="second" disabled={!canRemove(row.second_raw)}
                                            >Remove {row.second}</option
                                        >
                                    </select>
                                </div>
                            {:else}
                                <label class="screening-row">
                                    <input
                                        type="checkbox"
                                        aria-label={'Select ' +
                                            row.variable +
                                            ' in ' +
                                            group.title.toLowerCase()}
                                        checked={selected.has(row.raw_name)}
                                        disabled={!canRemove(row.raw_name)}
                                        onchange={(e) =>
                                            select(row.raw_name, e.currentTarget.checked)}
                                    />
                                    <span class="screening-variable"
                                        ><strong title={row.variable}>{row.variable}</strong><span
                                            >{group.key === 'leakage'
                                                ? row.reason + ' · ' + row.method
                                                : num(row.observations) + ' observed rows'}</span
                                        ></span
                                    >
                                    <b class="screening-value"
                                        >{group.key === 'missing'
                                            ? num(row.missing_share * 100) + '%'
                                            : num(row.association)}</b
                                    >
                                </label>
                            {/if}
                        {/each}
                    </div>
                    {#if rows.length > pageSize}<div class="screening-pagination">
                            <span
                                >{pages[group.key] * pageSize + 1}–{Math.min(
                                    rows.length,
                                    (pages[group.key] + 1) * pageSize,
                                )} of {num(rows.length)}</span
                            ><button
                                aria-label={'Previous ' + group.title.toLowerCase()}
                                disabled={!pages[group.key]}
                                onclick={() =>
                                    (pages = { ...pages, [group.key]: pages[group.key] - 1 })}
                                >Previous</button
                            ><button
                                aria-label={'Next ' + group.title.toLowerCase()}
                                disabled={(pages[group.key] + 1) * pageSize >= rows.length}
                                onclick={() =>
                                    (pages = { ...pages, [group.key]: pages[group.key] + 1 })}
                                >Next</button
                            >
                        </div>{/if}
                {/if}
            </section>
        {/each}
        {#if result.unsupported?.length || result.notes?.length || result.excluded_target_rows}
            <details class="screening-notes">
                <summary
                    >Coverage and limitations{result.unsupported?.length
                        ? ' · ' +
                          num(result.unsupported.length) +
                          (result.unsupported.length === 1 ? ' predictor' : ' predictors') +
                          ' not fully checked'
                        : ''}</summary
                >
                {#if result.excluded_target_rows}<p>
                        {num(result.excluded_target_rows)} rows excluded from target association.
                    </p>{/if}
                {#each result.notes || [] as item}<p>{item}</p>{/each}
                {#if result.unsupported?.length}<div class="screening-unsupported">
                        {#each result.unsupported as item}<p>
                                <strong>{item.variable}:</strong>
                                {item.reason}
                            </p>{/each}
                    </div>{/if}
            </details>
        {/if}
        <div class="screening-removal">
            <button
                class="primary"
                onclick={remove}
                disabled={disabled || !selected.size || reportScope !== scope}
                >Remove selected predictors ({selected.size})</button
            >
            <button disabled={!selected.size} onclick={() => (selected = new Set())}
                >Clear selection</button
            >
            <span>Review changes before applying.</span>
        </div>
        {#if conflicts}<p class="screening-note">
                Both predictors are selected in {num(conflicts)} related {conflicts === 1
                    ? 'pair'
                    : 'pairs'}. Confirm the removals in the preview.
            </p>{/if}
    {/if}
</section>

<style>
    .screening-card {
        margin-top: 20px;
        padding: 18px 20px;
        border: 1px solid var(--border);
        border-radius: 7px;
        background: white;
    }
    .screening-heading,
    .screening-group-heading,
    .screening-actions,
    .screening-removal,
    .screening-pagination {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
    }
    .screening-heading {
        justify-content: space-between;
        gap: 12px;
    }
    h2 {
        margin: 0 0 4px;
    }
    h3 {
        margin: 0;
        font-size: 13px;
    }
    h3 span {
        margin-left: 6px;
        color: var(--muted);
        font-weight: 400;
    }
    p,
    .screening-note,
    .screening-summary,
    .screening-settings,
    .screening-removal span {
        font-size: 11px;
        line-height: 1.5;
    }
    .screening-settings {
        margin-top: 12px;
    }
    summary {
        cursor: pointer;
        color: var(--muted);
    }
    .screening-options {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 10px;
        margin: 12px 0;
    }
    .screening-options label {
        display: grid;
        gap: 5px;
        color: var(--muted);
    }
    .screening-options input {
        width: 100%;
    }
    .screening-toggle {
        display: flex;
        gap: 7px;
        align-items: center;
    }
    input[type='checkbox'] {
        accent-color: #287762;
        width: 13px;
        height: 13px;
        margin: 0;
        flex-shrink: 0;
    }
    .screening-summary {
        display: flex;
        flex-direction: column;
        gap: 3px;
        margin: 16px 0 5px;
        color: var(--muted);
    }
    .screening-caution {
        margin-bottom: 12px;
    }
    .screening-search {
        width: 100%;
    }
    .screening-group {
        border-top: 1px solid var(--border);
        margin-top: 15px;
        padding-top: 12px;
    }
    .screening-group-heading {
        margin-bottom: 7px;
    }
    .screening-group-heading h3 {
        flex: 1;
    }
    .screening-group-heading button,
    .screening-pagination button {
        padding: 4px 8px;
        font-size: 11px;
    }
    .screening-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 9px 0;
        border-bottom: 1px solid #eff0f2;
        min-width: 0;
    }
    .screening-variable {
        display: grid;
        gap: 3px;
        min-width: 0;
        flex: 1;
        font-size: 12px;
    }
    .screening-variable strong {
        overflow-wrap: anywhere;
    }
    .screening-variable span {
        color: var(--muted);
        font-size: 10px;
    }
    .screening-value {
        font-size: 12px;
        font-variant-numeric: tabular-nums;
        flex-shrink: 0;
    }
    .screening-pair select {
        width: 190px;
        max-width: 40%;
        font-size: 11px;
    }
    .screening-empty {
        padding: 5px 0;
    }
    .screening-pagination {
        justify-content: flex-end;
        margin-top: 8px;
        font-size: 10px;
        color: var(--muted);
    }
    .screening-pagination span {
        margin-right: auto;
    }
    .screening-notes {
        margin-top: 12px;
        font-size: 11px;
    }
    .screening-notes p {
        margin-top: 5px;
    }
    .screening-unsupported {
        max-height: 140px;
        overflow: auto;
        margin-top: 6px;
    }
    .screening-removal {
        border-top: 1px solid var(--border);
        margin-top: 14px;
        padding-top: 12px;
    }
    .screening-removal span {
        color: var(--muted);
    }
    .screening-error {
        color: #994a39;
        margin-top: 10px;
    }
    .screening-note {
        margin-top: 10px;
        color: var(--muted);
    }
    .scan-progress {
        display: grid;
        gap: 7px;
        margin-top: 12px;
        font-size: 11px;
        color: var(--muted);
    }
    progress {
        width: 100%;
        height: 6px;
        accent-color: #287762;
    }
    @media (max-width: 1000px) {
        .screening-options {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }
    @media (max-width: 650px) {
        .screening-pair {
            flex-wrap: wrap;
        }
        .screening-pair select {
            max-width: 100%;
            width: 100%;
        }
    }
</style>
