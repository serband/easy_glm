<script>
    import { onDestroy } from 'svelte';
    import BinningHistogram from './BinningHistogram.svelte';

    export let setup;
    export let state;
    export let api;
    export let onchange;
    export let onvalidity;
    export let disabled = false;
    export let active = true;
    export let resetKey = 0;

    let search = '';
    let selected = '';
    let listScroll = 0;
    let defaultText = '';
    let pendingOverrides = {};
    let countText = {};
    let cutsText = {};
    let fallbackText = {};
    let validation = '';
    let preview = null;
    let previewError = '';
    let previewBusy = false;
    let requestId = 0;
    let previewTimer = null;
    let previousPreviewKey = '';
    let destroyed = false;

    function count(value, label) {
        const text = String(value).trim();
        if (!/^\d+$/.test(text) || Number(text) < 2 || Number(text) > 200)
            throw new Error(`${label} must be a whole number from 2 to 200.`);
        return Number(text);
    }
    function cuts(value) {
        if (value.trim() === '') return [];
        const pieces = value.split(',').map((part) => part.trim());
        if (pieces.some((part) => !part))
            throw new Error('Enter cuts separated by commas, without an empty item.');
        const numbers = pieces.map(Number);
        if (numbers.some((number) => !Number.isFinite(number)))
            throw new Error('Custom cuts must be finite numbers.');
        if (numbers.some((number, index) => index && number <= numbers[index - 1]))
            throw new Error('Custom cuts must be strictly increasing, with no duplicates.');
        return numbers;
    }
    function readBinning() {
        const overrides = {};
        for (const [name, setting] of Object.entries(pendingOverrides)) {
            if (setting.method === 'quantile')
                overrides[name] = {
                    method: 'quantile',
                    bins: count(countText[name] ?? setting.bins, `${name} bin count`),
                };
            else if (setting.method === 'cuts') {
                const points = cuts(cutsText[name] ?? setting.cuts.join(', '));
                if (!points.length && factorKind(name) === 'step' && eligibleRoles.includes(name))
                    throw new Error(`${name}: enter at least one cut for an active step factor.`);
                overrides[name] = { method: 'cuts', cuts: points };
            } else if (setting.method === 'integer') {
                const fallback =
                    fallbackText[name] ??
                    (setting.fallback_bins == null ? '' : String(setting.fallback_bins));
                overrides[name] =
                    fallback === ''
                        ? { method: 'integer' }
                        : {
                              method: 'integer',
                              fallback_bins: count(fallback, `${name} fallback bin count`),
                          };
            }
        }
        return { default_bins: count(defaultText, 'Default number of bins'), overrides };
    }
    function currentError() {
        try {
            readBinning();
            return '';
        } catch (error) {
            return error.message;
        }
    }
    function validate() {
        validation = currentError();
        onvalidity(validation);
        return !validation;
    }
    function syncValues() {
        try {
            const complete = readBinning();
            validation = '';
            onvalidity('');
            pendingOverrides = structuredClone(complete.overrides);
            onchange({ ...setup, binning: complete });
        } catch (error) {
            validation = error.message;
            onvalidity(validation);
        }
    }
    function setDefault(value) {
        defaultText = value;
        syncValues();
    }
    function chooseMethod(name, method) {
        const overrides = { ...pendingOverrides };
        if (method === 'default') delete overrides[name];
        else if (method === 'quantile') {
            overrides[name] = { method, bins: setup.binning.default_bins };
            if (countText[name] == null)
                countText = { ...countText, [name]: String(setup.binning.default_bins) };
        } else if (method === 'cuts') {
            overrides[name] = { method, cuts: [0] };
            if (cutsText[name] == null) cutsText = { ...cutsText, [name]: '0' };
        } else overrides[name] = { method: 'integer' };
        pendingOverrides = overrides;
        syncValues();
    }
    function setCount(name, value) {
        countText = { ...countText, [name]: value };
        syncValues();
    }
    function setCuts(name, value) {
        cutsText = { ...cutsText, [name]: value };
        syncValues();
    }
    function setFallback(name, value) {
        fallbackText = { ...fallbackText, [name]: value };
        syncValues();
    }
    function initialise() {
        defaultText = String(setup.binning?.default_bins ?? 20);
        pendingOverrides = structuredClone(setup.binning?.overrides || {});
        countText = Object.fromEntries(
            Object.entries(setup.binning?.overrides || {})
                .filter(([, setting]) => setting.method === 'quantile')
                .map(([name, setting]) => [name, String(setting.bins)]),
        );
        cutsText = Object.fromEntries(
            Object.entries(setup.binning?.overrides || {})
                .filter(([, setting]) => setting.method === 'cuts')
                .map(([name, setting]) => [name, setting.cuts.join(', ')]),
        );
        fallbackText = Object.fromEntries(
            Object.entries(setup.binning?.overrides || {})
                .filter(([, setting]) => setting.method === 'integer')
                .map(([name, setting]) => [
                    name,
                    setting.fallback_bins == null ? '' : String(setting.fallback_bins),
                ]),
        );
        validate();
    }
    function invalidate() {
        requestId++;
        if (previewTimer) clearTimeout(previewTimer);
        previewTimer = null;
        preview = null;
        previewError = '';
        previewBusy = false;
    }
    function schedulePreview() {
        invalidate();
        if (
            destroyed ||
            !active ||
            disabled ||
            !selected ||
            !names.includes(selected) ||
            !state?.session_id ||
            !validate()
        )
            return;
        const id = requestId;
        previewBusy = true;
        previewTimer = setTimeout(() => {
            previewTimer = null;
            void loadPreview(id);
        }, 300);
    }
    async function loadPreview(id) {
        if (destroyed || id !== requestId || !active || disabled || !names.includes(selected))
            return;
        const column = selected;
        const requestSetup = structuredClone(setup);
        try {
            const result = await api('variables/binning-preview', {
                session_id: state.session_id,
                revision: state.revision,
                setup: requestSetup,
                column,
            });
            if (!destroyed && id === requestId) preview = result;
        } catch (error) {
            if (!destroyed && id === requestId) previewError = error.message;
        } finally {
            if (!destroyed && id === requestId) previewBusy = false;
        }
    }
    function summary(name) {
        const setting = pendingOverrides[name];
        if (!setting) return `Default · ${setup.binning?.default_bins ?? 20}`;
        if (setting.method === 'quantile') return `Automatic · ${setting.bins}`;
        if (setting.method === 'cuts') return `Custom · ${setting.cuts.length} cuts`;
        return setting.fallback_bins
            ? `Integer · fallback ${setting.fallback_bins}`
            : 'Integer cuts';
    }
    function numericColumn(column) {
        return /^(?:u?int\d*|float\d*|decimal)/i.test(column.dtype || '');
    }
    function factorKind(name) {
        if ((setup.types.categorical || []).includes(name)) return 'categorical';
        const saved = state?.binning_columns?.[name]?.kind;
        if (saved) return saved;
        const column = state?.columns.find((item) => item.name === name);
        return (setup.types.numeric || []).includes(name) || (column && numericColumn(column))
            ? 'step'
            : 'categorical';
    }
    function activeSetting(name) {
        return (
            numericNames.includes(name) && !['continuous', 'categorical'].includes(factorKind(name))
        );
    }
    function clampLabel(name) {
        const clamp = state?.binning_columns?.[name]?.clamp;
        return Array.isArray(clamp) && clamp.length === 2 ? `${clamp[0]} to ${clamp[1]}` : '';
    }
    $: eligibleRoles = [...(setup.roles.predictor || []), ...(setup.roles.unassigned || [])];
    $: numericNames =
        state?.columns
            .filter(
                (column) =>
                    eligibleRoles.includes(column.name) &&
                    (setup.types.categorical || []).includes(column.name) === false &&
                    (numericColumn(column) || (setup.types.numeric || []).includes(column.name)),
            )
            .map((column) => column.name) || [];
    $: names = [...new Set([...numericNames, ...Object.keys(pendingOverrides)])];
    $: filtered = names.filter((name) => name.toLowerCase().includes(search.toLowerCase()));
    $: first = Math.max(0, Math.floor(listScroll / 35) - 3);
    $: if (!names.includes(selected)) selected = names[0] || '';
    $: previewKey = JSON.stringify([
        state?.session_id,
        state?.project_id,
        state?.revision,
        setup,
        selected,
        active,
        disabled,
        resetKey,
        defaultText,
        pendingOverrides,
        countText,
        cutsText,
        fallbackText,
    ]);
    $: if (previewKey !== previousPreviewKey) {
        previousPreviewKey = previewKey;
        schedulePreview();
    }
    $: if (resetKey !== undefined) {
        resetKey;
        initialise();
    }
    onDestroy(() => {
        destroyed = true;
        invalidate();
    });
</script>

<section class="model-card binning-card" aria-label="Numeric binning">
    <div class="binning-heading">
        <div>
            <h2>Numeric binning</h2>
            <p class="help-text">
                Shared by every model in this project. Applying changes requires fitted models to be
                refitted.
            </p>
        </div>
    </div>
    <label class="default-label"
        >Default number of bins
        <input
            aria-label="Default number of bins"
            type="text"
            inputmode="numeric"
            value={defaultText}
            {disabled}
            oninput={(event) => setDefault(event.currentTarget.value)}
        />
    </label>
    <p class="help-text">
        Automatic cuts use training rows. Tied values can produce fewer bins than requested.
    </p>
    {#if names.length}
        <div class="binning-layout">
            <div class="binning-list-wrap">
                <input
                    aria-label="Search numeric predictors"
                    placeholder="Find a numeric predictor…"
                    bind:value={search}
                    oninput={() => (listScroll = 0)}
                />
                <div
                    class="binning-list"
                    onscroll={(event) => (listScroll = event.currentTarget.scrollTop)}
                >
                    <div style:height={filtered.length * 35 + 'px'} class="binning-space">
                        <div
                            class="binning-rows"
                            style:transform={'translateY(' + first * 35 + 'px)'}
                        >
                            {#each filtered.slice(first, first + 14) as name}
                                <button
                                    type="button"
                                    class:selected={selected === name}
                                    onclick={() => {
                                        selected = name;
                                    }}
                                    title={name}
                                >
                                    <span>{setup.renames[name] || name}</span><small
                                        >{summary(name)}</small
                                    >
                                </button>
                            {/each}
                        </div>
                    </div>
                </div>
                <p class="help-text">
                    {filtered.length} of {names.length} numeric variables and saved settings
                </p>
            </div>
            {#if selected}<div class="binning-editor">
                    <h3>{setup.renames[selected] || selected}</h3>
                    {#if !activeSetting(selected)}<p class="help-text">
                            Binning is inactive for this {factorKind(selected)} factor or its current
                            role. Saved settings remain available if the factor becomes numeric and banded
                            again.
                        </p>{/if}
                    {#if factorKind(selected) === 'linear'}<p class="help-text">
                            For a piecewise-linear factor, custom cuts are slope-change points, not
                            flat rating bands.{#if clampLabel(selected)}
                                Current clamp: {clampLabel(selected)}.{/if}
                        </p>{/if}
                    <label
                        >Method<select
                            aria-label={'Binning method for ' + selected}
                            value={pendingOverrides[selected]?.method || 'default'}
                            {disabled}
                            onchange={(event) => chooseMethod(selected, event.currentTarget.value)}
                        >
                            <option value="default">Use default</option><option value="quantile"
                                >Automatic — own number of bins</option
                            ><option value="cuts">Custom cuts</option><option value="integer"
                                >Integer cuts</option
                            >
                        </select></label
                    >
                    {#if pendingOverrides[selected]?.method === 'quantile'}
                        <label
                            >Number of bins<input
                                aria-label={'Bins for ' + selected}
                                type="text"
                                inputmode="numeric"
                                value={countText[selected] ?? ''}
                                {disabled}
                                oninput={(event) => setCount(selected, event.currentTarget.value)}
                            /></label
                        >
                    {:else if pendingOverrides[selected]?.method === 'cuts'}
                        <label
                            >Cut points<input
                                aria-label={'Custom cuts for ' + selected}
                                type="text"
                                value={cutsText[selected] ?? ''}
                                placeholder="0, 1, 2, 3"
                                {disabled}
                                oninput={(event) => setCuts(selected, event.currentTarget.value)}
                            /></label
                        >
                        <p class="help-text">
                            Enter strictly increasing numbers.{#if factorKind(selected) === 'linear'}
                                Leave blank for one slope.{:else if !activeSetting(selected)}
                                Leave blank to preserve an inactive setting.{:else}
                                An active step factor needs at least one cut.{/if} A value on a cut enters
                            the interval on its right.
                        </p>
                    {:else if pendingOverrides[selected]?.method === 'integer'}
                        <label
                            >Fallback number of bins (optional)<input
                                aria-label={'Integer fallback for ' + selected}
                                type="text"
                                inputmode="numeric"
                                value={fallbackText[selected] ?? ''}
                                {disabled}
                                oninput={(event) =>
                                    setFallback(selected, event.currentTarget.value)}
                            /></label
                        >
                    {/if}
                </div>{/if}
        </div>
    {:else}<p class="help-text">No numeric predictor or unassigned variable is available.</p>{/if}
    {#if validation}<p role="alert" class="binning-error">{validation}</p>{/if}
    {#if previewBusy}<p class="help-text">Calculating intervals from training rows…</p>{/if}
    {#if previewError}<p role="alert" class="binning-error">{previewError}</p>{/if}
    {#if preview}
        <div class="binning-preview">
            <h3>Training preview · {preview.name || preview.column}</h3>
            <p>
                {preview.requested_bins == null
                    ? ''
                    : `${preview.requested_bins} requested · `}{preview.actual_bins} actual bins · {preview.training_rows?.toLocaleString()}
                training rows · {preview.missing_rows?.toLocaleString()} missing{preview.nonfinite_rows
                    ? ` · ${preview.nonfinite_rows.toLocaleString()} non-finite`
                    : ''}
            </p>
            {#if !preview.active}<p>Saved numeric setting is inactive for this factor kind.</p>{/if}
            {#each preview.warnings || [] as warning}<p class="binning-warning">{warning}</p>{/each}
            {#if preview.active && preview.rows?.length}<BinningHistogram
                    intervals={preview.rows || []}
                    name={preview.name || preview.column}
                />{/if}
            <details class="binning-intervals">
                <summary>Bin counts and intervals</summary>
                <div class="binning-preview-scroll">
                    <table>
                        <thead><tr><th>Interval</th><th>Rows</th><th>Exposure</th></tr></thead
                        ><tbody>
                            {#each preview.rows || [] as row}<tr
                                    ><td>{row.label}</td><td>{row.rows?.toLocaleString()}</td><td
                                        >{row.exposure == null
                                            ? '—'
                                            : row.exposure.toLocaleString()}</td
                                    ></tr
                                >{/each}
                        </tbody>
                    </table>
                </div>
            </details>
        </div>
    {/if}
</section>

<style>
    .binning-card {
        margin-top: 16px;
        padding: 18px;
    }
    .binning-heading {
        display: flex;
        justify-content: space-between;
        gap: 15px;
    }
    .binning-card h2 {
        margin: 0 0 5px;
    }
    .binning-card h3 {
        font-size: 13px;
        margin: 0 0 12px;
    }
    .binning-card p {
        line-height: 1.4;
    }
    .default-label {
        display: block;
        width: min(240px, 100%);
        margin: 18px 0 7px;
    }
    .binning-card label {
        font-size: 12px;
    }
    .binning-card input,
    .binning-card select {
        display: block;
        width: 100%;
        margin-top: 5px;
    }
    .binning-layout {
        display: grid;
        grid-template-columns: minmax(210px, 1fr) minmax(260px, 1.4fr);
        gap: 18px;
        margin-top: 18px;
        border-top: 1px solid #e5e8e7;
        padding-top: 16px;
    }
    .binning-list {
        height: 220px;
        overflow: auto;
        border: 1px solid #e1e5e3;
        margin: 8px 0 5px;
    }
    .binning-space {
        position: relative;
    }
    .binning-rows {
        position: absolute;
        inset: 0 0 auto;
    }
    .binning-rows button {
        width: 100%;
        height: 35px;
        padding: 4px 9px;
        border: 0;
        border-bottom: 1px solid #eff1ef;
        border-radius: 0;
        display: flex;
        justify-content: space-between;
        gap: 8px;
        text-align: left;
    }
    .binning-rows button.selected {
        background: #e8f2ee;
        color: #195f4d;
    }
    .binning-rows span {
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .binning-rows small {
        color: #63716a;
        font-size: 10px;
    }
    .binning-editor {
        min-width: 0;
    }
    .binning-editor label {
        display: block;
        margin-bottom: 12px;
    }
    .binning-preview {
        margin-top: 18px;
        border-top: 1px solid #e5e8e7;
        padding-top: 15px;
    }
    .binning-preview p {
        margin: 5px 0;
    }
    .binning-intervals {
        margin-top: 13px;
    }
    .binning-intervals summary {
        cursor: pointer;
        color: #284c69;
        font-size: 12px;
        font-weight: 600;
    }
    .binning-preview-scroll {
        overflow: auto;
        max-height: 300px;
        margin-top: 10px;
    }
    .binning-preview table {
        width: 100%;
        border-collapse: collapse;
        text-align: left;
        font-size: 11px;
    }
    .binning-preview th,
    .binning-preview td {
        border-bottom: 1px solid #e8ebe9;
        padding: 7px;
    }
    .binning-error {
        color: #aa392f;
        margin-top: 10px;
    }
    .binning-warning {
        color: #92601f;
    }
    @media (max-width: 799px) {
        .binning-layout {
            grid-template-columns: 1fr;
        }
    }
</style>
