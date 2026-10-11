<script>
    import {
        completeVariableDesign,
        copy,
        modelVariablesJson,
        normaliseDefaults,
        parseModelVariablesJson,
    } from './modelVariables.js';

    export let columns = [];
    export let factors = [];
    export let design = null;
    export let inheritedDesign = null;
    export let method = 'glm';
    export let selectionOnly = false;
    export let treatmentOnly = false;
    export let showHeading = true;
    export let onChange = () => {};
    export let onValidity = () => {};
    export let onDraftText = () => {};

    let tab = 'table';
    let search = '';
    let selectedOnly = false;
    let jsonText = '';
    let jsonError = '';
    let lastModelKey = '';
    let inputError = '';
    let cutsText = {};
    let bandsText = {};
    let customModes = {};
    let breaksText = {};
    let aglmCustomModes = {};

    $: eligible = [...columns]
        .filter((column) => column?.name)
        .sort((left, right) => left.name.localeCompare(right.name));
    $: eligibleNames = eligible.map((column) => column.name);
    $: columnByName = new Map(eligible.map((column) => [column.name, column]));
    $: chosen = new Set(factors);
    $: displayed = eligible.filter((column) =>
        treatmentOnly || selectedOnly
            ? chosen.has(column.name)
            : column.name.toLowerCase().includes(search.toLowerCase()),
    );
    $: modelKey = JSON.stringify({ factors: [...factors].sort(), design, inheritedDesign, method });
    $: if (modelKey !== lastModelKey) {
        jsonText = serialise();
        lastModelKey = modelKey;
        jsonError = '';
        onDraftText('');
        reportValidity();
    }

    function effectiveDesign(name) {
        return completeVariableDesign(name, inheritedDesign?.variables, design?.variables);
    }

    function serialise() {
        if (selectionOnly) return JSON.stringify([...factors].sort(), null, 2);
        const modelDesign =
            method === 'catboost'
                ? catBoostDesign()
                : design || {
                      defaults: normaliseDefaults(inheritedDesign?.defaults),
                      variables: {},
                  };
        return modelVariablesJson({
            factors,
            design: modelDesign,
            inheritedDesign:
                method === 'catboost'
                    ? { defaults: inheritedDesign?.defaults, variables: {} }
                    : inheritedDesign,
        });
    }

    function catBoostDesign() {
        const variables = {};
        for (const name of factors) {
            const current = effectiveDesign(name);
            if (current.kind === 'categorical') variables[name] = { kind: 'categorical' };
            else variables[name] = {};
        }
        return { defaults: normaliseDefaults(inheritedDesign?.defaults), variables };
    }

    function emit(nextFactors, nextDesign) {
        const ordered = [...new Set(nextFactors)].sort((a, b) => a.localeCompare(b));
        const selected = new Set(ordered);
        const baseVariables = method === 'catboost' ? {} : inheritedDesign?.variables;
        const variables = Object.fromEntries(
            ordered.map((name) => [
                name,
                completeVariableDesign(name, baseVariables, nextDesign?.variables),
            ]),
        );
        const complete = {
            defaults: normaliseDefaults({ ...inheritedDesign?.defaults, ...nextDesign?.defaults }),
            variables,
        };
        onChange({ factors: ordered, design: complete });
    }

    function toggle(name) {
        const next = chosen.has(name)
            ? factors.filter((factor) => factor !== name)
            : [...factors, name];
        if (!next.includes(name) && inputError.startsWith(`${name}:`)) {
            inputError = '';
            reportValidity();
        }
        emit(next, design || {});
    }

    function updateDefaults(key, value) {
        const current = design || {
            defaults: normaliseDefaults(inheritedDesign?.defaults),
            variables: {},
        };
        const number = Number(value);
        if (!Number.isInteger(number) || number < 2 || number > 200) return;
        emit(factors, { ...current, defaults: { ...current.defaults, [key]: number } });
    }

    function updateVariable(name, patch) {
        const current = design || {
            defaults: normaliseDefaults(inheritedDesign?.defaults),
            variables: {},
        };
        emit(factors, {
            ...current,
            variables: { ...current.variables, [name]: { ...effectiveDesign(name), ...patch } },
        });
    }

    function replaceVariable(name, nextVariable) {
        const current = design || {
            defaults: normaliseDefaults(inheritedDesign?.defaults),
            variables: {},
        };
        emit(factors, {
            ...current,
            variables: { ...current.variables, [name]: nextVariable },
        });
    }

    function isAglm(kind) {
        return kind === 'aglm' || kind === 'aglm_l';
    }

    function withoutAglmSettings(current = {}) {
        return { ...current, breaks: null, n_breaks: null, extrapolation: null };
    }

    function changeKind(name, value) {
        if (method === 'catboost') {
            replaceVariable(name, value === 'categorical' ? { kind: 'categorical' } : { kind: null });
            return;
        }
        const current = effectiveDesign(name);
        if (isAglm(value)) {
            aglmCustomModes = { ...aglmCustomModes, [name]: false };
            replaceVariable(name, {
                ...withoutAglmSettings(current),
                kind: value,
                knots: 'quantile',
                n_bins: null,
                clamp: null,
                monotone: null,
                breaks: null,
                n_breaks: 100,
                extrapolation: 'extend',
            });
            return;
        }
        replaceVariable(name, withoutAglmSettings({ ...current, kind: value }));
    }

    function changeCuts(name, value) {
        cutsText = { ...cutsText, [name]: value };
        const cuts =
            value.trim() === '' ? [] : value.split(',').map((piece) => Number(piece.trim()));
        if (
            !cuts.length ||
            cuts.some((cut) => !Number.isFinite(cut)) ||
            cuts.some((cut, index) => index && cut <= cuts[index - 1])
        ) {
            inputError = `${name}: enter strictly increasing custom cuts.`;
            reportValidity();
            return;
        }
        inputError = '';
        reportValidity();
        updateVariable(name, { kind: 'step', knots: cuts, n_bins: null });
    }

    function changeBands(name, value) {
        bandsText = { ...bandsText, [name]: value };
        const bands = Number(value);
        if (!Number.isInteger(bands) || bands < 2 || bands > 200) {
            inputError = `${name}: number of bands must be a whole number from 2 to 200.`;
            reportValidity();
            return;
        }
        inputError = '';
        reportValidity();
        updateVariable(name, { kind: 'step', n_bins: bands, knots: 'quantile' });
    }

    function useCustomCuts(name) {
        customModes = { ...customModes, [name]: true };
        cutsText = { ...cutsText, [name]: '' };
        inputError = `${name}: enter custom cuts.`;
        reportValidity();
    }

    function useAutomaticBreaks(name) {
        aglmCustomModes = { ...aglmCustomModes, [name]: false };
        inputError = '';
        reportValidity();
        updateVariable(name, { breaks: null, n_breaks: effectiveDesign(name).n_breaks ?? 100 });
    }

    function useCustomBreaks(name) {
        aglmCustomModes = { ...aglmCustomModes, [name]: true };
        breaksText = { ...breaksText, [name]: '' };
        inputError = `${name}: enter a full break-point grid, including the lowest and highest values.`;
        reportValidity();
    }

    function changeBreaks(name, value) {
        breaksText = { ...breaksText, [name]: value };
        const breaks = value.trim() === '' ? [] : value.split(',').map((piece) => Number(piece.trim()));
        if (
            breaks.length < 2 ||
            breaks.some((value) => !Number.isFinite(value)) ||
            breaks.some((value, index) => index && value <= breaks[index - 1])
        ) {
            inputError = `${name}: enter a full, strictly increasing break-point grid.`;
            reportValidity();
            return;
        }
        inputError = '';
        reportValidity();
        updateVariable(name, { breaks, n_breaks: null });
    }

    function changeBreakPointCount(name, value) {
        const count = Number(value);
        if (!Number.isInteger(count) || count < 2 || count > 200) {
            inputError = `${name}: number of break points must be a whole number from 2 to 200.`;
            reportValidity();
            return;
        }
        inputError = '';
        reportValidity();
        updateVariable(name, { breaks: null, n_breaks: count });
    }

    function reportValidity() {
        onValidity(jsonError || inputError);
    }

    function applyJson() {
        try {
            if (selectionOnly) {
                const parsed = JSON.parse(jsonText);
                if (!Array.isArray(parsed) || parsed.some((name) => typeof name !== 'string')) {
                    throw new Error('Use a JSON array of variable names.');
                }
                const unknown = parsed.filter((name) => !eligibleNames.includes(name));
                if (unknown.length) throw new Error(`Unknown variable: ${unknown[0]}.`);
                emit(parsed, {
                    defaults: normaliseDefaults(inheritedDesign?.defaults),
                    variables: {},
                });
                jsonText = serialise();
                jsonError = '';
                reportValidity();
                onDraftText('');
                return;
            }
            const parsed = parseModelVariablesJson(jsonText, {
                eligible: eligibleNames,
                columnByName,
                method,
                baseDefaults: design?.defaults || inheritedDesign?.defaults,
            });
            emit(parsed.factors, parsed.design);
            jsonText = modelVariablesJson({
                factors: parsed.factors,
                design: parsed.design,
                inheritedDesign,
            });
            lastModelKey = JSON.stringify({
                factors: parsed.factors,
                design: parsed.design,
                inheritedDesign,
                method,
            });
            jsonError = '';
            reportValidity();
            onDraftText('');
        } catch (error) {
            jsonError = error.message;
            reportValidity();
        }
    }

    function validateJsonText() {
        try {
            if (selectionOnly) {
                const parsed = JSON.parse(jsonText);
                if (!Array.isArray(parsed) || parsed.some((name) => typeof name !== 'string')) {
                    throw new Error('Use a JSON array of variable names.');
                }
                const unknown = parsed.filter((name) => !eligibleNames.includes(name));
                if (unknown.length) throw new Error(`Unknown variable: ${unknown[0]}.`);
                jsonError = '';
                reportValidity();
                onDraftText(jsonText === serialise() ? '' : jsonText);
                return;
            }
            parseModelVariablesJson(jsonText, {
                eligible: eligibleNames,
                columnByName,
                method,
                baseDefaults: design?.defaults || inheritedDesign?.defaults,
            });
            jsonError = '';
            reportValidity();
        } catch (error) {
            jsonError = error.message;
            reportValidity();
        }
        onDraftText(jsonText === serialise() ? '' : jsonText);
    }
</script>

<section class="model-variables" aria-label="Variables for this model">
    {#if showHeading}<div class="variables-heading">
            <div>
                <h3>Variables for this model</h3>
                <p>
                    These choices apply only to this model. The shared data and outcome stay
                    unchanged.
                </p>
            </div>
            <strong>{factors.length} selected</strong>
        </div>{/if}
    <div class="variables-tabs" role="tablist" aria-label="Model variables view">
        <button
            role="tab"
            aria-selected={tab === 'table'}
            class:active={tab === 'table'}
            onclick={() => (tab = 'table')}>Table</button
        >
        {#if !treatmentOnly}<button
                role="tab"
                aria-selected={tab === 'json'}
                class:active={tab === 'json'}
                onclick={() => (tab = 'json')}>JSON</button
            >{/if}
    </div>
    {#if tab === 'table'}
        {#if !selectionOnly && method === 'glm'}
            <div class="variable-defaults">
                <label
                    >Default numeric bands
                    <input
                        aria-label="Default numeric bands"
                        type="number"
                        min="2"
                        max="200"
                        value={normaliseDefaults(design?.defaults || inheritedDesign?.defaults)
                            .n_bins}
                        onchange={(event) => updateDefaults('n_bins', event.currentTarget.value)}
                    />
                </label>
                <p>
                    Numeric variables use this number of training-data bands unless you change that
                    variable below.
                </p>
            </div>
        {:else if !selectionOnly && method === 'catboost'}
            <p class="method-note">
                CatBoost uses the raw values. Set a variable to categorical only when it is a
                category.
            </p>
        {/if}
        <div class="variable-controls">
            <label
                >Find a variable
                <input
                    aria-label="Search model variables"
                    placeholder="Search variables"
                    bind:value={search}
                />
            </label>
            {#if !treatmentOnly}<button
                    class="text-button"
                    onclick={() => (selectedOnly = !selectedOnly)}
                    >{selectedOnly ? 'Show matching variables' : 'Show selected only'}</button
                >{/if}
        </div>
        <div class="variables-scroll">
            <table class:selectionOnly class:treatmentOnly>
                <thead
                    ><tr
                        >{#if !treatmentOnly}<th>Use</th>{/if}<th>Variable</th
                        >{#if !selectionOnly}<th>How this model uses it</th><th>Settings</th
                            >{/if}</tr
                    ></thead
                >
                <tbody>
                    {#each displayed as column (column.name)}
                        {@const selected = chosen.has(column.name)}
                        {@const local = effectiveDesign(column.name)}
                        <tr class:selected>
                            {#if !treatmentOnly}<td
                                    ><input
                                        aria-label={`Use ${column.name}`}
                                        type="checkbox"
                                        checked={selected}
                                        onchange={() => toggle(column.name)}
                                    /></td
                                >{/if}
                            <td
                                ><strong>{column.name}</strong><small
                                    >{column.numeric ? 'Numeric' : 'Categorical'}</small
                                ></td
                            >
                            {#if !selectionOnly}<td>
                                    {#if method === 'glm'}
                                        <select
                                            aria-label={`Type for ${column.name}`}
                                            disabled={!selected}
                                            value={local.kind ||
                                                (column.numeric ? 'step' : 'categorical')}
                                            onchange={(event) =>
                                                changeKind(column.name, event.currentTarget.value)}
                                        >
                                            {#if column.numeric}<option value="step"
                                                    >Numeric bands</option
                                                >{/if}
                                            <option value="categorical">Categorical</option>
                                            {#if column.numeric}<option value="linear"
                                                    >Piecewise linear</option
                                                ><option value="continuous">Continuous</option
                                                ><option value="aglm">AGLM smooth curve</option
                                                ><option value="aglm_l">AGLM L curve</option>{/if}
                                        </select>
                                    {:else}
                                        <select
                                            aria-label={`Type for ${column.name}`}
                                            disabled={!selected}
                                            value={local.kind === 'categorical'
                                                ? 'categorical'
                                                : 'numeric'}
                                            onchange={(event) =>
                                                changeKind(column.name, event.currentTarget.value)}
                                        >
                                            <option value="numeric"
                                                >{column.numeric
                                                    ? 'Raw numeric'
                                                    : 'Infer type from data'}</option
                                            >
                                            <option value="categorical">Categorical</option>
                                        </select>
                                    {/if}
                                </td>
                                <td>
                                    {#if method === 'glm' && selected && column.numeric && (local.kind == null || local.kind === 'step')}
                                        {#if customModes[column.name] || (Array.isArray(local.knots) && local.knots.length)}
                                            <label
                                                >Custom cuts<input
                                                    aria-label={`Custom cuts for ${column.name}`}
                                                    placeholder="e.g. 25, 35, 45"
                                                    value={cutsText[column.name] ??
                                                        (Array.isArray(local.knots)
                                                            ? local.knots.join(', ')
                                                            : '')}
                                                    oninput={(event) =>
                                                        changeCuts(
                                                            column.name,
                                                            event.currentTarget.value,
                                                        )}
                                                /></label
                                            >
                                        {:else}
                                            <label
                                                >Bands<input
                                                    aria-label={`Bands for ${column.name}`}
                                                    type="number"
                                                    min="2"
                                                    max="200"
                                                    value={bandsText[column.name] ??
                                                        (local.n_bins ||
                                                            normaliseDefaults(
                                                                design?.defaults ||
                                                                    inheritedDesign?.defaults,
                                                            ).n_bins)}
                                                    oninput={(event) =>
                                                        changeBands(
                                                            column.name,
                                                            event.currentTarget.value,
                                                        )}
                                                /></label
                                            >
                                            <button
                                                class="text-button"
                                                onclick={() => useCustomCuts(column.name)}
                                                >Use custom cuts</button
                                            >
                                        {/if}
                                    {:else if selected && local.kind === 'categorical'}
                                        <label
                                            >Levels to retain<input
                                                aria-label={`Levels for ${column.name}`}
                                                type="number"
                                                min="1"
                                                value={local.max_levels ?? ''}
                                                placeholder="—"
                                                onchange={(event) =>
                                                    updateVariable(column.name, {
                                                        max_levels: event.currentTarget.value
                                                            ? Number(event.currentTarget.value)
                                                            : null,
                                                    })}
                                            /></label
                                        >
                                        {#if local.max_levels == null}<small class="no-level-limit"
                                                >No count limit</small
                                            >{/if}
                                    {:else if selected && isAglm(local.kind)}
                                        <div class="aglm-settings">
                                            <label
                                                >Break points<select
                                                    aria-label={`Break points for ${column.name}`}
                                                    value={aglmCustomModes[column.name] ||
                                                    (Array.isArray(local.breaks) && local.breaks.length)
                                                        ? 'custom'
                                                        : 'automatic'}
                                                    onchange={(event) =>
                                                        event.currentTarget.value === 'custom'
                                                            ? useCustomBreaks(column.name)
                                                            : useAutomaticBreaks(column.name)}
                                                >
                                                    <option value="automatic">Automatic</option>
                                                    <option value="custom">Custom full grid</option>
                                                </select></label
                                            >
                                            {#if aglmCustomModes[column.name] || (Array.isArray(local.breaks) && local.breaks.length)}
                                                <label class="aglm-wide"
                                                    >Full grid<input
                                                        aria-label={`Custom break points for ${column.name}`}
                                                        placeholder="e.g. 18, 25, 35, 50, 80"
                                                        value={breaksText[column.name] ??
                                                            (Array.isArray(local.breaks)
                                                                ? local.breaks.join(', ')
                                                                : '')}
                                                        oninput={(event) =>
                                                            changeBreaks(
                                                                column.name,
                                                                event.currentTarget.value,
                                                            )}
                                                    /></label
                                                >
                                                <small class="aglm-wide"
                                                    >Include the lowest and highest values.</small
                                                >
                                            {:else}
                                                <label
                                                    >Points<input
                                                        aria-label={`Break-point count for ${column.name}`}
                                                        type="number"
                                                        min="2"
                                                        max="200"
                                                        value={local.n_breaks ?? 100}
                                                        oninput={(event) =>
                                                            changeBreakPointCount(
                                                                column.name,
                                                                event.currentTarget.value,
                                                            )}
                                                    /></label
                                                >
                                            {/if}
                                            <label
                                                >Tails<select
                                                    aria-label={`Tails for ${column.name}`}
                                                    value={local.extrapolation || 'extend'}
                                                    onchange={(event) =>
                                                        updateVariable(column.name, {
                                                            extrapolation: event.currentTarget.value,
                                                        })}
                                                >
                                                    <option value="extend">Extend</option>
                                                    <option value="flat">Flat</option>
                                                </select></label
                                            >
                                            <small class="aglm-help aglm-wide"
                                                >Extend continues the curve beyond the grid. Flat holds the end value.</small
                                            >
                                        </div>
                                    {:else if selected && ['linear', 'continuous'].includes(local.kind)}<span
                                            >Use JSON for this curve.</span
                                        >
                                    {:else}<span>—</span>{/if}
                                </td>{/if}
                        </tr>
                    {/each}
                </tbody>
            </table>
        </div>
    {:else}
        <p class="json-help">
            {selectionOnly
                ? 'Use a JSON array of variable names to select many factors at once.'
                : 'Use this to select and configure many variables at once. It affects this model only. Protected project columns cannot be added here.'}
        </p>
        <textarea
            aria-label="Model variables JSON"
            bind:value={jsonText}
            oninput={validateJsonText}
            spellcheck="false"></textarea>
        {#if jsonError}<p class="json-error" role="alert">{jsonError}</p>{/if}
        <button onclick={applyJson}>Apply variable JSON</button>
    {/if}
</section>

<style>
    .model-variables {
        margin-top: 18px;
        border-top: 1px solid #dae3db;
        padding-top: 18px;
    }
    .variables-heading {
        display: flex;
        justify-content: space-between;
        align-items: start;
        gap: 16px;
    }
    .variables-heading h3 {
        margin: 0;
    }
    .variables-heading p,
    .variable-defaults p,
    .method-note,
    .json-help {
        margin: 5px 0 0;
        color: #526e5d;
        font-size: 12px;
        line-height: 1.5;
    }
    .variables-heading strong {
        white-space: nowrap;
        font-size: 13px;
    }
    .variables-tabs {
        display: flex;
        gap: 4px;
        margin: 14px 0;
        border-bottom: 1px solid #dae3db;
    }
    .variables-tabs button {
        border: 0;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        background: transparent;
        padding: 8px 5px;
        color: #285c45;
    }
    .variables-tabs button.active {
        border-bottom-color: #287550;
        font-weight: bold;
    }
    .variable-defaults {
        display: flex;
        align-items: end;
        gap: 14px;
        padding: 10px;
        background: #f7faf7;
        border: 1px solid #d6e0d7;
    }
    .variable-defaults label {
        display: grid;
        gap: 5px;
        font-size: 12px;
    }
    .variable-defaults input {
        width: 90px;
    }
    .variable-defaults p {
        margin-bottom: 3px;
    }
    .variable-controls {
        display: flex;
        gap: 12px;
        align-items: end;
        margin: 14px 0;
    }
    .variable-controls label {
        display: grid;
        gap: 5px;
        flex: 1;
        max-width: 360px;
        font-size: 12px;
    }
    .text-button {
        border: 0;
        background: transparent;
        color: #154b35;
        text-decoration: underline;
        padding: 0 3px;
    }
    .variables-scroll {
        max-height: 360px;
        overflow: auto;
        border: 1px solid #d6e0d7;
    }
    .variables-scroll table {
        width: 100%;
        min-width: 760px;
        table-layout: fixed;
        border-collapse: collapse;
        font-size: 12px;
    }
    .variables-scroll th,
    .variables-scroll td {
        padding: 8px;
        text-align: left;
        border-bottom: 1px solid #e4ece5;
        vertical-align: middle;
        overflow-wrap: anywhere;
    }
    .variables-scroll th:first-child,
    .variables-scroll td:first-child {
        width: 36px;
    }
    .variables-scroll th:nth-child(2),
    .variables-scroll td:nth-child(2) {
        width: 26%;
    }
    .variables-scroll table.selectionOnly th:nth-child(2),
    .variables-scroll table.selectionOnly td:nth-child(2),
    .variables-scroll table.treatmentOnly th:first-child,
    .variables-scroll table.treatmentOnly td:first-child {
        width: auto;
    }
    .variables-scroll th {
        position: sticky;
        top: 0;
        background: #f5f8f5;
        z-index: 1;
        font-size: 10px;
        color: #526e5d;
    }
    .variables-scroll tr.selected {
        background: #f7faf7;
    }
    .variables-scroll small {
        display: block;
        color: #64776b;
        margin-top: 2px;
    }
    .variables-scroll .no-level-limit {
        font-size: 10px;
    }
    .variables-scroll select {
        min-width: 0;
        width: 100%;
    }
    .variables-scroll td label {
        display: flex;
        gap: 4px;
        align-items: center;
        font-size: 11px;
        min-width: 0;
    }
    .variables-scroll td label input {
        width: 84px;
        min-width: 0;
    }
    .variables-scroll td .text-button {
        font-size: 11px;
        margin-top: 4px;
    }
    .aglm-settings {
        display: grid;
        grid-template-columns: repeat(3, minmax(92px, 1fr));
        gap: 8px;
        align-items: end;
    }
    .variables-scroll td .aglm-settings label {
        display: grid;
        gap: 4px;
        align-items: normal;
        font-size: 11px;
    }
    .aglm-settings input,
    .aglm-settings select {
        width: 100%;
        box-sizing: border-box;
    }
    .aglm-wide {
        grid-column: 1 / -1;
    }
    .variables-scroll .aglm-settings small {
        margin: 0;
        font-size: 10px;
        line-height: 1.35;
    }
    .aglm-help {
        color: #526e5d;
    }
    @media (max-width: 900px) {
        /* The application sidebar leaves a fairly narrow working column at
           this viewport. Keep the table itself in view and put the AGLM
           controls one below another, rather than hiding a third control in
           a horizontal scroll area. */
        .variables-scroll table {
            min-width: 0;
        }
        .variables-scroll th:first-child,
        .variables-scroll td:first-child {
            width: 30px;
        }
        .variables-scroll th:nth-child(2),
        .variables-scroll td:nth-child(2) {
            width: 27%;
        }
        .variables-scroll th:nth-child(3),
        .variables-scroll td:nth-child(3) {
            width: 29%;
        }
        .aglm-settings {
            grid-template-columns: minmax(0, 1fr);
        }
    }
    textarea {
        box-sizing: border-box;
        width: 100%;
        min-height: 320px;
        resize: vertical;
        padding: 12px;
        font:
            12px/1.55 ui-monospace,
            SFMono-Regular,
            Consolas,
            monospace;
        background: #f7faf7;
        border: 1px solid #d6e0d7;
    }
    .json-error {
        color: #994a39;
        font-size: 12px;
    }
    @media (max-width: 680px) {
        .variable-defaults {
            align-items: start;
            flex-direction: column;
        }
        .variables-scroll th,
        .variables-scroll td {
            padding: 6px 4px;
            font-size: 10px;
        }
        .variables-scroll td label {
            align-items: start;
            flex-direction: column;
        }
        .variables-scroll td label input {
            width: 100%;
            box-sizing: border-box;
        }
        .aglm-settings {
            grid-template-columns: repeat(3, minmax(92px, 1fr));
        }
        .variables-scroll td .text-button {
            white-space: normal;
        }
        .variable-controls {
            align-items: start;
            flex-direction: column;
        }
        .variables-heading {
            align-items: start;
        }
    }
</style>
