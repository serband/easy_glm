const DESIGN_KEYS = [
    'kind', 'knots', 'clamp', 'n_bins', 'null_indicator', 'min_level_share',
    'max_levels', 'levels', 'monotone', 'penalty_weight',
];

const VARIABLE_DESIGN_DEFAULTS = {
    kind: null,
    knots: 'quantile',
    clamp: null,
    n_bins: null,
    null_indicator: null,
    min_level_share: null,
    max_levels: null,
    levels: null,
    monotone: null,
    penalty_weight: 1,
};

export const DEFAULT_DESIGN = {
    n_bins: 20,
    min_level_share: 0.0025,
    null_indicator: true,
    max_integer_knots: 150,
};

export function copy(value) {
    return structuredClone(value || {});
}

export function normaliseDefaults(defaults = {}) {
    return {
        ...DEFAULT_DESIGN,
        ...Object.fromEntries(
            Object.entries(copy(defaults)).filter(([key]) => Object.hasOwn(DEFAULT_DESIGN, key))
        ),
    };
}

export function completeVariableDesign(variable, inherited = {}, local = {}) {
    const base = { ...VARIABLE_DESIGN_DEFAULTS, ...copy(inherited[variable] || {}) };
    const override = copy(local[variable] || {});
    const complete = { ...base, ...override };
    return Object.fromEntries(DESIGN_KEYS.map((key) => [key, complete[key]]));
}

function validKnots(name, knots) {
    if (!Array.isArray(knots) || !knots.length)
        throw new Error(`${name}: custom cuts must contain at least one number.`);
    if (knots.some((value) => !Number.isFinite(value)))
        throw new Error(`${name}: custom cuts must be finite numbers.`);
    if (knots.some((value, index) => index && value <= knots[index - 1]))
        throw new Error(`${name}: custom cuts must be strictly increasing.`);
}

function validateDesign(name, design, { method, numeric }) {
    if (!design || typeof design !== 'object' || Array.isArray(design))
        throw new Error(`${name}: variable settings must be an object.`);
    if (Object.keys(design).some((key) => !DESIGN_KEYS.includes(key)))
        throw new Error(`${name}: variable settings contain an unknown field.`);
    if (design.kind != null && !['step', 'categorical', 'linear', 'continuous'].includes(design.kind))
        throw new Error(`${name}: choose numeric bands or categorical.`);
    if (method === 'catboost' && design.kind && design.kind !== 'categorical')
        throw new Error(`${name}: CatBoost uses raw numeric values; only categorical can be forced.`);
    if (method === 'glm' && design.kind === 'categorical' && numeric)
        return;
    if (method === 'glm' && design.kind && !numeric && design.kind !== 'categorical')
        throw new Error(`${name}: a non-numeric column must be categorical.`);
    if (design.n_bins != null && (!Number.isInteger(design.n_bins) || design.n_bins < 2 || design.n_bins > 200))
        throw new Error(`${name}: number of bands must be a whole number from 2 to 200.`);
    if (design.knots != null && design.knots !== 'quantile' && design.knots !== 'integer')
        validKnots(name, design.knots);
}

// JSON.parse keeps the final value when a key appears twice. That is dangerous
// in a settings editor: a pasted duplicate could quietly select the wrong
// variable or discard a cut. Scan objects first so applying remains atomic.
function rejectDuplicateJsonKeys(text) {
    let index = 0;
    const whitespace = () => { while (/\s/.test(text[index] || '')) index += 1; };
    const string = () => {
        const start = index;
        index += 1;
        while (index < text.length) {
            if (text[index] === '\\') { index += 2; continue; }
            if (text[index] === '"') { index += 1; return JSON.parse(text.slice(start, index)); }
            index += 1;
        }
        throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
    };
    const value = () => {
        whitespace();
        if (text[index] === '{') {
            index += 1; whitespace();
            const keys = new Set();
            if (text[index] === '}') { index += 1; return; }
            while (index < text.length) {
                whitespace();
                if (text[index] !== '"') throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
                const key = string();
                if (keys.has(key)) throw new Error(`Variable JSON repeats the key ${key}. Nothing has changed.`);
                keys.add(key); whitespace();
                if (text[index] !== ':') throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
                index += 1; value(); whitespace();
                if (text[index] === '}') { index += 1; return; }
                if (text[index] !== ',') throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
                index += 1;
            }
            throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
        }
        if (text[index] === '[') {
            index += 1; whitespace();
            if (text[index] === ']') { index += 1; return; }
            while (index < text.length) {
                value(); whitespace();
                if (text[index] === ']') { index += 1; return; }
                if (text[index] !== ',') throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
                index += 1;
            }
            throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
        }
        if (text[index] === '"') { string(); return; }
        while (index < text.length && !/[\s,}\]]/.test(text[index])) index += 1;
    };
    value(); whitespace();
    if (index !== text.length) throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
}

/**
 * Read the model-specific JSON editor. Variables present in `variables` are
 * exactly the factors selected for this model; roles remain project-wide.
 */
export function parseModelVariablesJson(text, { eligible, columnByName, method, baseDefaults = {} }) {
    let parsed;
    rejectDuplicateJsonKeys(text);
    try {
        parsed = JSON.parse(text);
    } catch (_) {
        throw new Error('Variable JSON is not valid JSON. Nothing has changed.');
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
        throw new Error('Variable JSON must be an object with defaults and variables.');
    if (Object.keys(parsed).some((key) => !['defaults', 'variables'].includes(key)))
        throw new Error('Variable JSON contains an unknown field. Nothing has changed.');
    if (parsed.defaults != null && (typeof parsed.defaults !== 'object' || Array.isArray(parsed.defaults)))
        throw new Error('defaults must be an object.');
    if (parsed.defaults && Object.keys(parsed.defaults).some((key) => !Object.hasOwn(DEFAULT_DESIGN, key)))
        throw new Error('defaults contain an unknown field.');
    if (!parsed.variables || typeof parsed.variables !== 'object' || Array.isArray(parsed.variables))
        throw new Error('variables must be an object keyed by column name.');
    const names = Object.keys(parsed.variables);
    const allowed = new Set(eligible);
    for (const name of names) {
        if (!allowed.has(name))
            throw new Error(`${name} is not available for this model. Nothing has changed.`);
        validateDesign(name, parsed.variables[name], {
            method,
            numeric: Boolean(columnByName.get(name)?.numeric),
        });
    }
    const defaults = normaliseDefaults({ ...baseDefaults, ...parsed.defaults });
    if (!Number.isInteger(defaults.n_bins) || defaults.n_bins < 2 || defaults.n_bins > 200)
        throw new Error('Default number of bands must be a whole number from 2 to 200.');
    return { factors: names.sort((a, b) => a.localeCompare(b)), design: { defaults, variables: copy(parsed.variables) } };
}

export function modelVariablesJson({ factors, design, inheritedDesign }) {
    const defaults = normaliseDefaults({ ...inheritedDesign?.defaults, ...design?.defaults });
    const local = design?.variables || {};
    const variables = Object.fromEntries(
        [...factors].sort((a, b) => a.localeCompare(b)).map((name) => [
            name,
            compactVariableDesign(
                completeVariableDesign(name, inheritedDesign?.variables, local),
                completeVariableDesign(name, inheritedDesign?.variables, {}),
            ),
        ])
    );
    return JSON.stringify({ defaults, variables }, null, 2);
}

// The saved recipe needs every field, but the editor should show only choices
// made for this model. Missing fields mean “use the inherited project setting”.
// An explicit null remains when it clears an inherited setting.
function compactVariableDesign(current, inherited) {
    return Object.fromEntries(
        DESIGN_KEYS
            .filter((key) => JSON.stringify(current[key]) !== JSON.stringify(inherited[key]))
            .map((key) => [key, current[key]])
    );
}
