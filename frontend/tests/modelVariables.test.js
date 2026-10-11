import assert from 'node:assert/strict';
import test from 'node:test';
import { completeVariableDesign, modelVariablesJson, parseModelVariablesJson } from '../src/modelVariables.js';

const columns = new Map([
    ['DriverAge', { numeric: true }],
    ['Region', { numeric: false }],
    ['VehicleAge', { numeric: true }],
]);

test('model variable JSON selects only its own variables and preserves local cuts', () => {
    const parsed = parseModelVariablesJson(JSON.stringify({
        defaults: { n_bins: 8 },
        variables: {
            DriverAge: { kind: 'step', knots: [25, 35, 45] },
            Region: { kind: 'categorical' },
        },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' });
    assert.deepEqual(parsed.factors, ['DriverAge', 'Region']);
    assert.deepEqual(parsed.design.variables.DriverAge.knots, [25, 35, 45]);
    assert.equal(parsed.design.defaults.n_bins, 8);
});

test('model variable JSON is atomic when a protected or unknown column is pasted', () => {
    assert.throws(() => parseModelVariablesJson(JSON.stringify({
        variables: { ClaimNb: { kind: 'step' } },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' }), /not available/);
});

test('duplicate JSON keys are rejected before they can replace a setting', () => {
    assert.throws(() => parseModelVariablesJson(`{
        "variables": {
            "DriverAge": { "kind": "step" },
            "DriverAge": { "kind": "categorical" }
        }
    }`, { eligible: [...columns.keys()], columnByName: columns, method: 'glm' }), /repeats the key DriverAge/);
});

test('prototype-looking and unrelated keys are rejected atomically', () => {
    for (const key of ['__proto__', 'constructor', 'toString']) {
        assert.throws(() => parseModelVariablesJson(JSON.stringify({
            defaults: { [key]: 8 }, variables: { DriverAge: {} },
        }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' }), /unknown field/);
    }
    assert.throws(() => parseModelVariablesJson(JSON.stringify({
        variables: { DriverAge: {} }, extra: true,
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' }), /unknown field/);
});

test('CatBoost refuses GLM bands and accepts a forced categorical field', () => {
    assert.throws(() => parseModelVariablesJson(JSON.stringify({
        variables: { DriverAge: { kind: 'step', n_bins: 8 } },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'catboost' }), /raw numeric/);
    const parsed = parseModelVariablesJson(JSON.stringify({
        variables: { Region: { kind: 'categorical' } },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'catboost' });
    assert.deepEqual(parsed.factors, ['Region']);
});

test('AGLM curve settings are accepted only for numeric GLM variables', () => {
    const parsed = parseModelVariablesJson(JSON.stringify({
        variables: {
            DriverAge: {
                kind: 'aglm',
                breaks: [18, 25, 35, 50, 80],
                extrapolation: 'flat',
            },
            VehicleAge: { kind: 'aglm_l', n_breaks: 100, extrapolation: 'extend' },
        },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' });
    assert.equal(parsed.design.variables.DriverAge.kind, 'aglm');
    assert.deepEqual(parsed.design.variables.DriverAge.breaks, [18, 25, 35, 50, 80]);
    assert.equal(parsed.design.variables.VehicleAge.n_breaks, 100);
    assert.equal(parsed.design.variables.VehicleAge.extrapolation, 'extend');

    assert.throws(() => parseModelVariablesJson(JSON.stringify({
        variables: { Region: { kind: 'aglm', n_breaks: 100 } },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' }), /numeric column/);
    assert.throws(() => parseModelVariablesJson(JSON.stringify({
        variables: { DriverAge: { kind: 'aglm', n_breaks: 100 } },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'catboost' }), /raw numeric values/);
});

test('AGLM curve settings reject incompatible controls and invalid grids', () => {
    const settings = (variable) => parseModelVariablesJson(JSON.stringify({
        variables: { DriverAge: variable },
    }), { eligible: [...columns.keys()], columnByName: columns, method: 'glm' });

    assert.throws(() => settings({ kind: 'step', n_breaks: 100 }), /only be used with an AGLM curve/);
    assert.throws(() => settings({ kind: 'aglm', n_bins: 8 }), /remove numeric bands/);
    assert.throws(() => settings({ kind: 'aglm_l', monotone: 'increasing' }), /remove numeric bands/);
    assert.throws(() => settings({ kind: 'aglm', knots: [18, 25, 80] }), /automatic break points/);
    assert.throws(() => settings({ kind: 'aglm', breaks: [18] }), /lowest and highest/);
    assert.throws(() => settings({ kind: 'aglm', breaks: [18, 18, 80] }), /strictly increasing/);
    assert.throws(() => settings({ kind: 'aglm', n_breaks: 201 }), /2 to 200/);
    assert.throws(() => settings({ kind: 'aglm', extrapolation: 'curve' }), /Extend or Flat/);
});

test('model JSON serialises variables alphabetically and does not include unselected inherited variables', () => {
    const text = modelVariablesJson({
        factors: ['VehicleAge', 'DriverAge'],
        design: { defaults: { n_bins: 6 }, variables: { DriverAge: { kind: 'step', knots: [30, 40] } } },
        inheritedDesign: { defaults: { n_bins: 8 }, variables: { VehicleAge: { kind: 'step', n_bins: 4 }, Region: { kind: 'categorical' } } },
    });
    const parsed = JSON.parse(text);
    assert.deepEqual(Object.keys(parsed.variables), ['DriverAge', 'VehicleAge']);
    assert.equal(parsed.variables.Region, undefined);
    assert.deepEqual(parsed.variables.VehicleAge, {});
});

test('generated automatic-band JSON can be pasted back unchanged', () => {
    const text = modelVariablesJson({
        factors: ['DriverAge'],
        design: { defaults: { n_bins: 8 }, variables: {} },
        inheritedDesign: { defaults: { n_bins: 8 }, variables: {} },
    });
    const parsed = parseModelVariablesJson(text, {
        eligible: [...columns.keys()], columnByName: columns, method: 'glm',
    });
    assert.deepEqual(parsed.factors, ['DriverAge']);
    assert.deepEqual(parsed.design.variables.DriverAge, {});
});

test('partial JSON defaults preserve the current model defaults', () => {
    const parsed = parseModelVariablesJson(JSON.stringify({
        defaults: { n_bins: 6 }, variables: { DriverAge: {} },
    }), {
        eligible: [...columns.keys()], columnByName: columns, method: 'glm',
        baseDefaults: { n_bins: 20, min_level_share: 0.01, null_indicator: false, max_integer_knots: 77 },
    });
    assert.deepEqual(parsed.design.defaults, {
        n_bins: 6, min_level_share: 0.01, null_indicator: false, max_integer_knots: 77,
    });
});

test('the compact JSON round-trips to the complete saved design', () => {
    const inherited = {
        defaults: { n_bins: 20, min_level_share: 0.01, null_indicator: false, max_integer_knots: 90 },
        variables: { DriverAge: { kind: 'step', knots: 'quantile', monotone: 'increasing' } },
    };
    const text = modelVariablesJson({
        factors: ['DriverAge', 'Region'],
        inheritedDesign: inherited,
        design: {
            defaults: inherited.defaults,
            variables: {
                DriverAge: completeVariableDesign('DriverAge', inherited.variables, {
                    DriverAge: { knots: [25, 35, 45], monotone: null },
                }),
            },
        },
    });
    assert.equal(text.includes('penalty_weight'), false);
    assert.equal(text.includes('"clamp"'), false);
    const shown = JSON.parse(text);
    assert.deepEqual(shown.variables.Region, {});
    assert.deepEqual(shown.variables.DriverAge, { knots: [25, 35, 45], monotone: null });
    const parsed = parseModelVariablesJson(text, {
        eligible: ['DriverAge', 'Region'],
        columnByName: new Map([['DriverAge', { numeric: true }], ['Region', { numeric: false }]]),
        method: 'glm',
        baseDefaults: inherited.defaults,
    });
    assert.deepEqual(
        completeVariableDesign('DriverAge', inherited.variables, parsed.design.variables),
        completeVariableDesign('DriverAge', inherited.variables, {
            DriverAge: { knots: [25, 35, 45], monotone: null },
        }),
    );
});
