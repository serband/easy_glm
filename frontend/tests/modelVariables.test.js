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
