import { test } from 'node:test';
import assert from 'node:assert/strict';
import { comparisonIssue, comparisonMetrics, comparisonSettings } from '../src/comparison.js';
const config = {
    family: 'poisson',
    target: 'Claims',
    weight: 'Exposure',
    divide_target_by_weight: true,
    predictors: ['Age'],
};
const first = {
    link: 'log',
    summary: { alpha_stage2: null },
    metrics: {
        train: {
            rows: 100,
            exposure: 80,
            actual: 9,
            expected: 10,
            ae: 0.9,
            gini: 0.2,
            mean_deviance: 0.5,
            deviance_explained: 0.1,
        },
    },
};
const second = {
    ...first,
    metrics: { train: { ...first.metrics.train, ae: 1.05, expected: 9 / 1.05, gini: null } },
};
test('comparison deltas preserve source precision and missing scores', () => {
    assert.equal(comparisonIssue(config, config, first, second, 'train'), '');
    const rows = comparisonMetrics(first, second, 'train');
    assert.equal(rows[0].delta, 1.05 - 0.9);
    assert.equal(rows[1].delta, null);
    assert.equal(rows[4].challenger, 9 / 1.05);
});
test('comparison refuses missing, different response, exposure, family, link and rows', () => {
    assert.ok(comparisonIssue(config, null, first, null, 'train'));
    for (const [key, value] of [
        ['target', 'Cost'],
        ['weight', null],
        ['family', 'gamma'],
        ['divide_target_by_weight', false],
    ])
        assert.ok(comparisonIssue(config, { ...config, [key]: value }, first, second, 'train'));
    assert.ok(comparisonIssue(config, config, first, { ...second, link: 'logit' }, 'train'));
    assert.ok(comparisonIssue(config, config, first, second, 'holdout'));
    assert.ok(
        comparisonIssue(
            config,
            config,
            first,
            { ...second, metrics: { train: { ...second.metrics.train, rows: 99 } } },
            'train',
        ),
    );
});
test('changed settings retain numeric values for display formatting and export', () => {
    const rows = comparisonSettings(
        { ...config, penalty: { alpha: 0.123456 } },
        { ...config, penalty: { alpha: 0.234567 } },
        first,
        second,
    );
    assert.deepEqual(rows, [
        { setting: 'penalty · alpha', baseline: 0.123456, challenger: 0.234567 },
    ]);
});
