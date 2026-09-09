import { test } from 'node:test';
import assert from 'node:assert/strict';
import { importanceChartData } from '../src/importanceChartData.js';
import {
    importanceCacheKey,
    cachedImportance,
    rememberImportance,
} from '../src/importanceCache.js';
test('importance ranks signed means without losing zero or negative values and spans whiskers', () => {
    const rows = [
        { variable: 'Zero', importance: 0, std: 0 },
        { variable: 'Negative', importance: -0.2, std: 0.1 },
        { variable: 'Positive', importance: 0.5, std: 0.2 },
    ];
    const original = structuredClone(rows);
    const plot = importanceChartData(rows);
    assert.deepEqual(
        plot.rows.map((row) => row.variable),
        ['Positive', 'Zero', 'Negative'],
    );
    assert.ok(plot.low < -0.3 && plot.high > 0.7);
    assert.deepEqual(rows, original);
    const zero = importanceChartData([{ variable: 'A', importance: 0, std: 0 }]);
    assert.ok(zero.low < 0 && zero.high > 0);
});
test('importance cache belongs to an immutable model fit within one session', () => {
    const key = importanceCacheKey('session', 'model', 'fit1');
    const result = { rows: [{ variable: 'A', importance: 0.123456789 }] };
    rememberImportance(key, result);
    assert.equal(cachedImportance(key), result);
    assert.equal(cachedImportance(importanceCacheKey('session', 'model', 'fit2')), undefined);
    assert.equal(cachedImportance(importanceCacheKey('other', 'model', 'fit1')), undefined);
    assert.equal(cachedImportance(importanceCacheKey('session', 'other', 'fit1')), undefined);
});

import { unsupportedImportanceAction } from '../src/importanceApi.js';
test('old-server bridge applies only to the action enum rejecting importance', () => {
    const detail = [
        {
            type: 'literal_error',
            loc: ['body', 'action'],
            input: 'importance',
            ctx: { expected: "'variable', 'coefficients'" },
        },
    ];
    assert.equal(unsupportedImportanceAction(new Error(JSON.stringify(detail))), true);
    for (const value of [
        'Server busy',
        [{ ...detail[0], type: 'value_error' }],
        [{ ...detail[0], loc: ['body', 'subset'] }],
        [{ ...detail[0], input: 'other' }],
        [{ ...detail[0], ctx: { expected: "'importance', 'variable'" } }],
    ]) {
        assert.equal(
            unsupportedImportanceAction(
                new Error(typeof value === 'string' ? value : JSON.stringify(value)),
            ),
            false,
        );
    }
});
