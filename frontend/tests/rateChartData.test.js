import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rateChartData, rateChartKind } from '../src/rateChartData.js';
test('linear curves follow log slopes, with separate null and flat clamps', () => {
    const plot = rateChartData({
        kind: 'linear',
        columns: ['slope'],
        rows: [
            { from: null, to: 0, fitted: 1, relativity: 2, exposure: 5 },
            { from: 0, to: 10, fitted: 1, relativity: 2, slope: Math.log(2) / 10, exposure: 20 },
            { from: 10, to: null, fitted: 3, relativity: 4, exposure: 1 },
            { from: null, to: null, fitted: 1, relativity: 1, exposure: 2 },
        ],
    });
    assert.equal(plot.points[0].current.at(-1).value, 2);
    assert.ok(Math.abs(plot.points[1].current[8].value - 2 * Math.sqrt(2)) < 1e-12);
    assert.equal(plot.points[1].current.at(-1).value, 4);
    assert.ok(Math.abs(plot.points[1].fitted.at(-1).value - 3) < 1e-12);
    assert.equal(plot.points[3].current.length, 1);
    assert.ok(plot.points[3].x > plot.points[2].current.at(-1).x);
});
test('band points preserve values and page boundaries do not invent fitted slopes', () => {
    const rows = [{ from: 0, to: 10, fitted: 1, relativity: 2, exposure: 7 }];
    assert.deepEqual(
        rateChartData({ kind: 'step', columns: [], rows }).points[0].current.map((p) => p.value),
        [2],
    );
    assert.equal(
        rateChartData({ kind: 'linear', columns: ['slope'], rows: [{ ...rows[0], slope: 0 }] })
            .points[0].fitted.length,
        1,
    );
    assert.equal(
        rateChartData({ columns: [], rows: [{ label: 'A', fitted: 1, relativity: 2 }] }).points[0]
            .current.length,
        1,
    );
    assert.equal(
        rateChartData({ kind: 'interaction', columns: ['label_a'], rows: [] }).interaction,
        true,
    );
});

test('connected band trend has one point per band, skips null, and keeps exposure row alignment', () => {
    const p = rateChartData({
        kind: 'step',
        columns: [],
        rows: [
            { from: null, to: 10, fitted: 1, relativity: 2, exposure: 4 },
            { from: 10, to: null, fitted: 3, relativity: 4, exposure: 8 },
            { from: null, to: null, fitted: 9, relativity: 9, exposure: 1 },
        ],
    });
    assert.equal(p.lines.current.length, 1);
    assert.deepEqual(
        p.lines.current[0].map((v) => v.value),
        [2, 4],
    );
    assert.ok(p.lines.current[0][1].x > p.lines.current[0][0].x);
    assert.deepEqual(
        p.points.map((p) => p.row.exposure),
        [4, 8, 1],
    );
});
test('applied types control numeric and categorical charts, including numeric-looking levels', () => {
    const wb = {
        design: { variables: { forced: { kind: 'categorical' }, curve: { kind: 'continuous' } } },
        columns: [
            { name: 'number', dtype: 'Int32' },
            { name: 'text', dtype: 'String' },
            { name: 'forced', dtype: 'Int32' },
        ],
    };
    const table = { columns: [] };
    assert.equal(rateChartKind(table, 'number', wb), 'step');
    assert.equal(rateChartKind(table, 'text', wb), 'categorical');
    assert.equal(rateChartKind(table, 'forced', wb), 'categorical');
    assert.equal(rateChartKind(table, 'curve', wb), 'continuous');
    const p = rateChartData({
        kind: 'categorical',
        columns: [],
        rows: [
            { from: '10', to: '10', label: '10', fitted: 1, relativity: 2 },
            { from: '20', to: '20', label: '20', fitted: 3, relativity: 4 },
        ],
    });
    assert.equal(p.numeric, false);
    assert.equal(p.lines.current.length, 0);
    assert.equal(p.points[0].current[0].value, 2);
});
