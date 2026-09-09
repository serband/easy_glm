import { test } from 'node:test';
import assert from 'node:assert/strict';
import { rateChartData } from '../src/rateChartData.js';
test('linear curves follow log slopes, with separate null and flat clamps', () => {
    const plot = rateChartData({
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
test('step and categorical values stay flat or unconnected, page boundaries do not invent fitted slopes', () => {
    const rows = [{ from: 0, to: 10, fitted: 1, relativity: 2, exposure: 7 }];
    assert.deepEqual(
        rateChartData({ columns: [], rows }).points[0].current.map((p) => p.value),
        [2, 2],
    );
    assert.equal(
        rateChartData({ columns: ['slope'], rows: [{ ...rows[0], slope: 0 }] }).points[0].fitted
            .length,
        1,
    );
    assert.equal(
        rateChartData({ columns: [], rows: [{ label: 'A', fitted: 1, relativity: 2 }] }).points[0]
            .current.length,
        1,
    );
    assert.equal(rateChartData({ columns: ['label_a'], rows: [] }).interaction, true);
});
