import { test } from 'node:test';
import assert from 'node:assert/strict';
import { formatNumber, formatRelativity, formatLabels, axisLabel } from '../src/format.js';
test('display numbers have at most three decimals without destroying tiny values', () => {
    const values = [
        0.051361,
        0.46796,
        0.001689654,
        18354.721987,
        -0,
        0,
        12345,
        NaN,
        Infinity,
        null,
    ];
    assert.deepEqual(
        values.map((v) => formatNumber(v)),
        ['0.051', '0.468', '1.69e-3', '18,354.722', '0', '0', '12,345', '—', '—', '—'],
    );
    assert.equal(formatNumber(1.23456789e-15), '1.235e-15');
    assert.equal(formatNumber(1.23001, { scientific: true }), '1.23e0');
    assert.equal(values[0], 0.051361);
});
test('categorical identities remain exact and colliding range labels are distinct', () => {
    assert.deepEqual(formatLabels(['001.2345', 'Zone 1.2345']), ['001.2345', 'Zone 1.2345']);
    const labels = formatLabels(['[10.0001, 20.0001)', '[10.0002, 20.0002)']);
    assert.notEqual(labels[0], labels[1]);
    assert.deepEqual(
        formatLabels(['< 4035.7709328438937', '[4035.7709328438937, 5234.523097661563)']),
        ['< 4035.771', '[4035.771, 5234.523)'],
    );
});

test('narrow bands keep an identity instead of a false zero-width range', () => {
    assert.deepEqual(formatLabels(['[1.0001, 1.0002)']), ['Band 1']);
    assert.equal(axisLabel('[4035.771, 5234.523)'), '4035.771');
});

test('relativity display is at most four decimal places without changing stored values', () => {
    const source = [0.688720763, 1.23456789, 12345.67891, 0.00001, -0.00001, null];
    const original = [...source];
    assert.deepEqual(
        source.map((value) => formatRelativity(value)),
        ['0.6887', '1.2346', '12,345.6789', '0', '0', '—'],
    );
    assert.equal(formatRelativity(12345.67891, { grouping: false }), '12345.6789');
    assert.equal(formatRelativity(1), '1');
    assert.deepEqual(source, original);
    assert.equal(formatNumber(0.688720763), '0.689');
});
