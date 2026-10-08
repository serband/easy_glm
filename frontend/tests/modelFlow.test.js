import assert from 'node:assert/strict';
import test from 'node:test';
import { alternativeBlockedReason, buildModelForest, nodeState } from '../src/modelFlow.js';

test('buildModelForest keeps roots, branches and server edge status in alphabetical order', () => {
    const forest = buildModelForest({
        nodes: [{ name: 'Zulu' }, { name: 'Alpha' }, { name: 'Child B' }, { name: 'Child A' }],
        edges: [
            { from: 'Alpha', to: 'Child B', edge_status: 'historical' },
            { from: 'Alpha', to: 'Child A', edge_status: 'live' },
        ],
    }, {}, {});
    assert.deepEqual(forest.roots, ['Alpha', 'Zulu']);
    assert.deepEqual(forest.nodes.find((node) => node.name === 'Alpha').children, ['Child A', 'Child B']);
    assert.deepEqual(forest.edges.map((edge) => edge.status), ['live', 'historical']);
});

test('actual edge-only payload breaks a three-node cycle and retains a missing parent', () => {
    const forest = buildModelForest({
        nodes: [{ name: 'A' }, { name: 'B' }, { name: 'C' }, { name: 'Orphan' }],
        edges: [
            { from: 'C', to: 'A', edge_status: 'live' },
            { from: 'A', to: 'B', edge_status: 'live' },
            { from: 'B', to: 'C', edge_status: 'live' },
            { from: 'Gone', to: 'Orphan', edge_status: 'missing' },
        ],
    });
    assert.deepEqual(forest.roots, ['A', 'Orphan']);
    assert.equal(forest.nodes.find((node) => node.name === 'B').parent, 'A');
    assert.equal(forest.nodes.find((node) => node.name === 'C').parent, 'B');
    const orphan = forest.nodes.find((node) => node.name === 'Orphan');
    assert.equal(orphan.parent, null);
    assert.equal(orphan.parent_name, 'Gone');
    assert.equal(orphan.edge_status, 'missing');
});

test('nodeState labels fitted and historical candidates without guessing a score', () => {
    assert.equal(nodeState({ job: { applicable: true } }), 'fitted');
    assert.equal(nodeState({ job: { status: 'running' } }), 'running');
    assert.equal(nodeState({ kind: 'legacy', job: {} }), 'legacy');
    assert.equal(nodeState({ kind: 'stack', job: {} }), 'unfitted');
    assert.equal(nodeState({ status: 'not_fitted', job: {} }), 'not fitted');
});

test('alternatives only copy a complete safe checkpoint', () => {
    assert.match(alternativeBlockedReason({ kind: 'legacy' }), /no staged recipe/);
    assert.match(alternativeBlockedReason({ components: [{}, {}] }), /would change the model/);
    assert.match(alternativeBlockedReason({ parent_name: 'Gone', edge_status: 'missing', components: [{}] }), /no longer current/);
    assert.equal(alternativeBlockedReason({ components: [{}] }), '');
    assert.equal(alternativeBlockedReason({ parent_name: 'Root', edge_status: 'live', components: [{}, {}] }), '');
});
