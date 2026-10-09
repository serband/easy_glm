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

test('nodeState gives job and scorer availability precedence over historical recovery markers', () => {
    assert.equal(nodeState({ job: { applicable: true } }), 'fitted');
    assert.equal(nodeState({ own_fit_available: true, missing_own_fit: true, job: {} }), 'fitted');
    assert.equal(nodeState({ own_fit_available: true, job: { status: 'running' } }), 'fitting');
    assert.equal(nodeState({ job: { status: 'queued' } }), 'fitting');
    assert.equal(nodeState({ status: 'missing_fit', missing_own_fit: true, job: { status: 'complete' } }), 'cannot score');
    assert.equal(nodeState({ full_refit_required: true, own_fit_available: false, job: {} }), 'needs a new fit');
    assert.equal(nodeState({ kind: 'legacy', job: {} }), 'draft');
    assert.equal(nodeState({ kind: 'stack', job: {} }), 'draft');
});

test('alternatives only copy a complete safe checkpoint', () => {
    assert.match(alternativeBlockedReason({ revise_available: false, components: [{}] }), /original parent fit is unavailable/);
    assert.match(alternativeBlockedReason({ kind: 'legacy' }), /no staged recipe/);
    assert.match(alternativeBlockedReason({ components: [{}, {}] }), /would change the model/);
    assert.match(alternativeBlockedReason({ parent_name: 'Gone', edge_status: 'missing', components: [{}] }), /no longer current/);
    assert.equal(alternativeBlockedReason({ components: [{}] }), '');
    assert.equal(alternativeBlockedReason({ parent_name: 'Root', edge_status: 'live', components: [{}, {}] }), '');
    assert.match(
        alternativeBlockedReason({ status: 'missing_fit', missing_own_fit: true, parent_name: 'Root', edge_status: 'live', components: [{}] }),
        /fitted files are unavailable/,
    );
    const liveChildWithMissingParent = {
        own_fit_available: true,
        append_available: true,
        parent_fit_available: false,
        revise_available: false,
        parent_name: 'Gone',
        edge_status: 'missing',
        components: [{}],
    };
    assert.match(alternativeBlockedReason(liveChildWithMissingParent), /original parent fit is unavailable/);
});
