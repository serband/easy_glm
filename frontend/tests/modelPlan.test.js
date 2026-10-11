import test from 'node:test';
import assert from 'node:assert/strict';
import { logicalModels, physicalStages, modelForComponent, modelSourceLabel, canonicalPlanPayload } from '../src/modelPlan.js';

test('API core projection does not hide accepted tables from physical order', () => {
    const main = { component_id: 'main', name: 'Main', kind: 'glm' };
    const pair = { component_id: 'pair', name: 'Age × Region', kind: 'interaction' };
    const vehicle = { component_id: 'vehicle', name: 'Vehicle', kind: 'catboost' };
    const plan = {
        format_version: 2,
        stages: [main, vehicle],
        models: [
            { model_id: 'main', name: 'Main', core: main, pair_corrections: [pair] },
            { model_id: 'vehicle', name: 'Vehicle', core: vehicle, pair_corrections: [] },
        ],
    };
    assert.deepEqual(physicalStages(plan).map((item) => item.component_id), ['main', 'pair', 'vehicle']);
    assert.equal(logicalModels(plan).length, 2);
    assert.equal(modelForComponent(plan, 'pair').model_id, 'main');
    assert.equal(modelSourceLabel(plan, 1), 'Offset: complete score through Main, including Age × Region.');
});

test('legacy interactions keep their explicit order and identity', () => {
    const stages = [
        { component_id: 'main', name: 'Main', kind: 'glm' },
        { component_id: 'pair', name: 'Pair', kind: 'interaction' },
        { component_id: 'vehicle', name: 'Vehicle', kind: 'catboost' },
    ];
    const plan = { stages };
    assert.deepEqual(physicalStages(plan), stages);
    assert.equal(logicalModels(plan).length, 3);
    assert.equal(modelForComponent(plan, 'pair').model_id, 'pair');
    assert.equal(modelForComponent(plan, 'pair').legacy, true);
});

test('rendered v2 plans save canonical data without mutating the displayed plan', () => {
    const core = { component_id: 'main', name: 'Main', kind: 'glm' };
    const displayed = {
        format_version: 2,
        models: [{ model_id: 'main', name: 'Main', core, pair_corrections: [] }],
        stages: [core],
        adopted_root: { model_name: 'Original', fit_id: 'frozen' },
        fit_refs: [{ model_id: 'main', through_component_id: 'main', fit_id: 'current' }],
    };
    const saved = canonicalPlanPayload(displayed);
    assert.equal('stages' in saved, false);
    assert.deepEqual(saved.models, displayed.models);
    assert.deepEqual(saved.adopted_root, displayed.adopted_root);
    assert.deepEqual(saved.fit_refs, displayed.fit_refs);
    assert.deepEqual(displayed.stages, [core]);
    assert.deepEqual(canonicalPlanPayload({ stages: [core] }), { stages: [core] });
});
