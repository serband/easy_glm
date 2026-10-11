/** Helpers for the v2 logical model-plan representation.
 *
 * Old saved plans expose peer `stages`; new plans keep a core model and its
 * interaction-table corrections together in `models`, while the server also
 * provides a read-compatible `stages` projection.
 */
export function isLogicalPlan(plan) {
    return Number(plan?.format_version) === 2 && Array.isArray(plan?.models);
}

export function physicalStages(plan) {
    if (!plan) return [];
    if (isLogicalPlan(plan))
        return plan.models.flatMap((model) => [model.core, ...(model.pair_corrections || [])]);
    return plan.stages || [];
}

export function logicalModels(plan) {
    if (!plan) return [];
    if (isLogicalPlan(plan)) return plan.models;
    return (plan.stages || []).map((stage) => ({
        model_id: stage.component_id,
        name: stage.name,
        core: stage,
        pair_corrections: [],
        legacy: true,
    }));
}

export function componentForModel(model) {
    return model?.core || null;
}

export function modelForComponent(plan, componentId) {
    return logicalModels(plan).find(
        (model) =>
            model.core?.component_id === componentId ||
            (model.pair_corrections || []).some((pair) => pair.component_id === componentId),
    );
}

export function methodLabel(kind) {
    return (
        {
            glm: 'GLM',
            catboost: 'CatBoost',
            catalogue: 'Catalogue',
            interaction: 'Interaction table',
        }[kind] ||
        kind ||
        'Model'
    );
}

export function modelScopeLabel(model) {
    const tables = (model?.pair_corrections || [])
        .map((pair) => pair.name || pair.label || pair.factors?.join(' × '))
        .filter(Boolean);
    return tables.length
        ? `${model?.name || 'Model'}, including ${tables.join(' + ')}`
        : model?.name || 'Model';
}

export function modelSourceLabel(plan, modelIndex) {
    const models = logicalModels(plan);
    const adoptedName = plan?.adopted_root?.model_name || plan?.adopted_model_name;
    if (modelIndex <= 0) {
        return adoptedName
            ? `Offset: complete score from ${adoptedName}.`
            : 'Starts from project data.';
    }
    const prefix = [
        ...(plan?.adopted_root && adoptedName ? [adoptedName] : []),
        ...models.slice(0, modelIndex).map(modelScopeLabel),
    ];
    if (adoptedName && !plan?.adopted_root) prefix[0] = adoptedName;
    return `Offset: complete score through ${prefix.join(' → ')}.`;
}

/** Return the canonical payload for a plan write.
 *
 * The API decorates v2 reads with a `stages` projection for older views. That
 * projection is intentionally not part of the persisted v2 schema.
 */
export function canonicalPlanPayload(plan) {
    if (!plan || !isLogicalPlan(plan)) return plan;
    const payload = structuredClone(plan);
    delete payload.stages;
    return payload;
}
