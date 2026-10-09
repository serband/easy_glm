/** Turn additive server lineage into a safe, stable forest for the UI. */
export function buildModelForest(modelFlow = [], models = {}, jobs = {}) {
    const serverNodes = Array.isArray(modelFlow) ? modelFlow : modelFlow?.nodes || [];
    const serverEdges = Array.isArray(modelFlow) ? [] : modelFlow?.edges || [];
    const source = new Map();
    for (const item of serverNodes) if (item?.name && !source.has(item.name)) source.set(item.name, item);
    for (const name of Object.keys(models || {})) {
        if (!source.has(name)) source.set(name, { name, kind: models[name]?.components?.length ? 'stack' : 'legacy' });
    }
    const nodes = [...source.values()]
        .map((item) => ({ ...item, model: models[item.name] || {}, job: jobs[item.name] || {}, parent: null, children: [] }))
        .sort((a, b) => a.name.localeCompare(b.name));
    const node = new Map(nodes.map((item) => [item.name, item]));

    // The public API carries durable lineage in edges. Normalise it before walking.
    const incoming = new Map();
    for (const edge of [...serverEdges].filter((edge) => edge?.to && edge?.from).sort((a, b) =>
        a.to.localeCompare(b.to) || a.from.localeCompare(b.from))) {
        if (!incoming.has(edge.to)) incoming.set(edge.to, edge);
    }
    for (const item of nodes) {
        const edge = incoming.get(item.name);
        if (edge) {
            item.parent_name = edge.from;
            item.edge_status = edge.edge_status || item.edge_status || 'historical';
        }
    }

    // Candidate links only include present parents. Detach the alphabetically first
    // node in each cycle so every model remains reachable from a displayed root.
    const candidate = new Map(nodes
        .filter((item) => item.parent_name && item.parent_name !== item.name && node.has(item.parent_name))
        .map((item) => [item.name, item.parent_name]));
    for (const start of nodes.map((item) => item.name)) {
        const path = [];
        const position = new Map();
        let current = start;
        while (candidate.has(current) && !position.has(current)) {
            position.set(current, path.length); path.push(current); current = candidate.get(current);
        }
        if (position.has(current)) {
            const cycle = path.slice(position.get(current));
            candidate.delete([...cycle].sort((a, b) => a.localeCompare(b))[0]);
        }
    }

    const edges = [];
    for (const child of nodes) {
        const parentName = candidate.get(child.name);
        if (!parentName) continue;
        child.parent = parentName;
        node.get(parentName).children.push(child.name);
        edges.push({ from: parentName, to: child.name, status: child.edge_status || 'historical' });
    }
    for (const item of nodes) item.children.sort((a, b) => a.localeCompare(b));
    return { nodes, roots: nodes.filter((item) => !item.parent).map((item) => item.name), edges };
}

export function nodeState(node) {
    const job = node?.job || {};
    // A job that has been accepted is the newest fact. Do not let an older
    // recovery marker make a model look unavailable while it is being rebuilt.
    if (job.status === 'accepted' || job.status === 'queued' || job.status === 'running') return 'fitting';
    // A complete scorer is still useful when the model it originally built on
    // has since disappeared. Only revising that last addition is unavailable.
    if (job.applicable || node?.own_fit_available) return 'fitted';
    if (node?.missing_own_fit || node?.status === 'missing_fit') return 'cannot score';
    if (node?.full_refit_required) return 'needs a new fit';
    return 'draft';
}

export function alternativeBlockedReason(node) {
    const components = node.components || node.model?.components || [];
    if (node.missing_own_fit || node.status === 'missing_fit') return 'This model’s fitted files are unavailable. Create and fit a new copy of the full model instead.';
    if (node.revise_available === false) return 'The original parent fit is unavailable, so this last addition cannot be revised.';
    if (node.parent_name && node.edge_status !== 'live') return 'The parent fit is no longer current.';
    if (node.parent_name) return '';
    if (!components.length) return 'A legacy single model has no staged recipe to copy.';
    if (components.length > 1) return 'This older stack has no lineage checkpoint, so copying only its last stage would change the model.';
    return '';
}
