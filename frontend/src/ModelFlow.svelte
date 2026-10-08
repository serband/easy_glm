<script>
    import { alternativeBlockedReason, buildModelForest, nodeState } from './modelFlow.js';
    export let modelFlow = [];
    export let models = {};
    export let jobs = {};
    export let selected = '';
    export let onSelect = () => {};
    export let onAdd = () => {};
    export let onAlternative = () => {};

    $: forest = buildModelForest(modelFlow, models, jobs);
    function label(node) {
        const components = node.components || node.model?.components || [];
        return components.length ? components.at(-1)?.name || 'Complete model' : 'Single model';
    }
    function inputSummary(node) {
        const components = node.components || node.model?.components || [];
        const last = components.at(-1);
        const inputs = last?.factors || last?.inputs || [];
        const type = { glm: 'GLM', catboost: 'CatBoost', interaction: 'Interaction', catalogue: 'Catalogue' }[last?.kind] || last?.kind || 'Stage';
        return last ? `${type} · ${inputs.length} variable${inputs.length === 1 ? '' : 's'}${inputs.length ? `: ${inputs.join(', ')}` : ''}` : 'Single model recipe';
    }
    function edgeText(node) {
        if (!node.parent_name) return 'Standalone model';
        if (node.edge_status === 'live') return `Builds on ${node.parent_name} · complete model predictions as offset`;
        if (node.edge_status === 'missing') return `Historical parent ${node.parent_name} is unavailable`;
        return `Historical link to ${node.parent_name}`;
    }
    function componentsFor(node) { return node.components || node.model?.components || []; }
</script>

<section class="model-flow" aria-label="Model flow">
    <div class="flow-heading"><div><div class="eyebrow">MODEL MAP</div><h2>Models</h2><p>Connections show which fitted model supplied each offset.</p></div><button onclick={() => onAdd('')}>New standalone model</button></div>
    {#if !forest.nodes.length}<p class="flow-empty">No models yet. Start with a standalone model.</p>{/if}
    <div class="flow-columns">
        {#each forest.roots as root}
            <div class="flow-tree">
                {#snippet card(name)}
                    {@const node = forest.nodes.find((item) => item.name === name)}
                    {@const state = nodeState(node)}
                    <article class:chosen={selected === node.name} class="flow-card" aria-label={`${node.name}, ${state}`}>
                        {#if node.parent}<div class:live={node.edge_status === 'live'} class="flow-connector" aria-label={edgeText(node)}></div>{/if}
                        <button class="flow-select" onclick={() => onSelect(node.name)}><strong>{node.name}</strong><span class="flow-state">{state}</span><span>{label(node)}</span><span class="flow-inputs">{inputSummary(node)}</span><small>{edgeText(node)}</small></button>
                        <div class="flow-actions"><button onclick={() => onAdd(node.name)} disabled={node.append_available === false || !node.job?.applicable}>Add model after this</button><button onclick={() => onAlternative(node.name)} disabled={Boolean(alternativeBlockedReason(node))}>Revise model</button></div>
                        {#if alternativeBlockedReason(node)}<small class="flow-warning">{alternativeBlockedReason(node)}</small>{/if}
                        {#if componentsFor(node).length}<details><summary>Model recipe</summary><ol>{#each componentsFor(node) as component}<li>{component.name} · {component.kind} · {(component.factors || component.inputs || []).join(', ') || 'no selected inputs'}</li>{/each}</ol></details>{/if}
                    </article>
                    {#if node.children.length}<div class="flow-children">{#each node.children as child}{@render card(child)}{/each}</div>{/if}
                {/snippet}
                {@render card(root)}
            </div>
        {/each}
    </div>
</section>

<style>
    .model-flow { margin: 0 0 18px; padding: 18px; border: 1px solid #c7d7ca; border-radius: 12px; background: #f7faf7; }
    .flow-heading { display:flex; align-items:start; justify-content:space-between; gap:16px; }.flow-heading h2 { margin:2px 0; }.flow-heading p { margin:4px 0 0; }
    .flow-columns { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(100%, 310px), 1fr)); gap:16px; padding:8px 0; }.flow-tree { min-width:0; }.flow-card { position:relative; width:100%; min-width:0; margin-top:12px; padding:10px; border:1px solid #b6c9ba; border-radius:9px; background:white; box-sizing:border-box; overflow-wrap:anywhere; }.flow-card.chosen { outline:3px solid #4a8a63; }.flow-select { display:grid; width:100%; min-width:0; white-space:normal; text-align:left; gap:4px; padding:4px; background:transparent; color:inherit; }.flow-select > * { min-width:0; white-space:normal; overflow-wrap:anywhere; }.flow-select strong { font-size:15px; }.flow-select small,.flow-warning { color:#526e5d; }.flow-inputs { color:#375443; font-size:12px; }.flow-state { width:max-content; border-radius:20px; padding:2px 7px; background:#e7f1e8; font-size:12px; text-transform:capitalize; }.flow-actions { display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }.flow-actions button { min-width:0; white-space:normal; font-size:12px; padding:5px 7px; }.flow-children { box-sizing:border-box; width:calc(100% - 34px); min-width:0; margin-left:18px; padding-left:16px; border-left:2px solid #9ab6a0; }.flow-connector { position:absolute; width:17px; height:24px; left:-18px; top:22px; border-left:2px dashed #a27944; border-bottom:2px dashed #a27944; }.flow-connector.live { border-color:#32734d; border-style:solid; }.flow-empty { padding:18px; color:#526e5d; }
    details { margin-top:8px; font-size:12px; } details ol { padding-left:20px; margin:6px 0; }
    @media (max-width: 620px) { .model-flow { padding:13px; }.flow-heading { display:block; }.flow-heading button { margin-top:10px; }.flow-columns { grid-template-columns:minmax(0, 1fr); }.flow-children { width:100%; margin-left:0; padding-left:0; border-left:0; }.flow-connector { display:none; } }
</style>
