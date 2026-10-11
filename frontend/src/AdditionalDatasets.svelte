<script>
    export let api;
    export let upload;
    export let state;
    export let datasets = [];
    export let onChanged = () => {};

    let name = '', source = 'upload', path = '', file = null;
    let pending = '', error = '', inspecting = '', replaceId = '';
    $: canAdd = name.trim() && (source === 'upload' ? file : path.trim());
    const label = (dataset) => `${dataset.rows?.toLocaleString?.() ?? '—'} rows · ${dataset.columns?.length ?? 0} columns`;
    function status(dataset) {
        if (dataset.status === 'available') return 'Ready';
        if (dataset.status === 'unavailable') return 'Unavailable';
        return 'Changed';
    }
    async function refresh() {
        const response = await api('datasets');
        onChanged(response.datasets || []);
    }
    async function add() {
        if (!canAdd || pending) return;
        pending = replaceId || 'add'; error = '';
        try {
            const body = { session_id: state.session_id, revision: state.revision, name: name.trim() };
            if (source === 'upload') await upload(file, { ...body, replace_id: replaceId || undefined });
            else await api('datasets/file', { ...body, path: path.trim(), replace_id: replaceId || undefined });
            name = ''; path = ''; file = null; replaceId = '';
            await refresh();
        } catch (e) { error = e.message; }
        finally { pending = ''; }
    }
    async function remove(dataset) {
        if (pending || !window.confirm(`Remove scoring dataset “${dataset.name}”?`)) return;
        pending = dataset.id; error = '';
        try {
            await api(`datasets/${encodeURIComponent(dataset.id)}/remove`, { session_id: state.session_id, revision: state.revision });
            await refresh();
        } catch (e) { error = e.message; }
        finally { pending = ''; }
    }
</script>

<section class="model-card additional-datasets" aria-label="Additional scoring datasets">
    <div class="dataset-heading"><div><h2>Additional scoring datasets</h2><p>Files you can score with saved fitted models. They never replace training data or refit a model.</p></div></div>
    {#if error}<p class="message error" role="alert">{error}</p>{/if}
    {#if datasets.length}
        <div class="dataset-list" role="list">
            {#each datasets as dataset (dataset.id)}
                <article class="dataset-row" role="listitem">
                    <div class="dataset-main"><strong>{dataset.name}</strong><span class:unavailable={dataset.status !== 'available'}>{status(dataset)}</span></div>
                    <span class="dataset-source" title={dataset.path || dataset.filename || ''}>{dataset.source === 'memory' ? 'Session only · register again after restarting EasyGLM' : dataset.path || dataset.filename || 'Uploaded file'}</span>
                    <span class="dataset-count">{label(dataset)}</span>
                    <div class="dataset-actions"><button onclick={() => (inspecting = inspecting === dataset.id ? '' : dataset.id)}>Inspect</button><button onclick={() => { name = dataset.name; replaceId = dataset.id; }} disabled={!!pending}>Replace</button><button onclick={() => remove(dataset)} disabled={!!pending}>Remove</button></div>
                    {#if dataset.message}<p class="dataset-message">{dataset.message}</p>{/if}
                    {#if inspecting === dataset.id}<div class="dataset-columns"><strong>Columns</strong>{#each dataset.columns || [] as column}<span>{column.name} <small>{column.dtype}</small></span>{/each}</div>{/if}
                </article>
            {/each}
        </div>
    {:else}<p class="help-text">No scoring datasets yet. Add a CSV or Parquet file below; your training source stays as it is.</p>{/if}
    <form class="dataset-add" onsubmit={(event) => { event.preventDefault(); void add(); }}>
        <label>Dataset name<input aria-label="Scoring dataset name" bind:value={name} placeholder="e.g. October renewal" /></label>
        <fieldset><legend>Source</legend><label><input type="radio" bind:group={source} value="upload" /> Browse file</label><label><input type="radio" bind:group={source} value="path" /> File path</label></fieldset>
        {#if source === 'upload'}<label>File<input aria-label="Scoring dataset file" type="file" accept=".csv,.parquet" onchange={(event) => (file = event.currentTarget.files?.[0] || null)} /></label>{:else}<label>File path<input aria-label="Scoring dataset path" bind:value={path} placeholder="/path/to/renewals.parquet" /></label>{/if}
        <button class="primary" disabled={!canAdd || !!pending}>{pending ? 'Saving…' : replaceId ? 'Replace dataset' : 'Add scoring dataset'}</button>
    </form>
</section>

<style>
    .dataset-heading p { margin-top: 7px; }.dataset-list { margin-top: 18px; border-top: 1px solid var(--border); }.dataset-row { display:grid; grid-template-columns:minmax(150px,1.2fr) minmax(150px,2fr) minmax(115px,.8fr) auto; gap:10px 16px; padding:14px 0; border-bottom:1px solid var(--border); align-items:center; }.dataset-main { display:grid; gap:4px; }.dataset-main span { color:#17664c; font-size:12px; }.dataset-main span.unavailable { color:#9a4e18; }.dataset-source { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--muted); }.dataset-count { white-space:nowrap; color:var(--muted); }.dataset-actions { display:flex; gap:7px; }.dataset-message { grid-column:1/-1; color:#9a4e18; }.dataset-columns { grid-column:1/-1; display:flex; gap:8px; flex-wrap:wrap; padding:9px 11px; background:var(--surface-muted); border-radius:5px; }.dataset-columns small { color:var(--muted); }.dataset-add { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; align-items:end; margin-top:20px; }.dataset-add label { display:grid; gap:6px; font-size:13px; }.dataset-add fieldset { border:0; margin:0; padding:0; display:flex; gap:12px; flex-wrap:wrap; align-self:end; }.dataset-add fieldset label { display:flex; flex-direction:row; align-items:center; white-space:nowrap; }.dataset-add input:not([type='radio']) { width:100%; }.dataset-add button { width:max-content; align-self:end; } @media (max-width:797px) { .dataset-row,.dataset-add { grid-template-columns:1fr; }.dataset-actions { width:100%; }.dataset-actions button { flex:1; }.dataset-add button { width:100%; }.dataset-source { max-width:100%; } }
</style>
