<script>
    import { onDestroy } from 'svelte';
    export let api,
        state,
        context,
        comparison = '',
        downloadProject,
        saveDownload;
    let selected = '',
        challenger = '',
        pending = '',
        error = '',
        alive = true;
    $: names = context?.fitted || [];
    $: if (!names.includes(selected)) {
        selected = names.includes(context?.selected)
            ? context.selected
            : names.includes(context?.champion)
              ? context.champion
              : names[0] || '';
    }
    $: if (challenger === selected || !names.includes(challenger)) challenger = '';
    let defaultComparison = null;
    $: if (comparison !== defaultComparison) {
        defaultComparison = comparison;
        challenger = comparison !== selected && names.includes(comparison) ? comparison : '';
    }
    async function download(format) {
        if (pending) return;
        pending = format;
        error = '';
        const model = selected;
        const reportChallenger = challenger;
        try {
            if (format === 'project') {
                await downloadProject();
            } else {
                const current = await api('workbench');
                if (!current.jobs[model]?.applicable)
                    throw new Error('Fit this model before exporting it.');
                const file = await api(
                    'exports/' + encodeURIComponent(model),
                    {
                        session_id: current.session_id,
                        revision: current.revision,
                        format,
                        challenger: format === 'html' ? reportChallenger || null : null,
                    },
                    true,
                );
                if (alive) saveDownload(file.blob, file.filename);
            }
        } catch (e) {
            if (alive) error = e.message;
        } finally {
            if (alive) pending = '';
        }
    }
    onDestroy(() => (alive = false));
</script>

<section class="workflow-page export-page">
    <div class="heading">
        <div>
            <h1>Export</h1>
            <p>Download your model, tables and project.</p>
        </div>
    </div>
    {#if error}<div class="message error" role="alert">{error}</div>{/if}
    {#if names.length}
        <label class="export-model"
            >Model<select aria-label="Export model" bind:value={selected} disabled={!!pending}>
                {#each names as name}<option value={name}>{name}</option>{/each}
            </select></label
        >
        <section class="model-card" aria-label="Fitted model exports">
            <h2>Fitted model</h2>
            <p class="help-text">Includes applied adjustments.</p>
            <div class="export-row">
                <div>
                    <h3>Excel rate tables</h3>
                    <p>One worksheet per rating factor.</p>
                </div>
                <button onclick={() => download('xlsx')} disabled={!!pending}>
                    {pending === 'xlsx' ? 'Preparing Excel…' : 'Download Excel (.xlsx)'}
                </button>
            </div>
            <div class="export-row">
                <div>
                    <h3>Scorer</h3>
                    <p>JSON model for scoring new data.</p>
                </div>
                <button onclick={() => download('easyglm')} disabled={!!pending}>
                    {pending === 'easyglm' ? 'Preparing scorer…' : 'Download scorer (.easyglm)'}
                </button>
            </div>
            <div class="export-row">
                <div>
                    <h3>HTML report</h3>
                    <p>Diagnostics, rate tables and model comparison.</p>
                    {#if names.length > 1}<label class="report-comparison"
                            >Compare with<select
                                aria-label="Report comparison model"
                                bind:value={challenger}
                                disabled={!!pending}
                            >
                                <option value="">None</option>
                                {#each names.filter((name) => name !== selected) as name}<option
                                        value={name}>{name}</option
                                    >{/each}
                            </select></label
                        >{/if}
                </div>
                <button onclick={() => download('html')} disabled={!!pending}>
                    {pending === 'html' ? 'Preparing report…' : 'Download report (.html)'}
                </button>
            </div>
        </section>
    {:else}
        <p class="export-empty">Fit a model to export its rate tables, scorer and report.</p>
    {/if}
    <section class="model-card" aria-label="Project exports">
        <h2>Project and script</h2>
        <div class="export-row">
            <div>
                <h3>Project JSON</h3>
                <p>Settings, applied adjustments and named snapshots.</p>
            </div>
            <button onclick={() => download('project')} disabled={!state || !!pending}>
                {pending === 'project' ? 'Preparing project…' : 'Download project JSON'}
            </button>
        </div>
        {#if names.length}<div class="export-row">
                <div>
                    <h3>Python script</h3>
                    <p>Rebuild the model from its source data.</p>
                </div>
                <button onclick={() => download('python')} disabled={!!pending}>
                    {pending === 'python' ? 'Preparing script…' : 'Download script (.py)'}
                </button>
            </div>{/if}
        <p class="help-text">
            Project JSON saves the setup. Use the scorer to keep the fitted rates.
        </p>
    </section>
</section>

<style>
    .export-model {
        display: grid;
        gap: 6px;
        max-width: 320px;
        margin: 0 0 20px;
    }
    .export-page .model-card {
        margin-bottom: 20px;
    }
    .export-page h2 {
        margin: 0 0 6px;
    }
    .export-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: center;
        gap: 20px;
        border-top: 1px solid #dce5df;
        padding: 18px 0;
    }
    .export-row:first-of-type {
        margin-top: 12px;
    }
    .export-row h3 {
        margin: 0;
        font-size: 15px;
    }
    .export-row p {
        margin: 5px 0 0;
        font-size: 12px;
        color: #6b8275;
    }
    .export-row button {
        font-size: 12px;
        white-space: nowrap;
    }
    .report-comparison {
        display: grid;
        gap: 5px;
        margin-top: 10px;
        max-width: 260px;
        font-size: 12px;
    }
    .export-empty {
        margin: 0 0 24px;
    }
    @media (max-width: 680px) {
        .export-row {
            grid-template-columns: minmax(0, 1fr);
            gap: 12px;
        }
        .export-row button {
            justify-self: start;
        }
    }
</style>
