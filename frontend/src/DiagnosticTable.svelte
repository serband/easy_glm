<script>
    import { formatNumber as num, formatLabels } from './format.js';
    export let rows = [],
        title = 'Table';
    let page = 0;
    $: columns = Object.keys(rows[0] || {});
    $: if (page * 100 >= rows.length) page = 0;
    $: labels = Object.fromEntries(
        columns
            .filter((c) => c === 'label' || c === 'label_a' || c === 'label_b' || c === 'band')
            .map((c) => [c, formatLabels(rows.map((r) => r[c]))]),
    );
    function download() {
        const quote = (value) => {
            const v = typeof value === 'string' && /^[=+\-@]/.test(value) ? "'" + value : value;
            return '"' + String(v ?? '').replaceAll('"', '""') + '"';
        };
        const csv = [columns, ...rows.map((r) => columns.map((c) => r[c]))]
            .map((r) => r.map(quote).join(','))
            .join('\n');
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
        const a = document.createElement('a');
        a.href = url;
        a.download = title.replaceAll(/[^a-z0-9_-]/gi, '_') + '.csv';
        a.click();
        URL.revokeObjectURL(url);
    }
</script>

<div class="diagnostic-table">
    <div class="results-toolbar">
        <strong>{title}</strong><span>{rows.length} rows</span><button
            onclick={download}
            disabled={!rows.length}>Download CSV</button
        >
    </div>
    {#if rows.length}<div class="review-scroll">
            <table>
                <thead
                    ><tr
                        >{#each columns as c}<th>{c.replaceAll('_', ' ')}</th>{/each}</tr
                    ></thead
                ><tbody
                    >{#each rows.slice(page * 100, (page + 1) * 100) as row, i}<tr
                            >{#each columns as c}<td title={labels[c] ? String(row[c]) : undefined}
                                    >{labels[c]?.[page * 100 + i] ?? num(row[c])}</td
                                >{/each}</tr
                        >{/each}</tbody
                >
            </table>
        </div>
        {#if rows.length > 100}<div class="results-toolbar">
                <button disabled={!page} onclick={() => page--}>Previous page</button><span
                    >Page {page + 1} of {Math.ceil(rows.length / 100)}</span
                ><button disabled={(page + 1) * 100 >= rows.length} onclick={() => page++}
                    >Next page</button
                >
            </div>{/if}
    {:else}<p class="help-text">No rows for this selection.</p>{/if}
</div>
