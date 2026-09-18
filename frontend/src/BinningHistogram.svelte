<script>
    export let intervals = [];
    export let name = '';

    const left = 49;
    const right = 16;
    const top = 25;
    const plotHeight = 174;
    const height = 292;

    $: bars = Array.isArray(intervals)
        ? intervals.filter((row) => Number.isFinite(row.rows) && row.rows >= 0)
        : [];
    $: width = Math.max(640, left + bars.length * 84 + right);
    $: plotWidth = width - left - right;
    $: slot = plotWidth / Math.max(1, bars.length);
    $: tallest = Math.max(1, ...bars.map((bar) => bar.rows));
    $: totalRows = bars.reduce((total, bar) => total + bar.rows, 0);

    function y(rows) {
        return top + plotHeight * (1 - rows / tallest);
    }
    function count(value) {
        return Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 });
    }
</script>

<figure class="binning-histogram" aria-label="Training distribution">
    <figcaption>
        <strong>Training distribution</strong>
        <span>One bar per model bin, showing training rows in that interval.</span>
    </figcaption>
    {#if !bars.length}
        <p>No numeric model bins to plot.</p>
    {:else}
        <div class="chart-scroll">
            <svg
                viewBox={`0 0 ${width} ${height}`}
                style:width={bars.length > 8 ? `${width}px` : '100%'}
                role="img"
                aria-label={`Training rows by model bin for ${name}: ${totalRows.toLocaleString()} rows in ${bars.length} bins`}
            >
                <title>Training rows by model bin for {name}</title>
                <desc>Each bar shows the exact training-row count for one model interval.</desc>
                <text class="axis-label" x={left} y={14}>Rows</text>
                <line class="axis" x1={left} y1={top} x2={left} y2={top + plotHeight} />
                <line
                    class="axis"
                    x1={left}
                    y1={top + plotHeight}
                    x2={left + plotWidth}
                    y2={top + plotHeight}
                />
                {#each [0, tallest / 2, tallest] as value}
                    <line
                        class="grid"
                        x1={left}
                        x2={left + plotWidth}
                        y1={y(value)}
                        y2={y(value)}
                    />
                    <text class="count-tick" x={left - 7} y={y(value) + 4} text-anchor="end"
                        >{count(value)}</text
                    >
                {/each}
                {#each bars as bar, index}
                    {@const centre = left + slot * (index + 0.5)}
                    <rect
                        class="distribution-bar"
                        data-lower={bar.lower == null ? undefined : bar.lower}
                        data-upper={bar.upper == null ? undefined : bar.upper}
                        data-rows={bar.rows}
                        x={centre - slot * 0.37}
                        y={y(bar.rows)}
                        width={slot * 0.74}
                        height={top + plotHeight - y(bar.rows)}
                    >
                        <title>{bar.label}: {bar.rows} rows</title>
                    </rect>
                    <text
                        class="value-tick"
                        x={centre}
                        y={top + plotHeight + 17}
                        text-anchor="end"
                        transform={`rotate(-42 ${centre} ${top + plotHeight + 17})`}
                        >{bar.label}</text
                    >
                {/each}
            </svg>
        </div>
    {/if}
</figure>

<style>
    .binning-histogram {
        width: 100%;
        min-width: 0;
        margin: 16px 0 0;
        padding: 12px;
        border: 1px solid #e2e8ec;
        border-radius: 7px;
        background: #fbfcfd;
    }
    figcaption {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 5px 10px;
        color: #40556a;
        font-size: 12px;
    }
    figcaption strong {
        font-size: 13px;
        color: #183b61;
    }
    .binning-histogram p {
        margin: 16px 0 4px;
        color: #566875;
        font-size: 12px;
    }
    .chart-scroll {
        max-width: 100%;
        overflow-x: auto;
        overflow-y: hidden;
    }
    svg {
        display: block;
        height: auto;
        margin-top: 9px;
    }
    .distribution-bar {
        fill: #4b82b6;
    }
    .distribution-bar:hover {
        fill: #1c5f9e;
    }
    .axis {
        stroke: #607486;
        stroke-width: 1;
    }
    .grid {
        stroke: #dce5ec;
        stroke-width: 1;
    }
    text {
        fill: #465e73;
        font-size: 11px;
    }
</style>
