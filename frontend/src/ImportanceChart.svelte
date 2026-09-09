<script>
    import { formatNumber as num } from './format.js';
    import { importanceChartData } from './importanceChartData.js';
    import DiagnosticTable from './DiagnosticTable.svelte';
    export let result;
    $: plot = importanceChartData(result.rows || []);
    $: left = Math.min(
        230,
        Math.max(115, ...plot.rows.map((row) => String(row.variable).length * 7 + 16)),
    );
    const right = 685;
    $: height = 70 + plot.rows.length * 32;
    $: x = (value) => left + ((value - plot.low) / (plot.high - plot.low)) * (right - left);
    $: ticks = [...new Set([plot.low, 0, plot.high])]
        .filter((tick) => tick === 0 || Math.abs(x(tick) - x(0)) > (num(tick).length + 1) * 3 + 10)
        .sort((a, b) => a - b);
</script>

<div class="importance-view">
    <p class="help-text importance-context">
        Training · Original fit · {result.repeats || 5} shuffles per variable
    </p>
    {#if plot.rows.length}
        <p class="importance-axis">Mean deviance increase <span>· whiskers ±1 SD</span></p>
        <div class="importance-scroll">
            <svg
                viewBox={'0 0 750 ' + height}
                role="img"
                aria-label="Training permutation importance by variable"
            >
                <title
                    >Permutation importance: increase in training mean deviance, ranked largest
                    first</title
                >
                {#each ticks as tick}
                    <line
                        x1={x(tick)}
                        x2={x(tick)}
                        y1="12"
                        y2={height - 35}
                        stroke={tick === 0 ? '#789187' : '#e0e8e3'}
                        stroke-width={tick === 0 ? 1.5 : 1}
                    />
                    <text class="axis-tick" x={x(tick)} y={height - 15} text-anchor="middle"
                        >{num(tick)}</text
                    >
                {/each}
                {#each plot.rows as row, i}
                    {@const cy = 27 + i * 32}
                    <g class="importance-row" data-variable={row.variable}>
                        <text class="variable-label" x={left - 12} y={cy + 4} text-anchor="end">
                            <title>{row.variable}</title>
                            {String(row.variable).length > 29
                                ? String(row.variable).slice(0, 27) + '…'
                                : row.variable}
                        </text>
                        <rect
                            class="importance-bar"
                            data-importance={row.importance}
                            x={Math.min(x(0), x(row.importance))}
                            y={cy - 9}
                            width={Math.abs(x(row.importance) - x(0))}
                            height="18"
                            rx="2"
                            fill={row.importance >= 0 ? '#287762' : '#737e9b'}
                        >
                            <title
                                >{row.variable}: mean deviance increase {num(row.importance)}; SD {num(
                                    row.std,
                                )}</title
                            >
                        </rect>
                        {#if row.importance === 0}<circle
                                cx={x(0)}
                                {cy}
                                r="2.5"
                                fill="#287762"
                            />{/if}
                        {#if row.std > 0}
                            <path
                                d={`M${x(row.importance - row.std)},${cy}H${x(row.importance + row.std)} M${x(row.importance - row.std)},${cy - 4}V${cy + 4} M${x(row.importance + row.std)},${cy - 4}V${cy + 4}`}
                                fill="none"
                                stroke="#263e36"
                                stroke-width="1.2"
                            >
                                <title>±1 standard deviation: {num(row.std)}</title>
                            </path>
                        {/if}
                        <text class="importance-value" x="740" y={cy + 4} text-anchor="end"
                            >{num(row.importance)}</text
                        >
                    </g>
                {/each}
            </svg>
        </div>
        <details>
            <summary>Values</summary><DiagnosticTable
                rows={result.rows}
                title="Permutation importance"
            />
        </details>
    {:else}<p class="help-text">No fitted source predictors to shuffle.</p>{/if}
</div>

<style>
    .importance-context {
        margin: 0 0 14px;
    }
    .importance-axis {
        margin: 0 0 6px;
        font-weight: 600;
        font-size: 13px;
    }
    .importance-axis span {
        color: #73877d;
        font-weight: 400;
    }
    .importance-scroll {
        max-height: 65vh;
        overflow-y: auto;
    }
    svg {
        display: block;
        width: 100%;
        min-height: 90px;
    }
    text {
        font-family: inherit;
        font-size: 11px;
        fill: #516b60;
    }
    .variable-label {
        font-weight: 500;
    }
    .importance-value {
        font-variant-numeric: tabular-nums;
        fill: #283d34;
    }
    details {
        margin-top: 12px;
    }
</style>
