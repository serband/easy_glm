<script>
    import { formatNumber as num, formatLabels, axisLabel } from './format.js';
    import { rateChartData } from './rateChartData.js';
    export let table,
        variable,
        label = 'relativity',
        fittedLabel = 'Original fit',
        currentLabel = 'Adjusted',
        preview = false;
    let cellView = 'relativity';
    $: plot = rateChartData(table);
    $: displayLabels = formatLabels((table?.rows || []).map((r) => r.label));
    $: rowNames = [...new Set((table?.rows || []).map((r) => r.label_a))];
    $: colNames = [...new Set((table?.rows || []).map((r) => r.label_b))];

    $: rowLabels = formatLabels(rowNames);
    $: colLabels = formatLabels(colNames);
    function path(points) {
        return points.map((p) => `${p.x},${190 - (p.value / plot.max) * 150}`).join(' ');
    }
    function color(row) {
        if (!row || !row.exposure) return '#eef2ef';
        const value = row[cellView];
        if (cellView === 'exposure')
            return `rgba(70,117,100,${0.1 + (0.8 * value) / plot.maxExposure})`;
        return value >= 1
            ? `rgba(195,91,72,${Math.min(0.8, 0.15 + Math.abs(Math.log(value)))})`
            : `rgba(40,119,98,${Math.min(0.8, 0.15 + Math.abs(Math.log(value)))})`;
    }
</script>

<section class="rate-chart-card" aria-label="Relativity chart">
    <h2>{variable} · {label}</h2>
    {#if plot.interaction}
        <label
            >Cell values<select aria-label="Interaction chart values" bind:value={cellView}
                ><option value="relativity">{currentLabel} relativity</option><option value="fitted"
                    >{fittedLabel} relativity</option
                ><option value="exposure">Exposure</option></select
            ></label
        >
        <div class="heatmap-scroll">
            <table class="relativity-heatmap">
                <thead
                    ><tr
                        ><th></th>{#each colNames as name, i}<th title={name}>{colLabels[i]}</th
                            >{/each}</tr
                    ></thead
                ><tbody
                    >{#each rowNames as name, i}<tr
                            ><th title={name}>{rowLabels[i]}</th
                            >{#each colNames as other}{@const cell = table.rows.find(
                                    (r) => r.label_a === name && r.label_b === other,
                                )}<td
                                    style:background={color(cell)}
                                    title={`${fittedLabel} ${num(cell?.fitted)}; ${currentLabel} ${num(cell?.relativity)}; exposure ${num(cell?.exposure)}`}
                                    >{cell?.exposure ? num(cell[cellView]) : '—'}</td
                                >{/each}</tr
                        >{/each}</tbody
                >
            </table>
        </div>
        <p class="help-text">
            Cells without exposure are blank. Choose exposure to see their support.
        </p>
    {:else}
        <div class="rate-plot-scroll">
            <div
                style:min-width={plot.numeric ? '0' : Math.max(500, plot.points.length * 44) + 'px'}
            >
                <div class="chart-legend">
                    <span style:color={'#737e9b'}>● {fittedLabel}</span><span
                        style:color={'#287762'}>● {currentLabel}</span
                    >
                </div>
                <svg
                    class="relativity-chart"
                    viewBox={plot.numeric ? '0 0 740 240' : '0 0 740 310'}
                    role="img"
                    aria-label={(preview
                        ? 'Original fit and adjusted preview for '
                        : 'Original fit and adjusted relativities for ') + variable}
                >
                    <title>Original fit and adjusted {label} by band or level</title>
                    {#each [0, 0.5, 1] as tick}<line
                            x1="55"
                            x2="705"
                            y1={190 - tick * 150}
                            y2={190 - tick * 150}
                            stroke="#e1e9e4"
                        /><text x="0" y={194 - tick * 150}>{num(plot.max * tick)}</text>{/each}
                    <line
                        x1="55"
                        x2="705"
                        y1={190 - 150 / plot.max}
                        y2={190 - 150 / plot.max}
                        stroke="#83928b"
                        stroke-dasharray="3 3"
                    />
                    {#if plot.numeric}
                        {#each ['fitted', 'current'] as field}
                            {#each plot.lines[field] as line}<polyline
                                    class="numeric-curve"
                                    points={path(line)}
                                    fill="none"
                                    stroke={field === 'fitted' ? '#737e9b' : '#287762'}
                                    stroke-width={field === 'fitted' ? 4 : 2}
                                />{/each}
                            {#each plot.points.filter((item) => item.missing || item[field].length === 1) as item}
                                {#each item[field] as point}<circle
                                        cx={point.x}
                                        cy={190 - (point.value / plot.max) * 150}
                                        r={field === 'fitted' ? 5 : 3}
                                        fill={field === 'fitted' ? '#737e9b' : '#287762'}
                                    >
                                        <title>{item.label}: {field} {num(point.value)}</title>
                                    </circle>{/each}
                            {/each}
                        {/each}
                    {:else}
                        {#each plot.points as item}
                            {#each ['fitted', 'current'] as field, index}
                                {#each item[field] as point}{@const width = Math.min(
                                        20,
                                        250 / plot.points.length,
                                    )}
                                    <rect
                                        class="category-bar"
                                        x={item.x + (index - 1) * width}
                                        y={190 - (point.value / plot.max) * 150}
                                        {width}
                                        height={(point.value / plot.max) * 150}
                                        fill={field === 'fitted' ? '#737e9b' : '#287762'}
                                    >
                                        <title>{item.label}: {field} {num(point.value)}</title>
                                    </rect>
                                {/each}
                            {/each}
                        {/each}
                    {/if}
                    {#each plot.points as item, i}{#if !plot.numeric || i % Math.max(1, Math.ceil(plot.points.length / 5)) === 0 || i === plot.points.length - 1}<text
                                x={item.x}
                                y="222"
                                text-anchor={plot.numeric ? 'middle' : 'end'}
                                transform={plot.numeric ? undefined : `rotate(-45 ${item.x} 222)`}
                                ><title>{item.label}</title>{plot.numeric && !plot.linear
                                    ? displayLabels[item.index]
                                    : axisLabel(displayLabels[item.index])}</text
                            >{/if}{/each}
                </svg>
                <div class="exposure-caption">
                    Training exposure by band / level · largest {num(plot.maxExposure)}
                </div>
                <svg
                    class="exposure-chart"
                    viewBox="0 0 740 85"
                    role="img"
                    aria-label={'Exposure for ' + variable}
                    ><title>Exposure aligned with the relativity bands</title><line
                        x1="55"
                        x2="705"
                        y1="72"
                        y2="72"
                        stroke="#d8e2dc"
                    />{#each plot.points as item}<rect
                            x={item.x - 5}
                            y={72 - ((item.row.exposure || 0) / plot.maxExposure) * 60}
                            width="10"
                            height={((item.row.exposure || 0) / plot.maxExposure) * 60}
                            fill="#91afa1"
                            ><title>{item.label}: exposure {num(item.row.exposure)}</title></rect
                        >{/each}</svg
                >
            </div>
        </div>
        {#if plot.linear}<p class="help-text">
                Curves follow the exported log slopes; open end bands stay flat. Fitted endpoints
                use the original fitted values.
            </p>{:else if plot.numeric}<p class="help-text">
                Points show each band’s relativity; lines connect them to show the trend. Scoring
                uses the value within each band.
            </p>{/if}
    {/if}
    {#if table.total > table.rows.length}<p class="help-text">
            Chart covers the displayed page, rows {table.offset + 1}–{table.offset +
                table.rows.length} of {table.total}.
        </p>{/if}
</section>
