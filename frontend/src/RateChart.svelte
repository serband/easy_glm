<script>
    import { rateChartData } from './rateChartData.js';
    export let table,
        variable,
        label = 'relativity';
    let cellView = 'relativity';
    $: plot = rateChartData(table);
    $: rowNames = [...new Set((table?.rows || []).map((r) => r.label_a))];
    $: colNames = [...new Set((table?.rows || []).map((r) => r.label_b))];
    function num(v) {
        return Number.isFinite(v)
            ? v.toLocaleString(undefined, { maximumSignificantDigits: 5 })
            : '—';
    }
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
                ><option value="relativity">Current relativity</option><option value="fitted"
                    >Fitted relativity</option
                ><option value="exposure">Exposure</option></select
            ></label
        >
        <div class="heatmap-scroll">
            <table class="relativity-heatmap">
                <thead
                    ><tr
                        ><th></th>{#each colNames as name}<th>{name}</th>{/each}</tr
                    ></thead
                ><tbody
                    >{#each rowNames as name}<tr
                            ><th>{name}</th>{#each colNames as other}{@const cell = table.rows.find(
                                    (r) => r.label_a === name && r.label_b === other,
                                )}<td
                                    style:background={color(cell)}
                                    title={`Fitted ${num(cell?.fitted)}; current ${num(cell?.relativity)}; exposure ${num(cell?.exposure)}`}
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
        <div class="chart-legend">
            <span style:color={'#737e9b'}>● Fitted</span><span style:color={'#287762'}
                >● Current</span
            >
        </div>
        <svg
            class="relativity-chart"
            viewBox="0 0 740 240"
            role="img"
            aria-label={'Fitted and current relativities for ' + variable}
        >
            <title>Canonical fitted and current {label} by band or level</title>
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
            {#each plot.points as item}{#each ['fitted', 'current'] as field}<polyline
                        points={path(item[field])}
                        fill="none"
                        stroke={field === 'fitted' ? '#737e9b' : '#287762'}
                        stroke-width={field === 'fitted' ? 4 : 2}
                    />{#if item[field].length === 1}<circle
                            cx={item[field][0].x}
                            cy={190 - (item[field][0].value / plot.max) * 150}
                            r={field === 'fitted' ? 5 : 3}
                            fill={field === 'fitted' ? '#737e9b' : '#287762'}
                            ><title>{item.label}: {field} {num(item[field][0].value)}</title
                            ></circle
                        >{/if}{/each}{/each}
            {#each plot.points as item, i}{#if i % Math.max(1, Math.ceil(plot.points.length / 5)) === 0 || i === plot.points.length - 1}<text
                        x={item.x}
                        y="222"
                        text-anchor="middle">{String(item.label).slice(0, 17)}</text
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
        {#if plot.linear}<p class="help-text">
                Curves follow the exported log slopes; open end bands stay flat. Fitted endpoints
                use the original fitted values.
            </p>{/if}
    {/if}
    {#if table.total > table.rows.length}<p class="help-text">
            Chart covers the displayed page, rows {table.offset + 1}–{table.offset +
                table.rows.length} of {table.total}.
        </p>{/if}
</section>
