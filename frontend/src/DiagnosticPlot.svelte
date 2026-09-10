<script>
    import { formatNumber as num, formatLabels, axisLabel } from './format.js';
    import DiagnosticTable from './DiagnosticTable.svelte';
    export let rows = [],
        title = 'Diagnostic chart',
        series = [
            { key: 'actual_rate', label: 'Actual' },
            { key: 'expected_rate', label: 'Expected' },
        ],
        kind = 'numeric',
        xKey = 'bin',
        logX = false,
        ariaLabel = '',
        showTable = true;
    const colors = ['#287762', '#c35b48', '#737e9b', '#bd8a45', '#439da5'];
    $: shown = rows.filter((r) => r.exposure === undefined || r.exposure > 0);
    $: labels = formatLabels(shown.map((r, i) => r.label ?? r[xKey] ?? i + 1));
    $: visible = series.filter((s) => shown.some((r) => Number.isFinite(r[s.key])));
    $: values = shown.flatMap((r) => visible.map((s) => r[s.key])).filter(Number.isFinite);
    $: lo = Math.min(0, ...values);
    $: hi = Math.max(0.000001, ...values);
    $: xs = shown.map((r, i) => (logX ? Math.log10(Math.max(r[xKey], 1e-12)) : i));
    $: xmin = xs.length ? Math.min(...xs) : 0;
    $: xmax = xs.length ? Math.max(...xs) : 1;
    $: y = (v) => 190 - ((v - lo) / (hi - lo)) * 150;
    $: hasExposure = shown.some((row) => Number.isFinite(row.exposure));
    $: maxExposure = Math.max(
        0.000001,
        ...shown.map((row) => (Number.isFinite(row.exposure) ? row.exposure : 0)),
    );
    $: exposureLabels = [0, 0.5, 1].map((tick) =>
        num(tick * maxExposure, { scientific: tick * maxExposure >= 1e12 }),
    );
    $: rightGutter = Math.max(110, ...exposureLabels.map((value) => value.length * 7 + 34));
    $: plotRight = hasExposure ? 750 - rightGutter : 710;
    $: inset = Math.max(26, kind === 'categorical' ? Math.min((16 * visible.length) / 2, 40) : 0);
    $: x = (i) => {
        if (hasExposure && (shown.length === 1 || (logX && xmin === xmax)))
            return (50 + plotRight) / 2;
        const fraction = logX
            ? (xs[i] - xmin) / Math.max(1e-12, xmax - xmin)
            : i / Math.max(1, shown.length - 1);
        return hasExposure
            ? 50 + inset + fraction * (plotRight - 50 - 2 * inset)
            : 50 + fraction * 650;
    };
    $: exposureWidth = (i) => {
        const gaps = [];
        if (i > 0) gaps.push(Math.abs(x(i) - x(i - 1)));
        if (i + 1 < shown.length) gaps.push(Math.abs(x(i + 1) - x(i)));
        return Math.min(52, ...(gaps.length ? gaps.map((gap) => gap * 0.78) : [52]));
    };
</script>

<section class="diagnostic-plot">
    <h3>{title}</h3>
    <div class="chart-legend">
        {#each visible as s, i}<span style:color={s.color || colors[i % colors.length]}
                >● {s.label}</span
            >{/each}
        {#if hasExposure}<span style:color={'var(--chart-exposure-text, #805a29)'}
                >■ Exposure (right axis)</span
            >{/if}
    </div>
    <svg viewBox="0 0 750 250" role="img" aria-label={ariaLabel || title}>
        <title>{title}</title>
        {#if hasExposure}<desc
                >Exposure bars use the right axis; diagnostic series use the left axis.</desc
            >{/if}
        {#each [0, 0.5, 1] as tick}<line
                x1="50"
                x2={plotRight}
                y1={190 - tick * 150}
                y2={190 - tick * 150}
                stroke="var(--chart-grid, #e3e5e8)"
            /><text x="0" y={194 - tick * 150}>{num(lo + tick * (hi - lo))}</text>{/each}
        {#if hasExposure}
            {#each shown as row, i}{#if Number.isFinite(row.exposure)}
                    <rect
                        class="exposure-bar"
                        x={x(i) - exposureWidth(i) / 2}
                        y={190 - (row.exposure / maxExposure) * 150}
                        width={exposureWidth(i)}
                        height={(row.exposure / maxExposure) * 150}
                        fill="var(--chart-exposure, #c9a16a)"
                        fill-opacity="0.6"
                        stroke="var(--chart-exposure-edge, #9b733b)"
                        stroke-width="0.6"
                        ><title>{labels[i]}: Exposure {num(row.exposure)}</title></rect
                    >
                {/if}{/each}
            <line
                x1={plotRight}
                x2={plotRight}
                y1="40"
                y2="190"
                stroke="var(--chart-exposure-edge, #9b733b)"
            />
            {#each [0, 0.5, 1] as tick, i}
                <line
                    x1={plotRight}
                    x2={plotRight + 4}
                    y1={190 - tick * 150}
                    y2={190 - tick * 150}
                    stroke="var(--chart-exposure-edge, #9b733b)"
                /><text class="exposure-tick" x={plotRight + 8} y={194 - tick * 150}
                    >{exposureLabels[i]}</text
                >
            {/each}
            <text
                class="axis-label exposure-axis-label"
                x="740"
                y="115"
                text-anchor="middle"
                transform="rotate(90 740 115)">Exposure</text
            >
        {/if}
        {#each visible as s, j}
            {#if kind !== 'categorical'}<polyline
                    points={shown
                        .map((r, i) => (Number.isFinite(r[s.key]) ? `${x(i)},${y(r[s.key])}` : ''))
                        .join(' ')}
                    fill="none"
                    stroke={s.color || colors[j % colors.length]}
                    stroke-width="2"
                />{/if}
            {#each shown as row, i}{#if Number.isFinite(row[s.key])}
                    {#if kind === 'categorical'}{@const width = Math.min(
                            16,
                            (hasExposure ? plotRight - 50 : 500) /
                                Math.max(1, shown.length * visible.length),
                        )}<rect
                            class="diagnostic-series-bar"
                            x={x(i) + (j - visible.length / 2) * width}
                            y={Math.min(y(row[s.key]), y(0))}
                            {width}
                            height={Math.abs(y(row[s.key]) - y(0))}
                            fill={s.color || colors[j % colors.length]}
                            ><title>{labels[i]}: {s.label} {num(row[s.key])}</title></rect
                        >
                    {:else}<circle
                            cx={x(i)}
                            cy={y(row[s.key])}
                            r={row.selected ? 6 : 3}
                            fill={s.color || colors[j % colors.length]}
                            ><title
                                >{labels[i]}: {s.label}
                                {num(row[s.key])}{row.selected ? ' · selected' : ''}</title
                            ></circle
                        >{/if}
                {/if}{/each}
        {/each}
        {#each shown as row, i}{#if i % Math.max(1, Math.ceil(shown.length / 7)) === 0}<text
                    x={x(i)}
                    y="220"
                    text-anchor="middle"
                    ><title>{String(row.label ?? row[xKey] ?? i + 1)}</title>{axisLabel(
                        labels[i],
                    )}</text
                >{/if}{/each}
    </svg>
    {#if showTable}<details>
            <summary>Table · {title}</summary><DiagnosticTable {rows} {title} />
        </details>{/if}
</section>

<style>
    .diagnostic-plot svg .exposure-tick,
    .diagnostic-plot svg .exposure-axis-label {
        fill: var(--chart-exposure-text, #805a29);
    }
</style>
