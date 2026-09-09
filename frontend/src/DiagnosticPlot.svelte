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
    $: x = (i) =>
        50 +
        (logX ? (xs[i] - xmin) / Math.max(1e-12, xmax - xmin) : i / Math.max(1, shown.length - 1)) *
            650;
    $: maxExposure = Math.max(1, ...shown.map((r) => r.exposure || 0));
</script>

<section class="diagnostic-plot">
    <h3>{title}</h3>
    <div class="chart-legend">
        {#each visible as s, i}<span style:color={s.color || colors[i % colors.length]}
                >● {s.label}</span
            >{/each}
    </div>
    <svg viewBox="0 0 750 250" role="img" aria-label={ariaLabel || title}>
        <title>{title}</title>
        {#each [0, 0.5, 1] as tick}<line
                x1="50"
                x2="710"
                y1={190 - tick * 150}
                y2={190 - tick * 150}
                stroke="#dde5df"
            /><text x="0" y={194 - tick * 150}>{num(lo + tick * (hi - lo))}</text>{/each}
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
                            500 / Math.max(1, shown.length * visible.length),
                        )}<rect
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
    {#if shown.some((r) => r.exposure !== undefined)}<div class="exposure-caption">
            Exposure by group
        </div>
        <svg viewBox="0 0 750 85" role="img" aria-label={'Exposure · ' + title}
            >{#each shown as row, i}<rect
                    x={x(i) - 5}
                    y={75 - ((row.exposure || 0) / maxExposure) * 65}
                    width="10"
                    height={((row.exposure || 0) / maxExposure) * 65}
                    fill="#91afa1"><title>{labels[i]}: {num(row.exposure)}</title></rect
                >{/each}</svg
        >{/if}
    {#if showTable}<details>
            <summary>Table · {title}</summary><DiagnosticTable {rows} {title} />
        </details>{/if}
</section>
