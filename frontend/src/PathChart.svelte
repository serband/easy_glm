<script>
    import DiagnosticTable from './DiagnosticTable.svelte';
    export let rows = [],
        title = 'Regularisation path';
    const series = [
        { key: 'cv_deviance', label: 'Mean CV deviance', color: '#287762' },
        { key: 'train_deviance', label: 'Training deviance', color: '#439da5' },
        {
            key: 'n_nonzero',
            label: 'Retained coefficients · right axis',
            color: '#bd8a45',
            right: true,
        },
    ];
    $: shown = [...rows]
        .filter((r) => Number.isFinite(r.alpha) && r.alpha >= 0)
        .sort((a, b) => a.alpha - b.alpha);
    $: visible = series.filter((s) => shown.some((r) => Number.isFinite(r[s.key])));
    $: values = shown
        .flatMap((r) => [
            r.train_deviance,
            r.cv_deviance,
            Number.isFinite(r.cv_deviance) ? r.cv_deviance - (r.cv_deviance_std || 0) : null,
            Number.isFinite(r.cv_deviance) ? r.cv_deviance + (r.cv_deviance_std || 0) : null,
        ])
        .filter(Number.isFinite);
    $: min = values.length ? Math.min(...values) : 0;
    $: max = values.length ? Math.max(...values) : 1;
    $: pad = Math.max((max - min) * 0.1, Math.abs(max) * 0.001, 1e-9);
    $: low = min - pad;
    $: high = max + pad;
    $: countMax = Math.max(1, ...shown.map((r) => r.n_nonzero || 0));
    $: logs = shown.map((r) => Math.log10(r.alpha || 1));
    $: first = logs.length ? Math.min(...logs) : -1;
    $: last = logs.length ? Math.max(...logs) : 1;
    $: start = first === last ? first - 0.5 : first;
    $: end = first === last ? last + 0.5 : last;
    $: x = (alpha) => 80 + ((Math.log10(alpha || 1) - start) / (end - start)) * 590;
    $: y = (value) => 240 - ((value - low) / (high - low)) * 190;
    $: cy = (value) => 240 - (value / countMax) * 190;
    $: ticks =
        first === last
            ? [shown[0]?.alpha ?? 1]
            : Array.from({ length: 5 }, (_, i) => 10 ** (start + ((end - start) * i) / 4));
    const label = (n) =>
        n === 0 ? '0' : n.toExponential(1).replace('.0e', 'e').replace('e+', 'e');
    $: countTicks = [...new Set([0, Math.round(countMax / 2), countMax])];
</script>

<section class="diagnostic-plot path-chart">
    <h3>{title}</h3>
    <div class="chart-legend">
        {#each visible as s}<span style:color={s.color}>● {s.label}</span>{/each}<span
            >┆ Selected penalty</span
        >
    </div>
    <svg viewBox="0 0 750 305" role="img" aria-label={title}>
        <title>{title}: deviance on the left, retained coefficient count on the right</title>
        <text x="80" y="25">Deviance</text><text x="670" y="25" text-anchor="end"
            >Retained count</text
        >
        {#each [0, 0.5, 1] as tick}<line
                x1="80"
                x2="670"
                y1={240 - tick * 190}
                y2={240 - tick * 190}
                stroke="#dde5df"
            /><text x="70" y={244 - tick * 190} text-anchor="end"
                >{(low + tick * (high - low)).toPrecision(5)}</text
            >{/each}
        {#each countTicks as tick}<text class="count-tick" x="682" y={cy(tick) + 4} fill="#9a6a30"
                >{tick}</text
            >{/each}
        {#each shown.filter((r) => r.selected) as row}<line
                class="selected-penalty"
                x1={x(row.alpha)}
                x2={x(row.alpha)}
                y1="42"
                y2="240"
                stroke="#287762"
                stroke-dasharray="5 4"><title>Selected alpha: {row.alpha}</title></line
            >{/each}
        {#each visible as s}
            <path
                d={shown
                    .map((r, i) =>
                        Number.isFinite(r[s.key])
                            ? `${i === 0 || !Number.isFinite(shown[i - 1][s.key]) ? 'M' : 'L'} ${x(r.alpha)} ${s.right ? cy(r[s.key]) : y(r[s.key])}`
                            : '',
                    )
                    .join(' ')}
                fill="none"
                stroke={s.color}
                stroke-width="2"
                stroke-dasharray={s.right ? '4 4' : undefined}
            />
            {#each shown as row}{#if Number.isFinite(row[s.key])}
                    {#if s.key === 'cv_deviance' && Number.isFinite(row.cv_deviance_std)}<line
                            x1={x(row.alpha)}
                            x2={x(row.alpha)}
                            y1={y(row.cv_deviance - row.cv_deviance_std)}
                            y2={y(row.cv_deviance + row.cv_deviance_std)}
                            stroke={s.color}
                        />{/if}
                    <circle
                        cx={x(row.alpha)}
                        cy={s.right ? cy(row[s.key]) : y(row[s.key])}
                        r={row.selected ? 5 : 3}
                        fill={s.color}
                        ><title
                            >Alpha: {row.alpha} · {s.label}: {row[s.key]}{s.key === 'cv_deviance' &&
                            Number.isFinite(row.cv_deviance_std)
                                ? ' · CV standard deviation: ' + row.cv_deviance_std
                                : ''}{row.selected ? ' · selected' : ''}</title
                        ></circle
                    >
                {/if}{/each}
        {/each}
        {#each ticks as tick}<text class="alpha-tick" x={x(tick)} y="264" text-anchor="middle"
                >{label(tick)}</text
            >{/each}
        <text x="375" y="295" text-anchor="middle"
            >{shown.length === 1 && shown[0].alpha === 0
                ? 'Alpha · fixed unpenalised fit'
                : 'Alpha · logarithmic scale'}</text
        >
    </svg>
    <details><summary>Table · {title}</summary><DiagnosticTable {rows} {title} /></details>
</section>
