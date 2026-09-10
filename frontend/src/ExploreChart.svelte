<script>
    import { onMount } from 'svelte';
    import { formatNumber as num, formatLabels, axisLabel } from './format.js';
    export let rows = [],
        column = '',
        kind = 'numeric',
        rateLabel = 'Observed rate',
        exposureLabel = 'Exposure',
        hasTarget = true;
    $: shown =
        kind === 'numeric'
            ? [...rows].sort(
                  (a, b) =>
                      (Number.isFinite(a.order) ? a.order : Infinity) -
                      (Number.isFinite(b.order) ? b.order : Infinity),
              )
            : rows;
    $: labels = formatLabels(shown.map((row) => row.label));
    $: rates = hasTarget ? shown.map((row) => row.rate).filter(Number.isFinite) : [];
    $: low = Math.min(0, ...rates);
    $: high = Math.max(0, ...rates);
    $: span = high > low ? high - low : 1;
    $: maxExposure =
        Math.max(0, ...shown.map((row) => (Number.isFinite(row.exposure) ? row.exposure : 0))) || 1;
    $: leftTicks = [0, 0.5, 1].map((part) => num(low + part * span));
    $: rightTicks = [0, 0.5, 1].map((part) => num(part * maxExposure));
    $: left = hasTarget ? Math.max(62, ...leftTicks.map((label) => 24 + label.length * 7)) : 35;
    $: right = 800 - Math.max(100, ...rightTicks.map((label) => 37 + label.length * 7));
    $: step = (right - left) / Math.max(1, shown.length);
    $: x = (i) => left + step * (i + 0.5);
    $: y = (rate) => 232 - ((rate - low) / span) * 180;
    $: barWidth = Math.min(54, step * 0.7);
    $: line = ratePath(shown, kind, hasTarget, x, y);
    let svgElement;
    let labelWidth = (label) => tickLabel(label).length * 8;
    $: tickIndexes = spacedTicks(labels, x, labelWidth, (right - left) / 6);
    onMount(() => {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        if (context) {
            context.font = '12px ' + getComputedStyle(svgElement).fontFamily;
            labelWidth = (label) => context.measureText(tickLabel(label)).width + 4;
        }
    });
    function spacedTicks(values, position, width, preferredGap) {
        if (values.length < 2) return new Set(values.map((_, i) => i));
        const last = values.length - 1,
            chosen = [0];
        const lastLeft = position(last) - width(values[last]) / 2;
        let previousRight = position(0) + width(values[0]) / 2;
        for (let i = 1; i < last; i += 1) {
            const half = width(values[i]) / 2;
            if (
                position(i) - position(chosen[chosen.length - 1]) < preferredGap ||
                position(i) - half < previousRight + 12 ||
                position(i) + half > lastLeft - 12
            )
                continue;
            chosen.push(i);
            previousRight = position(i) + half;
        }
        chosen.push(last);
        return new Set(chosen);
    }
    function ratePath(data, type, target, xPos, yPos) {
        if (type !== 'numeric' || !target) return '';
        let connected = false;
        return data
            .map((row, i) => {
                const valid =
                    Number.isFinite(row.rate) &&
                    Number.isFinite(row.order) &&
                    row.label !== 'Other / Unknown';
                if (!valid) {
                    connected = false;
                    return '';
                }
                const point = `${connected ? 'L' : 'M'}${xPos(i)},${yPos(row.rate)}`;
                connected = true;
                return point;
            })
            .join(' ');
    }
    function tickLabel(label) {
        const compact = axisLabel(label);
        return compact.length > 15 ? compact.slice(0, 14) + '…' : compact;
    }
</script>

<div class="one-way-chart">
    <div class="legend">
        {#if hasTarget}<span class="rate-key">● {rateLabel} (left axis)</span>{/if}
        <span class="exposure-key">■ {exposureLabel} (right axis)</span>
    </div>
    <svg
        bind:this={svgElement}
        viewBox="0 0 800 295"
        role="img"
        aria-label={'One-way effects of ' + column}
    >
        <title>One-way effects of {column}</title>
        <desc
            >{!hasTarget
                ? 'Exposure only; no target is assigned.'
                : kind === 'numeric'
                  ? 'Observed rates follow numeric bands; missing values are separate.'
                  : 'Observed rates are separate points for each category.'}
            {exposureLabel} bars use the right axis.</desc
        >
        {#each [0, 0.5, 1] as part, i}
            <line
                x1={left}
                x2={right}
                y1={232 - part * 180}
                y2={232 - part * 180}
                stroke="var(--chart-grid, #e3e5e8)"
            />
            {#if hasTarget}<text
                    class="rate-tick"
                    x={left - 9}
                    y={236 - part * 180}
                    text-anchor="end">{leftTicks[i]}</text
                >{/if}
            <text class="exposure-tick" x={right + 9} y={236 - part * 180}>{rightTicks[i]}</text>
        {/each}
        <text class="axis-title" x={left} y="26">{hasTarget ? 'Observed response' : ''}</text>
        <text class="axis-title exposure-axis-label" x={right} y="26" text-anchor="end"
            >{exposureLabel}</text
        >
        <line x1={right} x2={right} y1="52" y2="232" stroke="var(--chart-exposure-edge, #9b733b)" />
        {#each shown as row, i}
            {#if Number.isFinite(row.exposure)}
                <rect
                    class="exposure-bar"
                    x={x(i) - barWidth / 2}
                    y={232 - (row.exposure / maxExposure) * 180}
                    width={barWidth}
                    height={(row.exposure / maxExposure) * 180}
                    fill="var(--chart-exposure, #c9a16a)"
                    fill-opacity="0.6"
                    stroke="var(--chart-exposure-edge, #9b733b)"
                    stroke-width="0.6"
                >
                    <title
                        >{row.label}: {exposureLabel}
                        {num(row.exposure)}{hasTarget
                            ? ' · ' + rateLabel + ' ' + num(row.rate)
                            : ''}</title
                    >
                </rect>
            {/if}
        {/each}
        {#if line}<path
                class="observed-line"
                d={line}
                fill="none"
                stroke="var(--chart-observed, #22735e)"
                stroke-width="2.4"
            />{/if}
        {#if hasTarget}
            {#each shown as row, i}{#if Number.isFinite(row.rate)}
                    <circle
                        class="observed-point"
                        cx={x(i)}
                        cy={y(row.rate)}
                        r="4"
                        fill="var(--chart-observed, #22735e)"
                        stroke="white"
                        stroke-width="1"
                    >
                        <title
                            >{row.label}: {rateLabel}
                            {num(row.rate)} · {exposureLabel}
                            {num(row.exposure)}</title
                        >
                    </circle>
                {/if}{/each}
        {/if}
        {#each shown as row, i}
            {#if tickIndexes.has(i)}
                <text class="band-label" x={x(i)} y="259" text-anchor="middle"
                    ><title>{row.label}</title>{tickLabel(labels[i])}</text
                >
            {/if}
        {/each}
    </svg>
</div>

<style>
    .legend {
        display: flex;
        flex-wrap: wrap;
        gap: 12px 22px;
        font-size: 14px;
        margin-bottom: 6px;
    }
    .rate-key {
        color: var(--chart-observed, #22735e);
    }
    .exposure-key {
        color: var(--chart-exposure-text, #805a29);
    }
    svg {
        display: block;
        width: 100%;
        height: auto;
        overflow: visible;
    }
    svg text {
        fill: var(--muted, #616a75);
        font-size: 12px;
    }
    svg .exposure-tick,
    svg .exposure-axis-label {
        fill: var(--chart-exposure-text, #805a29);
    }
    svg .axis-title {
        font-size: 13px;
    }
</style>
