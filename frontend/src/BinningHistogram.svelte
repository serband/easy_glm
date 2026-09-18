<script>
    export let histogram;
    export let intervals = [];
    export let name = '';

    const width = 640;
    const height = 260;
    const left = 48;
    const right = 14;
    const top = 22;
    const bottom = 39;
    const plotWidth = width - left - right;
    const plotHeight = height - top - bottom;

    $: bars = Array.isArray(histogram?.bars)
        ? histogram.bars.filter(
              (bar) =>
                  Number.isFinite(bar.lower) &&
                  Number.isFinite(bar.upper) &&
                  bar.upper > bar.lower &&
                  Number.isFinite(bar.rows) &&
                  bar.rows >= 0,
          )
        : [];
    $: finiteRows = histogram?.finite_rows ?? 0;
    $: minimum = bars.length ? Math.min(...bars.map((bar) => bar.lower)) : 0;
    $: maximum = bars.length ? Math.max(...bars.map((bar) => bar.upper)) : 1;
    $: range = maximum - minimum;
    $: scale = Math.max(Math.abs(minimum), Math.abs(maximum), 1);
    $: scaledMinimum = minimum / scale;
    $: scaledRange = maximum / scale - scaledMinimum;
    $: tallest = Math.max(1, ...bars.map((bar) => bar.rows));
    $: ticks = Array.from({ length: 5 }, (_, index) =>
        Number.isFinite(range)
            ? minimum + (range * index) / 4
            : (scaledMinimum + (scaledRange * index) / 4) * scale,
    );
    $: boundaries = [
        ...new Set(
            intervals
                .flatMap((interval) => [interval.lower, interval.upper])
                .filter((value) => typeof value === 'number' && Number.isFinite(value)),
        ),
    ].sort((a, b) => a - b);
    $: cuts = boundaries.filter((value) => value >= minimum && value <= maximum);
    $: omittedCuts = boundaries.length - cuts.length;

    function x(value) {
        const proportion = Number.isFinite(range)
            ? (value - minimum) / range
            : (value / scale - scaledMinimum) / scaledRange;
        return left + proportion * plotWidth;
    }
    function y(rows) {
        return top + plotHeight * (1 - rows / tallest);
    }
    function short(value) {
        if (value && (Math.abs(value) >= 1e6 || Math.abs(value) < 0.001))
            return Number(value).toExponential(2);
        return Number(value).toLocaleString(undefined, { maximumSignificantDigits: 4 });
    }
</script>

<figure class="binning-histogram" aria-label="Training distribution">
    <figcaption>
        <strong>Training distribution</strong>
        <span
            >Equal-width bars show the distribution; dashed lines show proposed model cuts within
            the observed range.{#if omittedCuts}
                {omittedCuts} model {omittedCuts === 1 ? 'boundary' : 'boundaries'} outside the range
                {omittedCuts === 1 ? 'is' : 'are'} omitted.{/if}</span
        >
    </figcaption>
    {#if !finiteRows || !bars.length || !(maximum > minimum) || (!Number.isFinite(range) && !scaledRange)}
        <p>No finite training values to plot.</p>
    {:else}
        <svg
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label={`Training distribution for ${name}: ${finiteRows.toLocaleString()} finite training rows and ${cuts.length} proposed cut lines`}
            preserveAspectRatio="xMidYMid meet"
        >
            <title>Training distribution for {name}</title>
            <desc>
                Equal-width histogram of finite training values with dashed proposed model cuts.
            </desc>
            <text class="axis-label" x={left} y={13}>Rows</text>
            <line class="axis" x1={left} y1={top} x2={left} y2={top + plotHeight} />
            <line
                class="axis"
                x1={left}
                y1={top + plotHeight}
                x2={left + plotWidth}
                y2={top + plotHeight}
            />
            {#each [0, tallest / 2, tallest] as value}
                <line class="grid" x1={left} x2={left + plotWidth} y1={y(value)} y2={y(value)} />
                <text class="count-tick" x={left - 7} y={y(value) + 4} text-anchor="end"
                    >{short(value)}</text
                >
            {/each}
            {#each bars as bar}
                <rect
                    class="distribution-bar"
                    data-lower={bar.lower}
                    data-upper={bar.upper}
                    x={x(bar.lower) + 0.4}
                    y={y(bar.rows)}
                    width={Math.max(0.5, x(bar.upper) - x(bar.lower) - 0.8)}
                    height={top + plotHeight - y(bar.rows)}
                >
                    <title>Training values {bar.lower} to {bar.upper}: {bar.rows} rows</title>
                </rect>
            {/each}
            {#each cuts as cut}
                <line
                    class="model-cut"
                    data-cut={cut}
                    x1={x(cut)}
                    x2={x(cut)}
                    y1={top}
                    y2={top + plotHeight}
                >
                    <title>Proposed model cut at {cut}</title>
                </line>
            {/each}
            {#each ticks as value, index}
                <line
                    class="tick"
                    x1={x(value)}
                    x2={x(value)}
                    y1={top + plotHeight}
                    y2={top + plotHeight + 5}
                />
                <text
                    class="value-tick"
                    x={x(value)}
                    y={height - 12}
                    text-anchor={index === 0 ? 'start' : index === 4 ? 'end' : 'middle'}
                    >{short(value)}</text
                >
            {/each}
        </svg>
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
    svg {
        display: block;
        width: 100%;
        height: auto;
        margin-top: 9px;
        overflow: visible;
    }
    .distribution-bar {
        fill: #4b82b6;
    }
    .distribution-bar:hover {
        fill: #1c5f9e;
    }
    .model-cut {
        stroke: #bf5a1d;
        stroke-width: 2;
        stroke-dasharray: 6 4;
        pointer-events: stroke;
    }
    .axis {
        stroke: #607486;
        stroke-width: 1;
    }
    .grid {
        stroke: #dce5ec;
        stroke-width: 1;
    }
    .tick {
        stroke: #607486;
        stroke-width: 1;
    }
    text {
        fill: #465e73;
        font-size: 11px;
    }
</style>
