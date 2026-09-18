<script>
    import { formatNumber as num } from './format.js';

    export let rows = [];
    export let filterKey = '';

    const pageSize = 20;
    let page = 0;
    let seenFilter = '';

    $: if (filterKey !== seenFilter) {
        seenFilter = filterKey;
        page = 0;
    }
    $: scored = rows
        .filter(
            (row) =>
                ['signal', 'no_signal'].includes(row.status) &&
                Number.isFinite(row.importance) &&
                Number.isFinite(row.threshold),
        )
        .sort(
            (left, right) =>
                right.importance - left.importance || left.variable.localeCompare(right.variable),
        );
    $: signalCount = rows.filter((row) => row.status === 'signal').length;
    $: noSignalCount = rows.filter((row) => row.status === 'no_signal').length;
    $: unplottedCount = rows.length - scored.length;
    $: values = scored.flatMap((row) => [row.importance, row.threshold]);
    $: domain = values.reduce(
        ([minimum, maximum], value) => [Math.min(minimum, value), Math.max(maximum, value)],
        [0, 0],
    );
    $: minimum = domain[0];
    $: maximum = domain[1];
    $: scale = Math.max(Math.abs(minimum), Math.abs(maximum), 1);
    $: range = maximum - minimum;
    $: scaledMinimum = minimum / scale;
    $: scaledRange = maximum / scale - scaledMinimum;
    $: zero = position(0, minimum, maximum, range, scale, scaledMinimum, scaledRange);
    $: visible = scored.slice(page * pageSize, (page + 1) * pageSize);

    function position(
        value,
        lower = minimum,
        upper = maximum,
        width = range,
        factor = scale,
        scaledLower = scaledMinimum,
        scaledWidth = scaledRange,
    ) {
        if (lower === upper) return 50;
        const proportion = Number.isFinite(width)
            ? (value - lower) / width
            : (value / factor - scaledLower) / scaledWidth;
        return Math.max(0, Math.min(100, proportion * 100));
    }
    function barLeft(value) {
        return Math.min(zero, position(value)) + '%';
    }
    function barWidth(value) {
        return Math.abs(position(value) - zero) + '%';
    }
    function outcome(status) {
        return status === 'signal' ? 'Signal detected' : 'No signal detected';
    }
</script>

<section class="importance-chart" aria-label="Ranked feature importance">
    <div class="chart-heading">
        <h3>Ranked importance</h3>
        <p>
            Each candidate was screened separately against its controls. The orange marker is the
            strongest control benchmark (four shuffled copies and random noise, floored at zero). No
            signal detected is not proof of no effect.
        </p>
        <p class="chart-counts">
            {signalCount} signal detected · {noSignalCount} no signal detected{#if unplottedCount}
                · {unplottedCount} without a plotted score (see table){/if}
        </p>
    </div>
    {#if scored.length}
        <div class="chart-legend" aria-hidden="true">
            <span><i class="legend-bar"></i>Real importance</span>
            <span><i class="legend-marker"></i>Control benchmark</span>
            <span><i class="legend-zero"></i>Zero</span>
        </div>
        <div class="scale-labels" aria-hidden="true">
            {#if minimum === maximum}<span class="only-zero">0</span>{:else}
                <span>{num(minimum)}</span>
                {#if zero > 15 && zero < 85}<span class="zero-label" style:left={zero + '%'}>0</span
                    >{/if}
                <span>{num(maximum)}</span>
            {/if}
        </div>
        <ol class="ranked-list" start={page * pageSize + 1}>
            {#each visible as row (row.variable)}
                <li class:signal={row.status === 'signal'} data-variable={row.variable}>
                    <div class="row-heading">
                        <strong title={row.variable}>{row.variable}</strong>
                        <span>{outcome(row.status)}</span>
                    </div>
                    <div
                        class="importance-track"
                        role="img"
                        aria-label={`${row.variable}: real importance ${row.importance}; strongest control benchmark ${row.threshold}; ${outcome(row.status)}`}
                    >
                        <span class="zero-line" style:left={zero + '%'} aria-hidden="true"></span>
                        <span
                            class="importance-bar"
                            style:left={barLeft(row.importance)}
                            style:width={barWidth(row.importance)}
                            title={`Real importance: ${row.importance}`}
                            aria-hidden="true"
                        ></span>
                        <span
                            class="control-marker"
                            style:left={position(row.threshold) + '%'}
                            title={`Strongest control benchmark: ${row.threshold}`}
                            aria-hidden="true"
                        ></span>
                    </div>
                    <div class="row-values">
                        <span>Real {num(row.importance)}</span>
                        <span>Control {num(row.threshold)}</span>
                    </div>
                </li>
            {/each}
        </ol>
        <div class="chart-footer">
            <span>
                Ranked {page * pageSize + 1}–{Math.min(scored.length, (page + 1) * pageSize)} of
                {scored.length} scored candidates
            </span>
            {#if scored.length > pageSize}<div class="chart-pages">
                    <button
                        type="button"
                        aria-label="Previous importance ranks"
                        disabled={!page}
                        onclick={() => (page -= 1)}>Previous ranks</button
                    >
                    <button
                        type="button"
                        aria-label="Next importance ranks"
                        disabled={(page + 1) * pageSize >= scored.length}
                        onclick={() => (page += 1)}>Next ranks</button
                    >
                </div>{/if}
        </div>
        <p class="axis-title">Training deviance increase after shuffling</p>
    {:else}
        <p class="empty-chart">
            No scored candidates match this filter. Skipped and failed candidates remain in the
            table below.
        </p>
    {/if}
</section>

<style>
    .importance-chart {
        min-width: 0;
        margin: 13px 0 18px;
        padding: 14px;
        border: 1px solid #dde7ed;
        border-radius: 7px;
        background: #fbfdfe;
        font-size: 11px;
    }
    .chart-heading h3 {
        margin: 0 0 5px;
        color: #173a55;
        font-size: 14px;
    }
    .chart-heading p,
    .importance-chart .empty-chart {
        margin: 0 0 8px;
        color: #526777;
    }
    .chart-heading .chart-counts {
        color: #254d65;
        font-weight: 600;
    }
    .chart-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 17px;
        margin: 12px 0 7px;
        color: #435d6c;
    }
    .chart-legend span {
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }
    .chart-legend i {
        display: inline-block;
        width: 13px;
        height: 11px;
    }
    .legend-bar {
        background: #257b93;
    }
    .legend-marker {
        border-left: 3px solid #c26124;
    }
    .legend-zero {
        border-left: 1px dashed #6c8190;
    }
    .scale-labels {
        position: relative;
        display: flex;
        justify-content: space-between;
        height: 18px;
        margin: 0 1px 0 25px;
        color: #526777;
    }
    .scale-labels .zero-label,
    .scale-labels .only-zero {
        position: absolute;
        transform: translateX(-50%);
    }
    .scale-labels .only-zero {
        left: 50%;
    }
    .ranked-list {
        margin: 0;
        padding: 0 0 0 24px;
    }
    .ranked-list li {
        min-width: 0;
        padding: 8px 0;
        border-top: 1px solid #e6edf0;
        color: #476477;
    }
    .row-heading,
    .row-values,
    .chart-footer {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 3px 10px;
    }
    .row-heading strong {
        min-width: 0;
        overflow-wrap: anywhere;
        color: #1b4358;
    }
    .ranked-list li.signal .row-heading span {
        color: #116275;
        font-weight: 600;
    }
    .importance-track {
        position: relative;
        height: 17px;
        margin: 5px 1px 2px;
        border-radius: 2px;
        background: #eaf0f3;
    }
    .zero-line,
    .control-marker {
        position: absolute;
        top: 0;
        bottom: 0;
        transform: translateX(-50%);
    }
    .zero-line {
        border-left: 1px dashed #708695;
        z-index: 1;
    }
    .importance-bar {
        position: absolute;
        top: 3px;
        height: 11px;
        background: #8fa3af;
        z-index: 2;
    }
    .ranked-list li.signal .importance-bar {
        background: #257b93;
    }
    .control-marker {
        border-left: 3px solid #c26124;
        z-index: 3;
    }
    .row-values {
        color: #4e6676;
    }
    .chart-footer {
        align-items: center;
        padding-top: 10px;
        border-top: 1px solid #e6edf0;
        color: #526777;
    }
    .chart-pages {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .chart-pages button {
        font-size: 10px;
        padding: 5px 8px;
    }
    .importance-chart .axis-title {
        margin: 7px 0 0;
        color: #3f5869;
        text-align: center;
        font-size: 11px;
    }
    @media (max-width: 650px) {
        .importance-chart {
            padding: 11px;
        }
        .row-heading span {
            width: 100%;
        }
    }
</style>
