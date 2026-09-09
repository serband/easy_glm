// Geometry from canonical exported rows. Linear curves interpolate in log space.
export function rateChartData(table) {
    const rows = table?.rows || [];
    const linear = table?.columns?.includes('slope');
    const interaction = table?.columns?.includes('label_a');
    const numeric =
        !interaction && rows.some((r) => typeof r.from === 'number' || typeof r.to === 'number');
    const edges = rows
        .flatMap((r) => [r.from, r.to])
        .filter((v) => typeof v === 'number' && Number.isFinite(v));
    const lo = edges.length ? Math.min(...edges) : 0;
    const hi = edges.length ? Math.max(...edges) : 1;
    const pad = Math.max(hi - lo, 1) * 0.06;
    const nullRow = numeric && rows.some((r) => r.from === null && r.to === null);
    const right = nullRow ? 615 : 690;
    const x = (value) => 55 + ((value - (lo - pad)) / (hi - lo + 2 * pad)) * (right - 55);
    const points = [];
    rows.forEach((row, i) => {
        const missing = numeric && row.from === null && row.to === null;
        const center = numeric
            ? missing
                ? 682
                : x(((row.from ?? lo - pad) + (row.to ?? hi + pad)) / 2)
            : 55 + (i * 635) / Math.max(1, rows.length - 1);
        const item = {
            row,
            index: i,
            x: center,
            label: row.label || `${row.label_a} × ${row.label_b}`,
            fitted: [],
            current: [],
        };
        if (numeric && !missing) {
            const start = row.from ?? lo - pad,
                end = row.to ?? hi + pad;
            const neighbor = rows.find((r) => r.from === row.to && r.from !== null);
            const fittedEnd =
                linear && row.from !== null && row.to !== null ? neighbor?.fitted : row.fitted;
            for (const field of ['fitted', 'current']) {
                const initial = field === 'fitted' ? row.fitted : row.relativity;
                if (!Number.isFinite(initial)) continue;
                const graded = linear && row.from !== null && row.to !== null;
                if (field === 'fitted' && graded && !Number.isFinite(fittedEnd)) {
                    item[field] = [{ x: x(start), value: initial }]; // page boundary: do not invent an endpoint
                    continue;
                }
                const count = graded ? 16 : 1;
                item[field] = Array.from({ length: count + 1 }, (_, j) => {
                    const t = j / count,
                        value =
                            field === 'current' && graded
                                ? initial * Math.exp(row.slope * (end - start) * t)
                                : graded
                                  ? initial * Math.exp(Math.log(fittedEnd / initial) * t)
                                  : initial;
                    return { x: x(start + (end - start) * t), value };
                });
            }
        } else {
            if (Number.isFinite(row.fitted)) item.fitted = [{ x: center, value: row.fitted }];
            if (Number.isFinite(row.relativity))
                item.current = [{ x: center, value: row.relativity }];
        }
        points.push(item);
    });
    return {
        points,
        linear,
        interaction,
        numeric,
        nullRow,
        max: Math.max(1, ...points.flatMap((p) => [...p.fitted, ...p.current].map((v) => v.value))),
        maxExposure: Math.max(1, ...rows.map((r) => r.exposure || 0)),
    };
}
