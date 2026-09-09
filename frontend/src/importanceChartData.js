export function importanceChartData(rows) {
    const ranked = rows
        .filter((row) => Number.isFinite(row.importance))
        .map((row) => ({ ...row, std: Number.isFinite(row.std) ? Math.max(0, row.std) : 0 }))
        .sort((a, b) => b.importance - a.importance || a.variable.localeCompare(b.variable));
    const low = Math.min(0, ...ranked.map((row) => row.importance - row.std));
    const high = Math.max(0, ...ranked.map((row) => row.importance + row.std));
    const span = high - low || 1;
    return { rows: ranked, low: low - span * 0.04, high: high + span * 0.04 };
}
