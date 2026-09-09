// Display only: never pass formatted values back into model inputs or exports.
export function formatNumber(value, options = {}) {
    if (
        value === null ||
        value === undefined ||
        (typeof value === 'number' && !Number.isFinite(value))
    )
        return '—';
    if (typeof value !== 'number') return String(value);
    if (value === 0) return '0';
    if (options.scientific || Math.abs(value) < 0.01) {
        return value
            .toExponential(3)
            .replace(/\.?0+e/, 'e')
            .replace('e+', 'e');
    }
    return value.toLocaleString('en-GB', {
        maximumFractionDigits: 3,
        useGrouping: options.grouping !== false,
    });
}

// Range labels carry identity: keep the original in tooltips and disambiguate
// labels that collapse at display precision. Ordinary categorical names stay exact.
export function formatLabels(labels) {
    const compact = labels.map((value, index) => {
        const text = String(value ?? '—');
        if (!/^(?:\[|\(|<|>|≤|≥)/.test(text)) return text;
        const bounds = text.match(/[-+]?\d*\.?\d+(?:e[-+]?\d+)?/gi) || [];
        if (
            bounds.length === 2 &&
            Number(bounds[0]) !== Number(bounds[1]) &&
            formatNumber(Number(bounds[0])) === formatNumber(Number(bounds[1]))
        )
            return `Band ${index + 1}`;
        return text.replace(/[-+]?\d*\.?\d+(?:e[-+]?\d+)?/gi, (token) =>
            formatNumber(Number(token), { grouping: false }),
        );
    });
    return compact.map((label, i) =>
        compact.some(
            (other, j) => j !== i && other === label && String(labels[j]) !== String(labels[i]),
        )
            ? `${label} · band ${i + 1}`
            : label,
    );
}

// Axis ticks show a band's lower boundary; the full range remains in its tooltip/table.
export function axisLabel(label) {
    const range = String(label).match(/^[[(]([^,]+),/);
    return range ? range[1].trim() : String(label);
}
