// Display preparation only: metrics and aligned rate differences come from the workflow.
export function comparisonIssue(a, b, first, second, subset) {
    if (!a || !b || !first || !second) return 'Select two fitted models to compare.';
    for (const [field, label] of [
        ['target', 'target'],
        ['weight', 'exposure / weight'],
        ['divide_target_by_weight', 'target scaling'],
        ['family', 'family'],
    ]) {
        if (a[field] !== b[field])
            return `These models have different ${label} settings. Select models with the same response and scoring basis.`;
    }
    if (
        first.link !== second.link ||
        (a.family === 'tweedie' && a.tweedie_power !== b.tweedie_power)
    )
        return 'These models have different links or Tweedie powers. Select models with the same response and scoring basis.';
    const x = first.metrics[subset],
        y = second.metrics[subset];
    if (!x || !y || !x.rows || !y.rows)
        return 'Both models need observations in the selected subset.';
    if (
        ['rows', 'exposure', 'actual'].some(
            (k) => Math.abs(x[k] - y[k]) > 1e-9 * Math.max(1, Math.abs(x[k]), Math.abs(y[k])),
        )
    )
        return 'These results do not cover the same rows and exposure. Refresh the fitted models before comparing.';
    return '';
}
export function comparisonMetrics(first, second, subset) {
    const a = first.metrics[subset],
        b = second.metrics[subset];
    return [
        ['ae', 'Actual / expected'],
        ['gini', 'Normalised Gini'],
        ['deviance_explained', 'Deviance explained'],
        ['mean_deviance', 'Mean deviance'],
        ['expected', 'Total expected'],
    ].map(([key, metric]) => ({
        metric,
        baseline: a[key],
        challenger: b[key],
        delta: Number.isFinite(a[key]) && Number.isFinite(b[key]) ? b[key] - a[key] : null,
    }));
}
export function comparisonSettings(a, b, first, second) {
    const facts = (r) => r.diagnostic_info?.facts || {};
    const rows = [];
    const add = (setting, x, y) => {
        if (JSON.stringify(x) === JSON.stringify(y)) return;
        if ((x && typeof x === 'object') || (y && typeof y === 'object')) {
            for (const key of new Set([...Object.keys(x || {}), ...Object.keys(y || {})]))
                add(`${setting} · ${key.replaceAll('_', ' ')}`, x?.[key], y?.[key]);
        } else rows.push({ setting, baseline: x, challenger: y });
    };
    add('Main factors', a.predictors.length, b.predictors.length);
    add('Defined interactions', (a.interactions || []).length, (b.interactions || []).length);
    add('Total rate tables', (first.table_index || []).length, (second.table_index || []).length);
    for (const key of ['alpha', 'features', 'nonzero', 'adjustments'])
        add(
            {
                nonzero: 'Retained coefficients',
                features: 'Design columns',
                adjustments: 'Applied adjustments',
                alpha: 'Fitted alpha',
            }[key],
            facts(first)[key],
            facts(second)[key],
        );
    for (const key of ['offset', 'base', 'penalty', 'monotone', 'interactions'])
        add(key, a[key], b[key]);
    const predictors = new Set([...(a.predictors || []), ...(b.predictors || [])]);
    for (const name of predictors)
        add(
            `${name} design`,
            a.predictors.includes(name) ? 'Included' : 'Not included',
            b.predictors.includes(name) ? 'Included' : 'Not included',
        );
    add('Stage 2 alpha', first.summary.alpha_stage2, second.summary.alpha_stage2);
    return rows;
}
