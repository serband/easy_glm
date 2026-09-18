import { test, expect } from '@playwright/test';

async function project(page) {
    return page.evaluate(async () => {
        const { token } = await (await fetch('/api/session')).json();
        const response = await fetch('/api/project', {
            headers: { 'X-EasyGLM-Token': token },
        });
        return response.json();
    });
}

async function variablesSnapshot(page) {
    return page.evaluate(async () => {
        const { token } = await (await fetch('/api/session')).json();
        const response = await fetch('/api/variables', {
            headers: { 'X-EasyGLM-Token': token },
        });
        return response.json();
    });
}

test('one-way screen uses training rows and stages reviewed role changes', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    const panel = page.getByRole('group', { name: 'One-way feature selection' });
    await panel.locator(':scope > summary').click();
    await expect(panel).toContainText('4 variables to test');
    await expect(panel.locator('.selection-scope')).toContainText(
        '2 assigned predictors + 2 unassigned variables',
    );
    await panel.getByText('Variables to test (4)', { exact: true }).click();
    await expect(panel.locator('.selection-variable-list li')).toHaveCount(4);
    await expect(panel.locator('.selection-variable-list')).toContainText('Risk — Predictor');
    await page.getByLabel('Also test unassigned variables').uncheck();
    await expect(panel).toContainText('2 variables to test');
    await expect(panel.locator('.selection-scope')).toContainText(
        'Unassigned variables are excluded',
    );
    await expect(panel.locator('.selection-variable-list li')).toHaveCount(2);
    await page.getByLabel('Also test unassigned variables').check();
    await expect(panel.locator('.selection-variable-list li')).toHaveCount(4);
    await expect(page.getByLabel('Divide target by weight for feature selection')).toBeChecked();
    await expect(panel).toContainText('you choose what to keep');
    await panel.getByText('Advanced settings', { exact: true }).click();
    await expect(page.getByLabel('Feature selection repeats')).toHaveValue('5');
    const before = await project(page);
    const request = page.waitForRequest('**/api/variables/feature-selection');
    await page.getByRole('button', { name: 'Run one-way feature selection' }).click();
    const submitted = (await request).postDataJSON();
    expect(submitted.options).toMatchObject({
        family: 'poisson',
        link: null,
        divide_target_by_weight: true,
        l1_ratio: 1,
        n_alphas: 20,
        repeats: 5,
        seed: 42,
        include_unassigned: true,
    });
    expect(submitted.setup.binning.default_bins).toBe(6);
    expect(submitted.setup.binning.overrides.Risk).toEqual({ method: 'cuts', cuts: [0.5] });
    await expect(panel.getByText('3 of 4 variables tested')).toBeVisible({ timeout: 150000 });
    const resultRows = panel.locator('.selection-table-wrap tbody tr:not(.selection-detail)');
    await expect(resultRows).toHaveCount(4);
    await expect(panel.getByLabel('Feature selection table order')).toHaveValue('importance');
    await expect(resultRows.first().locator('td:nth-child(2) strong')).toHaveText('Risk');
    const chart = panel.getByRole('region', { name: 'Ranked feature importance' });
    await expect(chart.locator('.ranked-list li')).toHaveCount(3);
    await expect(chart.locator('.ranked-list li').first()).toHaveAttribute('data-variable', 'Risk');
    await expect(chart.locator('[data-variable="Flat"]')).toHaveCount(0);
    await expect(chart.getByText('Training deviance increase after shuffling')).toBeVisible();
    await expect(chart.getByRole('img', { name: /Risk: real importance/ })).toBeVisible();
    await expect(panel.getByLabel('Details for Risk')).toBeVisible();
    await panel.getByLabel('Details for Risk').click();
    await expect(panel).toContainText('Shadow 4:');
    await expect(panel).toContainText('Random:');
    await page.setViewportSize({ width: 799, height: 820 });
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    const stickyColumns = await panel.locator('.selection-table-wrap').evaluate((wrapper) => {
        const select = wrapper.querySelector('tbody tr:not(.selection-detail) td:first-child');
        const variable = wrapper.querySelector('tbody tr:not(.selection-detail) td:nth-child(2)');
        const start = variable.getBoundingClientRect().left;
        wrapper.scrollLeft = wrapper.scrollWidth;
        return {
            scrollable: wrapper.scrollWidth > wrapper.clientWidth,
            selectVisible:
                select.getBoundingClientRect().left >= wrapper.getBoundingClientRect().left,
            variableVisible:
                variable.getBoundingClientRect().left >= wrapper.getBoundingClientRect().left &&
                Math.abs(variable.getBoundingClientRect().left - start) < 2,
        };
    });
    expect(stickyColumns).toEqual({
        scrollable: true,
        selectVisible: true,
        variableVisible: true,
    });
    await page.screenshot({
        path: testInfo.outputPath('feature-selection-results.png'),
        fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1000 });
    const afterRun = await project(page);
    expect(afterRun.data.roles).toEqual(before.data.roles);

    // Choose a completed, actionable row. Skipped/failed rows are never staged.
    const actionable = await panel
        .locator('input[aria-label^="Select feature "]:not([disabled])')
        .count();
    expect(actionable).toBeGreaterThan(0);
    const first = panel.locator('input[aria-label^="Select feature "]:not([disabled])').first();
    const displayName = (await first.getAttribute('aria-label')).replace('Select feature ', '');
    const rawName = displayName;
    const previousRole = before.data.roles[rawName];
    await first.check();
    const action = previousRole === 'predictor' ? 'Ignore selected' : 'Make selected predictors';
    await panel.getByRole('button', { name: action, exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply changes' })).toBeVisible();
    expect((await project(page)).data.roles).toEqual(before.data.roles);
    const intended = previousRole === 'predictor' ? 'ignore' : 'predictor';
    await expect(page.getByLabel('Role for ' + rawName)).toHaveValue(intended);
    await page.getByRole('button', { name: 'Variables JSON' }).click();
    const json = JSON.parse(await page.getByLabel('Variables JSON').inputValue());
    expect(json[intended]).toContain(rawName);
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes' }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    expect((await project(page)).data.roles[rawName]).toBe(intended);
    expect(errors).toEqual([]);
});

test('ranked chart shares a numeric scale, filters and pages without inventing skipped scores', async ({
    page,
}) => {
    await page.goto('/');
    const panel = page.getByRole('group', { name: 'One-way feature selection' });
    await panel.locator(':scope > summary').click();
    const snapshot = await variablesSnapshot(page);
    const rows = [
        {
            variable: 'Top',
            raw_name: 'Risk',
            status: 'signal',
            importance: 3,
            threshold: 0.5,
        },
        {
            variable: 'Weak signal',
            raw_name: 'Weak',
            status: 'signal',
            importance: 0.2,
            threshold: 0.1,
        },
        ...Array.from({ length: 43 }, (_, index) => ({
            variable: `Variable ${String(index + 1).padStart(2, '0')}`,
            raw_name: `Variable ${index + 1}`,
            status: 'no_signal',
            importance: (43 - index) / 100,
            threshold: 0.5,
        })),
        { variable: 'Zero', raw_name: 'Zero', status: 'no_signal', importance: 0, threshold: 0 },
        {
            variable: 'Negative',
            raw_name: 'Negative',
            status: 'no_signal',
            importance: -0.4,
            threshold: 0.2,
        },
        { variable: 'Skipped', raw_name: 'Skipped', status: 'skipped', importance: null },
        { variable: 'Failed', raw_name: 'Failed', status: 'failed', importance: null },
    ];
    await page.route('**/api/variables/feature-selection', async (route) => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                id: 'synthetic-ranking',
                status: 'complete',
                session_id: snapshot.session_id,
                project_id: snapshot.project_id,
                revision: snapshot.revision,
                fingerprint: 'synthetic-ranking',
                result: {
                    training_rows: 180,
                    candidate_count: rows.length,
                    tested_count: rows.length - 2,
                    cv_folds: 5,
                    repeats: 5,
                    rows,
                },
            }),
        });
    });
    await panel.getByRole('button', { name: 'Run one-way feature selection' }).click();
    const chart = panel.getByRole('region', { name: 'Ranked feature importance' });
    await expect(chart.locator('.ranked-list li')).toHaveCount(20);
    await expect(chart.locator('.ranked-list li').first()).toHaveAttribute('data-variable', 'Top');
    await expect(chart).toContainText('2 signal detected · 45 no signal detected');
    await expect(chart).toContainText('2 without a plotted score');
    await expect(chart).toContainText('Ranked 1–20 of 47 scored candidates');
    await expect(chart.locator('[data-variable="Skipped"]')).toHaveCount(0);
    await expect(chart.locator('[data-variable="Failed"]')).toHaveCount(0);
    const topGeometry = await chart.locator('[data-variable="Top"]').evaluate((item) => ({
        zero: parseFloat(item.querySelector('.zero-line').style.left),
        realLeft: parseFloat(item.querySelector('.importance-bar').style.left),
        realWidth: parseFloat(item.querySelector('.importance-bar').style.width),
        marker: parseFloat(item.querySelector('.control-marker').style.left),
    }));
    expect(topGeometry.realLeft).toBeCloseTo(topGeometry.zero);
    expect(topGeometry.marker).toBeGreaterThan(topGeometry.zero);
    expect(topGeometry.realLeft + topGeometry.realWidth).toBeGreaterThan(topGeometry.marker);
    await expect(
        chart.getByRole('img', {
            name: /Top: real importance 3; strongest control benchmark 0.5; Signal detected/,
        }),
    ).toBeVisible();

    await chart.getByRole('button', { name: 'Next importance ranks' }).click();
    await expect(chart).toContainText('Ranked 21–40 of 47 scored candidates');
    await chart.getByRole('button', { name: 'Next importance ranks' }).click();
    await expect(chart).toContainText('Ranked 41–47 of 47 scored candidates');
    await expect(chart.locator('.ranked-list li').last()).toHaveAttribute(
        'data-variable',
        'Negative',
    );
    const negativeGeometry = await chart.locator('[data-variable="Negative"]').evaluate((item) => ({
        zero: parseFloat(item.querySelector('.zero-line').style.left),
        realLeft: parseFloat(item.querySelector('.importance-bar').style.left),
        realWidth: parseFloat(item.querySelector('.importance-bar').style.width),
        marker: parseFloat(item.querySelector('.control-marker').style.left),
    }));
    expect(negativeGeometry.realLeft).toBeLessThan(negativeGeometry.zero);
    expect(negativeGeometry.realLeft + negativeGeometry.realWidth).toBeCloseTo(
        negativeGeometry.zero,
    );
    expect(negativeGeometry.marker).toBeGreaterThan(negativeGeometry.zero);
    await expect(chart.locator('[data-variable="Zero"] .importance-bar')).toHaveCSS('width', '0px');

    const tableRows = panel.locator('.selection-table-wrap tbody tr:not(.selection-detail)');
    const tableNames = tableRows.locator('td:nth-child(2) strong');
    const tableOrder = panel.getByLabel('Feature selection table order');
    await expect(tableOrder).toHaveValue('importance');
    await expect(tableNames.first()).toHaveText('Top');
    await expect(tableNames.nth(1)).toHaveText('Variable 01');
    await panel.getByLabel('Select feature Top').check();
    await panel.getByRole('button', { name: 'Next feature results' }).click();
    await expect(tableNames.first()).toHaveText('Variable 10');
    await tableOrder.selectOption('outcome');
    await expect(tableNames.first()).toHaveText('Top');
    await expect(tableNames.nth(1)).toHaveText('Weak signal');
    await expect(panel.getByLabel('Select feature Top')).toBeChecked();
    await expect(chart).toContainText('Ranked 41–47 of 47 scored candidates');
    for (let index = 0; index < 4; index++)
        await panel.getByRole('button', { name: 'Next feature results' }).click();
    await expect(tableNames.nth((await tableNames.count()) - 2)).toHaveText('Skipped');
    await expect(tableNames.last()).toHaveText('Failed');
    await tableOrder.selectOption('importance');
    await expect(tableNames.first()).toHaveText('Top');
    await expect(tableNames.nth(1)).toHaveText('Variable 01');
    await expect(panel.getByLabel('Select feature Top')).toBeChecked();
    for (let index = 0; index < 4; index++)
        await panel.getByRole('button', { name: 'Next feature results' }).click();
    await expect(tableNames.nth((await tableNames.count()) - 2)).toHaveText('Failed');
    await expect(tableNames.last()).toHaveText('Skipped');

    const search = panel.getByLabel('Search feature selection results');
    await search.fill('Variable');
    await expect(chart).toContainText('Ranked 1–20 of 43 scored candidates');
    const positiveZero = await chart
        .locator('.zero-line')
        .first()
        .evaluate((line) => line.style.left);
    expect(positiveZero).toBe('0%');
    await search.fill('Negative');
    await expect(chart.locator('.ranked-list li')).toHaveCount(1);
    const mixedZero = await chart
        .locator('.zero-line')
        .first()
        .evaluate((line) => parseFloat(line.style.left));
    expect(mixedZero).toBeGreaterThan(50);
    await search.fill('Zero');
    await expect(chart.locator('.importance-bar')).toHaveCSS('width', '0px');
    await search.fill('');
    await panel.getByLabel('Filter feature selection status').selectOption('skipped');
    await expect(chart.locator('.ranked-list li')).toHaveCount(0);
    await expect(chart).toContainText('No scored candidates match this filter');
    await expect(
        panel.locator('.selection-table-wrap tbody tr:not(.selection-detail)'),
    ).toHaveCount(1);
});

test('cancelled or changed drafts suppress late reports, and invalid JSON cannot start', async ({
    page,
}) => {
    await page.goto('/');
    const panel = page.getByRole('group', { name: 'One-way feature selection' });
    await panel.locator(':scope > summary').click();
    let release;
    const gate = new Promise((resolve) => {
        release = resolve;
    });
    await page.route('**/api/variables/feature-selection', async (route) => {
        const response = await route.fetch();
        await gate;
        await route.fulfill({ response });
    });
    const request = page.waitForRequest('**/api/variables/feature-selection');
    await panel.getByRole('button', { name: 'Run one-way feature selection' }).click();
    await request;
    await panel.getByRole('button', { name: 'Cancel feature selection' }).click();
    await page.getByLabel('Role for Noise').selectOption('ignore');
    const lateResponse = page.waitForResponse('**/api/variables/feature-selection');
    release();
    await lateResponse;
    await expect(panel.getByText(/of .* variables tested/)).toHaveCount(0);
    await page.getByRole('button', { name: 'Variables JSON' }).click();
    await page.getByLabel('Variables JSON').fill('{unfinished');
    let starts = 0;
    page.on('request', (item) => {
        if (item.url().endsWith('/api/variables/feature-selection')) starts++;
    });
    await panel.getByRole('button', { name: 'Run one-way feature selection' }).click();
    await expect(page.getByRole('alert')).toContainText('JSON');
    expect(starts).toBe(0);
});
