import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('training importance is automatic, cached by original fit and compatible with old local servers', async ({
    page,
}) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    const tab = (n) => page.getByRole('tab', { name: n, exact: true });
    const chart = page.getByRole('img', {
        name: 'Training permutation importance by variable',
        exact: true,
    });
    const starts = [],
        errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (r.method() === 'POST' && /\/api\/review\/[^/]+$/.test(r.url())) {
            const payload = r.postDataJSON();
            if (payload.action === 'importance' || payload.options?.view === 'importance')
                starts.push(payload);
        }
    });
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((n) => n.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'x-easyglm-token': token } })).json();
    const originalProject = await get('project'),
        originalJobs = await get('jobs');
    await button('Diagnostics').click();
    await expect(page.getByLabel('Diagnostic subset')).toHaveValue('holdout');
    await tab('Variable importance').click();
    await expect(chart).toBeVisible();
    await expect(
        page.getByRole('heading', { name: 'Permutation importance', exact: true }),
    ).toBeVisible();
    await expect(
        page.getByText('Training · Original fit · 5 shuffles per variable', { exact: true }),
    ).toBeVisible();
    await expect(
        page.locator('.model-workbench .metrics-grid,.model-workbench .result-totals'),
    ).toHaveCount(0);
    await expect(page.getByLabel('Diagnostic subset')).toHaveCount(0);
    expect(starts).toHaveLength(1);
    expect(starts[0].subset).toBe('train');
    expect(starts[0]).not.toHaveProperty('challenger');
    const rank = await chart.locator('.importance-row').evaluateAll((rows) =>
        rows.map((row) => ({
            variable: row.getAttribute('data-variable'),
            importance: Number(
                row.querySelector('.importance-bar').getAttribute('data-importance'),
            ),
        })),
    );
    expect(rank).toHaveLength(4);
    expect(rank.map((r) => r.variable).sort()).toEqual([
        'AnnualMileage',
        'DriverAge',
        'Region',
        'VehicleAge',
    ]);
    expect(rank.map((r) => r.importance)).toEqual(
        rank.map((r) => r.importance).sort((a, b) => b - a),
    );
    const tickBoxes = await chart
        .locator('.axis-tick')
        .evaluateAll((ticks) => ticks.map((tick) => tick.getBoundingClientRect().toJSON()));
    for (let i = 1; i < tickBoxes.length; i++)
        expect(tickBoxes[i].left).toBeGreaterThan(tickBoxes[i - 1].right);
    const initialChart = await chart.innerHTML();
    expect(await get('project')).toEqual(originalProject);
    expect(await get('jobs')).toEqual(originalJobs);
    await tab('A/E by variable').click();
    await expect(page.getByLabel('Diagnostic subset')).toHaveValue('holdout');
    await page.getByLabel('Diagnostic subset').selectOption('all');
    await tab('Variable importance').click();
    await expect(chart).toBeVisible();
    expect(await chart.innerHTML()).toBe(initialChart);
    expect(starts).toHaveLength(1);
    // Adjusting a rate changes the applied model only; importance must reuse its original-fit cache.
    await button('Rate tables').click();
    await page.getByLabel('Adjustment method', { exact: true }).selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('1.7');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await button('Diagnostics').click();
    await tab('Variable importance').click();
    await expect(chart).toBeVisible();
    expect(await chart.innerHTML()).toBe(initialChart);
    expect(starts).toHaveLength(1);
    expect((await get('jobs')).Frequency.id).toBe(originalJobs.Frequency.id);
    // Reload resets only the client cache in this disposable test session.
    await page.route('**/api/review/Frequency', async (route) => {
        if (route.request().postDataJSON().action === 'importance') {
            await route.fulfill({
                status: 422,
                json: {
                    detail: [
                        {
                            type: 'literal_error',
                            loc: ['body', 'action'],
                            input: 'importance',
                            ctx: { expected: "'variable', 'coefficients'" },
                        },
                    ],
                },
            });
        } else await route.continue();
    });
    await page.reload();
    await button('Diagnostics').click();
    await tab('Variable importance').click();
    await expect(chart).toBeVisible();
    expect(starts.at(-1).action).toBe('coefficients');
    expect(starts.at(-1).options.view).toBe('importance');
    expect(starts.at(-1)).not.toHaveProperty('challenger');
    expect(await chart.innerHTML()).toBe(initialChart);
    await page.unroute('**/api/review/Frequency');
    // Other failures must never invoke the bridge or show a stale chart.
    await page.route('**/api/review/Frequency', async (route) => {
        if (route.request().postDataJSON().action === 'importance')
            await route.fulfill({
                status: 422,
                json: { detail: 'Importance unavailable for test' },
            });
        else await route.continue();
    });
    await page.reload();
    await button('Diagnostics').click();
    const beforeFailure = starts.length;
    await tab('Variable importance').click();
    await expect(page.getByText('Importance unavailable for test', { exact: true })).toBeVisible();
    expect(starts).toHaveLength(beforeFailure + 1);
    await expect(chart).toHaveCount(0);
    await page.unroute('**/api/review/Frequency');
    await tab('A/E by variable').click();
    await tab('Variable importance').click();
    await expect(chart).toBeVisible();
    // A different fit's delayed response cannot replace the selected model's cached chart.
    await button('Model').click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill('Challenger');
    await page.getByLabel('Fixed alpha').fill('.1');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    let held, release, delayedId;
    const waiting = new Promise((r) => (held = r)),
        gate = new Promise((r) => (release = r));
    await page.route('**/api/review/Challenger', async (route) => {
        const response = await route.fetch(),
            data = await response.json();
        if (route.request().postDataJSON().action === 'importance') delayedId = data.id;
        await route.fulfill({ response, json: data });
    });
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch();
        if (
            route.request().method() === 'GET' &&
            route
                .request()
                .url()
                .endsWith('/' + delayedId)
        ) {
            const data = await response.json();
            if (data.status === 'complete') {
                held();
                await gate;
            }
            await route.fulfill({ response, json: data });
        } else await route.fulfill({ response });
    });
    await button('Diagnostics').click();
    await tab('Variable importance').click();
    await waiting;
    await page.getByLabel('Model selection').selectOption('Frequency');
    release();
    await expect(chart).toBeVisible();
    expect(await chart.innerHTML()).toBe(initialChart);
    await page.waitForTimeout(250);
    expect(await chart.innerHTML()).toBe(initialChart);
    await page.unrouteAll({ behavior: 'wait' });
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    expect(errors).toEqual([]);
});
