import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('adjustment drafts do nothing until one explicit Apply, with safe failures and cancellation', async ({
    page,
}) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    let tools = 0,
        commits = 0;
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (
            r.method() === 'POST' &&
            /\/api\/review\/[^/]+$/.test(r.url()) &&
            ['moving', 'isotonic', 'cap', 'round', 'edit'].includes(r.postDataJSON().action)
        )
            tools++;
        if (r.method() === 'POST' && /\/api\/reviews\/[^/]+\/apply$/.test(r.url())) commits++;
    });
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((n) => n.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Rate tables').click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'x-easyglm-token': token } })).json();
    const initial = await get('project');
    const jobs = await get('jobs');
    const chart = page.locator('.relativity-chart');
    const html = await chart.innerHTML();
    const method = page.getByLabel('Adjustment method', { exact: true });
    await expect(method).toHaveValue('');
    await method.selectOption('moving');
    await page.getByLabel('Smoothing window').fill('5');
    await page.waitForTimeout(450);
    expect(tools).toBe(0);
    expect(commits).toBe(0);
    expect(await chart.innerHTML()).toBe(html);
    await expect(page.getByText('Preview · not applied', { exact: true })).toHaveCount(0);
    await expect(button('Preview adjustment')).toHaveCount(0);
    await button('Apply').dblclick();
    await expect(button('Apply')).toBeDisabled();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect(tools).toBe(1);
    expect(commits).toBe(1);
    const applied = await get('project');
    expect(applied.models.Frequency.adjustments).not.toEqual(initial.models.Frequency.adjustments);
    await expect(page.locator('.rate-chart-card .chart-legend')).toContainText('Original fit');
    await expect(page.locator('.rate-chart-card .chart-legend')).toContainText('Adjusted');
    // A calculation failure changes neither saved state nor chart.
    await method.selectOption('round');
    const savedChart = await chart.innerHTML();
    await page.route('**/api/review/Frequency', (route) =>
        route.fulfill({ status: 422, json: { detail: 'Calculation failed for test' } }),
    );
    await button('Apply').click();
    await expect(page.getByText('Calculation failed for test', { exact: true })).toBeVisible();
    expect(await get('project')).toEqual(applied);
    expect(await chart.innerHTML()).toBe(savedChart);
    expect(commits).toBe(1);
    await page.unroute('**/api/review/Frequency');
    // A delayed calculation cannot apply after the selected factor changes.
    let release, held;
    const gate = new Promise((r) => (release = r)),
        waiting = new Promise((r) => (held = r));
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        if (route.request().method() === 'GET' && data.status === 'complete') {
            held();
            await gate;
        }
        await route.fulfill({ response, json: data });
    });
    await method.selectOption('cap');
    await page.getByLabel('Relativity floor').fill('2');
    await page.getByLabel('Relativity cap').fill('3');
    await button('Apply').click();
    await waiting;
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('Region');
    release();
    await expect(button('Apply')).toBeEnabled();
    await page.waitForTimeout(350);
    expect(commits).toBe(1);
    expect(await get('project')).toEqual(applied);
    await page.unrouteAll({ behavior: 'wait' });
    // Cancellation is local immediately and never commits the completed worker later.
    let release2, held2;
    const gate2 = new Promise((r) => (release2 = r)),
        waiting2 = new Promise((r) => (held2 = r));
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        if (route.request().method() === 'GET' && data.status === 'complete') {
            held2();
            await gate2;
        }
        await route.fulfill({ response, json: data });
    });
    await method.selectOption('round');
    await button('Apply').click();
    await waiting2;
    await button('Cancel review').click();
    release2();
    await page.waitForTimeout(350);
    expect(commits).toBe(1);
    await page.unrouteAll({ behavior: 'wait' });
    // Two manual cells form one saved action; one undo restores both.
    await method.selectOption('manual');
    const beforeManual = await get('project');
    await page.getByLabel('Relativity row 1', { exact: true }).fill('1.6');
    await page.getByLabel('Relativity row 1', { exact: true }).press('Tab');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('2.3');
    await page.getByLabel('Relativity row 2', { exact: true }).press('Tab');
    const count = commits;
    await button('Apply row edits (2)').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect(commits).toBe(count + 1);
    await button('Preview undo').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect((await get('project')).models.Frequency.adjustments).toEqual(
        beforeManual.models.Frequency.adjustments,
    );
    // A successful commit remains saved if reloading the displayed results fails.
    const beforeRefreshFailure = await get('project');
    await method.selectOption('manual');
    await page.getByLabel('Relativity row 1', { exact: true }).fill('1.83');
    await page.getByLabel('Relativity row 1', { exact: true }).press('Tab');
    await page.route('**/api/workbench', (route) =>
        route.fulfill({ status: 500, json: { detail: 'Refresh failed for test' } }),
    );
    const beforeCommitCount = commits;
    await button('Apply row edits (1)').click();
    await expect(
        page.getByText('Adjustments saved. Refresh the page to reload the charts.', {
            exact: true,
        }),
    ).toBeVisible();
    expect(commits).toBe(beforeCommitCount + 1);
    expect((await get('project')).models.Frequency.adjustments).not.toEqual(
        beforeRefreshFailure.models.Frequency.adjustments,
    );
    await page.unroute('**/api/workbench');
    expect((await get('jobs')).Frequency.id).toBe(jobs.Frequency.id);
    expect(errors).toEqual([]);
});
