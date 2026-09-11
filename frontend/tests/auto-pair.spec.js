import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('pair A/E follows latest selections and context without stale heatmaps', async ({ page }) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    const tab = (n) => page.getByRole('tab', { name: n, exact: true });
    const starts = [],
        errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (
            r.method() === 'POST' &&
            /\/api\/review\/[^/]+$/.test(r.url()) &&
            r.postDataJSON().action === 'pair'
        )
            starts.push({ ...r.postDataJSON(), model: r.url().split('/').at(-1) });
    });
    await page.goto('/');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('unassigned');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await button('Model').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill('Challenger');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await page.getByLabel('Model selection').selectOption('Frequency');
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'x-easyglm-token': token } })).json();
    const before = await get('jobs');
    await button('Diagnostics').click();
    await expect(page.locator('.diagnostic-plot').first()).toBeVisible();
    let held = false,
        release,
        delayed = false;
    const gate = new Promise((r) => (release = r));
    await page.route('**/api/reviews/*', async (route) => {
        if (route.request().method() !== 'GET') return route.continue();
        const response = await route.fetch();
        const body = await response.json();
        if (!delayed && body.status === 'complete' && body.data?.rows?.[0]?.label_a) {
            held = true;
            delayed = true;
            await gate;
        }
        await route.fulfill({ response });
    });
    let refused = false;
    await page.route('**/api/review/Frequency', async (route) => {
        if (!refused && route.request().postDataJSON()?.action === 'pair') {
            refused = true;
            return route.fulfill({
                status: 409,
                contentType: 'application/json',
                body: JSON.stringify({
                    detail: 'A review is running. Wait for it or cancel it first.',
                }),
            });
        }
        return route.continue();
    });
    await tab('A/E by pair').click();
    await expect.poll(() => held).toBeTruthy();
    await expect(button('Show pair A/E')).toHaveCount(0);
    expect(refused).toBeTruthy();
    expect(starts).toHaveLength(2);
    expect(starts[0]).toEqual(starts[1]);
    const first = page.getByLabel('Pair first variable'),
        second = page.getByLabel('Pair second variable');
    await first.selectOption('Region');
    await second.selectOption('AnnualMileage');
    const bins = page.getByLabel('Pair diagnostic bins');
    await bins.fill('11');
    await bins.fill('13');
    await expect(page.locator('.ae-heatmap')).toHaveCount(0);
    release();
    await expect(
        page.getByRole('heading', { name: 'Region × AnnualMileage · Holdout', exact: true }),
    ).toBeVisible();
    expect(starts.at(-1)).toMatchObject({ a: 'Region', b: 'AnnualMileage', n_bins: 13 });
    const x = await first.boundingBox(),
        y = await second.boundingBox();
    expect(x.y).toBe(y.y);
    expect(x.height).toBe(y.height);
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await bins.fill('2');
    await expect(page.locator('.ae-heatmap')).toHaveCount(0);
    await expect(page.locator('.pair-status')).toContainText('whole number');
    const invalid = starts.length;
    await page.waitForTimeout(450);
    expect(starts).toHaveLength(invalid);
    await bins.fill('9');
    await expect(page.locator('.ae-heatmap')).toBeVisible();
    await second.selectOption('Region');
    await expect(page.locator('.ae-heatmap')).toHaveCount(0);
    await expect(page.getByText('Choose two different variables.', { exact: true })).toBeVisible();
    const same = starts.length;
    await page.waitForTimeout(450);
    expect(starts).toHaveLength(same);
    await second.selectOption('DriverAge');
    await expect(
        page.getByRole('heading', { name: 'Region × DriverAge · Holdout', exact: true }),
    ).toBeVisible();
    await page.getByLabel('Diagnostic subset', { exact: true }).selectOption('train');
    await expect(
        page.getByRole('heading', { name: 'Region × DriverAge · Training', exact: true }),
    ).toBeVisible();
    expect(starts.at(-1).subset).toBe('train');
    await expect(page.getByLabel('Compare with challenger')).toHaveCount(0);
    expect(starts.at(-1).challenger).toBeNull();
    await expect(page.locator('.ae-heatmap')).toBeVisible();
    await page.getByLabel('Model selection').selectOption('Challenger');
    await expect.poll(() => starts.at(-1).model).toBe('Challenger');
    await expect(page.locator('.ae-heatmap')).toBeVisible();
    await tab('Coefficients').click();
    await expect(page.locator('.diagnostic-table').first()).toBeVisible();
    const hidden = starts.length;
    await page.waitForTimeout(500);
    expect(starts).toHaveLength(hidden);
    expect((await get('jobs')).Frequency.id).toBe(before.Frequency.id);
    expect((await get('jobs')).Challenger.id).toBe(before.Challenger.id);
    expect(errors).toEqual([]);
});
