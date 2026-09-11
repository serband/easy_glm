import { test, expect } from '@playwright/test';
import { formatNumber } from '../src/format.js';
test.use({ viewport: { width: 884, height: 773 } });
test('Compare requires two compatible fits and shows exact metric deltas', async ({ page }) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    await page.goto('/');
    await button('Model').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Compare').click();
    await expect(
        page.getByRole('heading', { name: 'Compare needs two fitted models' }),
    ).toBeVisible();
    await button('Open Model').click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill('Challenger');
    await page.getByLabel('Fixed alpha').fill('.1');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await page.getByLabel('Model selection').selectOption('Frequency');
    await button('Compare').click();
    await page.getByLabel('Compare with challenger').selectOption('');
    await expect(page.getByRole('heading', { name: 'Select a challenger' })).toHaveCount(0);
    await expect(page.locator('.metrics-grid,.job-card,[role="tablist"]')).toHaveCount(0);
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'x-easyglm-token': token } })).json();
    const before = JSON.stringify(await get('jobs'));
    await page.getByLabel('Compare with challenger').selectOption('Challenger');
    await expect(page.getByRole('heading', { name: 'Metrics side by side' })).toBeVisible();
    const a = await get('results/Frequency'),
        b = await get('results/Challenger');
    for (const subset of ['train', 'holdout']) {
        await page.getByLabel('Comparison subset').selectOption(subset);
        const row = page
            .locator('.diagnostic-table')
            .first()
            .locator('tbody tr')
            .filter({ hasText: 'Actual / expected' });
        await expect(row.locator('td').nth(3)).toHaveText(
            formatNumber(b.metrics[subset].ae - a.metrics[subset].ae),
        );
    }
    await expect(page.getByText(/Numeric factors use the union/)).toBeVisible();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await page.screenshot({ path: '/tmp/easyglm-focused-compare.png', fullPage: true });
    await page.getByLabel('Compare with challenger').selectOption('');
    await expect(page.getByRole('heading', { name: 'Select a challenger' })).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Relativities that differ' })).toHaveCount(0);
    await page.route('**/api/workbench', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        body.models.Challenger.family = 'gamma';
        await route.fulfill({ response, json: body });
    });
    await page.reload();
    await button('Compare').click();
    await page.getByLabel('Model selection').selectOption('Frequency');
    await page.getByLabel('Compare with challenger').selectOption('Challenger');
    await expect(page.getByText(/These models have different family settings/)).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Metrics side by side' })).toHaveCount(0);
    expect(JSON.stringify(await get('jobs'))).toBe(before);
});
