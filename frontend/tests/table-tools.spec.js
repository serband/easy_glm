import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('dropdown tools apply once and preserve undo and snapshots', async ({ page }) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    const method = page.getByLabel('Adjustment method', { exact: true });
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
    await expect(method).toHaveValue('');
    await expect(method.locator('option[value="round"]')).toHaveCount(0);
    await expect(button('Moving average')).toHaveCount(0);
    await method.selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('1.37');
    await page.getByLabel('Relativity row 2', { exact: true }).press('Tab');
    await page.getByLabel('Relativity row 3', { exact: true }).fill('2.41');
    await page.getByLabel('Relativity row 3', { exact: true }).press('Tab');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await page.locator('.table-snapshots > summary').click();
    await page.getByLabel('Snapshot name').fill('Manual starting point');
    await button('Save snapshot').click();
    await expect(page.getByText('Snapshot saved in the project.', { exact: true })).toBeVisible();
    for (const [mode, parameters] of [
        ['moving', async () => page.getByLabel('Smoothing window').fill('3')],
        ['isotonic', async () => page.getByLabel('Smoothing direction').selectOption('increasing')],
        ['isotonic', async () => page.getByLabel('Smoothing direction').selectOption('decreasing')],
        [
            'cap',
            async () => {
                await page.getByLabel('Relativity floor').fill('1.1');
                await page.getByLabel('Relativity cap').fill('1.2');
            },
        ],
    ]) {
        await method.selectOption(mode);
        await parameters();
        await expect(button('Apply adjustment')).toBeEnabled();
        await expect(page.getByText('Preview · not applied', { exact: true })).toBeVisible();
        await button('Apply adjustment').click();
        await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
        await button('Preview undo').click();
        await expect(button('Apply adjustment')).toBeEnabled();
        await button('Apply adjustment').click();
        await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    }
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('Region');
    await method.selectOption('moving');
    await expect(button('Apply adjustment')).toHaveCount(0);
    await page
        .getByRole('checkbox', {
            name: 'The levels of this factor are in a meaningful order',
            exact: true,
        })
        .check();
    await expect(button('Apply adjustment')).toBeEnabled();
    await expect(page.getByText('Preview · not applied', { exact: true })).toBeVisible();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
});
