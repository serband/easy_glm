import { test, expect } from '@playwright/test';

test('interaction-only refits visibly reuse mains; main settings changes refit', async ({ page }) => {
    const button = (name) => page.getByRole('button', { name, exact: true });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    await button('Model').click();
    await page.getByLabel('Penalty mode').selectOption('fixed');
    await page.getByLabel('Fixed alpha').fill('0.01');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.getByLabel('Interaction first factor', { exact: true }).selectOption('DriverAge');
    await page.getByLabel('Interaction second factor', { exact: true }).selectOption('Region');
    await button('Add interaction').click();
    await button('Save model settings').click();
    await button('Fit model').click();
    await expect(page.getByText('Unchanged main effects reused. Diagnostics and rate tables are ready.', { exact: true })).toBeVisible({ timeout: 45000 });
    await page.screenshot({ path: '/private/tmp/easyglm-main-effects-reuse.png', fullPage: true });
    await page.getByLabel('Fixed alpha').fill('0.02');
    await button('Save model settings').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    await expect(page.getByText('Unchanged main effects reused. Diagnostics and rate tables are ready.', { exact: true })).toHaveCount(0);
    expect(errors).toEqual([]);
});
