import { test, expect } from '@playwright/test';

test('Variables → split/model → background fit → diagnostics → rate tables → invalidation', async ({
    page,
}) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/');
    await expect(page.getByLabel('Role for Claims', { exact: true })).toHaveValue('target');
    await expect(page.getByRole('button', { name: 'Diagnostics', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Design & models', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Design & models', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeDisabled();
    await page.getByLabel('Split method', { exact: true }).selectOption('random');
    await page.getByRole('button', { name: 'Apply split', exact: true }).click();
    await expect(page.getByText('Split applied.', { exact: true })).toBeVisible();
    await page.getByLabel('Design kind for DriverAge', { exact: true }).selectOption('continuous');
    await page.getByRole('button', { name: 'Create model', exact: true }).click();
    await expect(page.getByLabel('Model selection', { exact: true })).toHaveValue('Frequency');
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeEnabled();
    await page.screenshot({ path: 'test-results/model-setup.png', fullPage: true });
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    const started = performance.now();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByLabel('Role for Claims', { exact: true })).toBeVisible();
    expect(performance.now() - started).toBeLessThan(1000);
    await page.getByRole('button', { name: 'Design & models', exact: true }).click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Diagnostics', exact: true })).toBeVisible();
    await expect(
        page.getByRole('img', { name: 'Actual and expected lift on holdout', exact: true }),
    ).toBeVisible();
    await page.screenshot({ path: 'test-results/diagnostics.png', fullPage: true });
    await page.getByLabel('Diagnostic subset', { exact: true }).selectOption('train');
    await expect(
        page.getByRole('img', { name: 'Actual and expected lift on train', exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Rate tables', exact: true }).click();
    await expect(page.getByLabel('Rate table variable', { exact: true })).toBeVisible();
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('DriverAge');
    await expect(page.locator('.rate-grid')).toContainText('relativity');
    await expect(page.locator('.rate-grid')).toContainText('slope');
    await page.screenshot({ path: 'test-results/rate-tables.png', fullPage: true });
    await page.getByRole('button', { name: /^Variables/ }).click();
    await page.getByLabel('Role for VehicleAge', { exact: true }).selectOption('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Diagnostics', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Rate tables', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: 'Design & models', exact: true }).click();
    await expect(page.getByText('Fit needs updating', { exact: true })).toBeVisible();
    expect(errors).toEqual([]);
});

test('unapplied Variables draft survives navigation through model setup', async ({ page }) => {
    await page.goto('/');
    await page.getByLabel('Name for Claims', { exact: true }).fill('UnappliedClaims');
    await page.getByLabel('Name for Claims', { exact: true }).press('Tab');
    await page.getByRole('button', { name: 'Design & models', exact: true }).click();
    await expect(page.getByLabel('Model target', { exact: true })).toHaveValue('Claims');
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByLabel('Name for Claims', { exact: true })).toHaveValue(
        'UnappliedClaims',
    );
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
});
