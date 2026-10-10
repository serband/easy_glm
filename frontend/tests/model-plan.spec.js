import { test, expect } from '@playwright/test';
import { useFreshModelPlanProject } from './model-plan-fixture.helpers.js';

useFreshModelPlanProject(test);

async function startSequential(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    let route = page.getByRole('button', { name: 'Build in stages', exact: true });
    if (!(await route.count())) {
        await page.getByRole('button', { name: 'Change model structure', exact: true }).click();
        page.once('dialog', (dialog) => dialog.accept());
        route = page.getByRole('button', { name: 'Build in stages', exact: true });
    }
    await route.click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).first().click();
}

test('numbered models keep their type, variables and cumulative offsets', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize({ width: 797, height: 900 });
    await startSequential(page);
    await expect(page.getByRole('heading', { name: 'Variables for Main effects' })).toBeVisible();
    await page.getByLabel(/^Use (DriverAge|DrivAge)$/).check();
    await page.getByRole('button', { name: 'Save model variables', exact: true }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Fit Main effects', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Fit Main effects', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({ timeout: 120000 });
    await expect(page.getByRole('button', { name: 'Diagnostics', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByText(/Actual versus expected/)).toBeVisible();
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await expect(page.getByLabel('Add stage method')).toHaveCount(0);
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await expect(page.getByLabel('Model 2 name')).toBeVisible();
    await expect(page.locator('[aria-label="Model 2 type"]')).toBeFocused();
    await page.getByLabel('Model 2 name').fill('Vehicle CatBoost');
    await page.getByLabel('Model 2 name').press('Tab');
    await expect(page.getByLabel('Model 2 name')).toHaveValue('Vehicle CatBoost');
    await page.getByLabel('Model 2 type').selectOption('catboost');
    await expect(page.getByText('Offset: complete score from Main effects.')).toBeVisible();
    await page.getByLabel('Model 2 name').locator('xpath=ancestor::li').getByRole('button', { name: 'Choose variables', exact: true }).click();
    await expect(page.getByLabel('Model stage')).toHaveValue(await page.locator('[aria-label="Model stage"] option').nth(1).getAttribute('value'));
    await page.getByLabel(/^Use (VehicleAge|VehAge)$/).check();
    await page.getByRole('button', { name: 'Save model variables', exact: true }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.getByText(/^Will use the complete score through Main effects as its offset/)).toBeVisible();
    await expect(page.getByLabel('Optuna trials')).toHaveValue('8');
    await page.getByLabel('Optuna trials').fill('1');
    await page.getByRole('button', { name: /Fit Vehicle CatBoost/ }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, { timeout: 120000 });
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 3 name').fill('Occupation CatBoost');
    await page.getByLabel('Model 3 name').press('Tab');
    await expect(page.getByLabel('Model 3 name')).toHaveValue('Occupation CatBoost');
    await page.getByLabel('Model 3 type').selectOption('catboost');
    await expect(
        page.getByText('Offset: complete score from Main effects → Vehicle CatBoost.'),
    ).toBeVisible();
    expect(errors).toEqual([]);
});

test('plan picker guards unsaved work and remains narrow', async ({ page }) => {
    await page.setViewportSize({ width: 797, height: 900 });
    await startSequential(page);
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).first().click();
    await page.getByLabel(/^Use (DriverAge|DrivAge)$/).check();
    page.once('dialog', (dialog) => dialog.dismiss());
    await page.getByLabel('Model stage').selectOption({ index: 1 });
    await expect(page.getByLabel('Model stage')).toHaveValue(await page.locator('select[aria-label="Model stage"] option').first().getAttribute('value'));
    expect(await page.locator('.plan-variables').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
});
