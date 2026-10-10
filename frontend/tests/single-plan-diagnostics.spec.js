import { expect, test } from '@playwright/test';
import { useFreshModelPlanProject } from './model-plan-fixture.helpers.js';

useFreshModelPlanProject(test);

test('a planned Single model shows holdout A/E only after it is selected', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: 'Single model', exact: true }).click();

    await page.getByLabel(/^Use (DriverAge|DrivAge)$/).check();
    await page.getByRole('button', { name: 'Continue to model details', exact: true }).click();
    await page
        .getByRole('button', { name: 'Continue to variable treatment', exact: true })
        .click();
    await page.getByRole('button', { name: 'Save model setup', exact: true }).click();

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Fit Main effects', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({
        timeout: 120000,
    });

    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    const plots = page.locator('.diagnostic-plot h3');
    await expect(plots.first()).toContainText(/Training/);
    await expect(plots.filter({ hasText: /holdout/i })).toHaveCount(0);

    await page.getByLabel('Diagnostic subset').selectOption('holdout');
    await expect(plots.first()).toContainText(/Holdout/);
    await expect(plots.filter({ hasText: /holdout/i })).toHaveCount(1);
});
