import { test, expect } from '@playwright/test';
import { useFreshModelPlanProject } from './model-plan-fixture.helpers.js';

useFreshModelPlanProject(test);

const steps = (page) => page.getByRole('navigation', { name: 'Project setup steps' });
const step = (page, name) => steps(page).getByRole('button', { name: new RegExp(name) });

async function chooseSequentialRoute(page) {
    await page.goto('/');
    await step(page, 'Model structure').click();
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await expect(step(page, 'Shared columns')).toHaveAttribute('aria-current', 'step');
}

test('a fresh staged setup keeps one panel active from shared roles through named fits', async ({
    page,
}) => {
    await chooseSequentialRoute(page);
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Model plan' })).not.toBeVisible();
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    await expect(page.getByLabel('Variables JSON', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await page.getByLabel('Training fraction', { exact: true }).fill('0.75');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();

    await expect(step(page, 'Models')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('button', { name: 'Choose variables', exact: true })).toHaveCount(1);
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 name').fill('Vehicle');
    await page.getByLabel('Model 2 name').press('Tab');
    await page.getByLabel('Model 2 type').selectOption('catboost');
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 3 name').fill('Occupation');
    await page.getByLabel('Model 3 name').press('Tab');
    await page.getByLabel('Model 3 type').selectOption('catboost');
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 4 name').fill('Territory');
    await page.getByLabel('Model 4 name').press('Tab');
    await expect(page.getByText('Offset: complete score from Main effects → Vehicle.')).toBeVisible();

    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await expect(page.getByText(/Model 1 of 4 · Main effects · GLM/)).toBeVisible();
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Save variables for Main effects and continue', exact: true }).click();
    await expect(page.getByText(/Model 2 of 4 · Vehicle · CatBoost/)).toBeVisible();
    await page.getByLabel('Use VehicleAge').check();
    await page.getByRole('button', { name: 'Save variables for Vehicle and continue', exact: true }).click();
    await expect(page.getByText(/Model 3 of 4 · Occupation · CatBoost/)).toBeVisible();
    await page.getByLabel('Use Region').check();
    await page
        .getByRole('button', { name: 'Save variables for Occupation and continue', exact: true })
        .click();
    await expect(page.getByText(/Model 4 of 4 · Territory · GLM/)).toBeVisible();
    await step(page, 'Fit').click();

    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({ timeout: 120000 });
    await expect(page.getByRole('button', { name: 'Fit Model 2 — Vehicle', exact: true })).toBeEnabled();
    await page.getByLabel('Optuna trials').fill('1');
    await page.getByRole('button', { name: 'Fit Model 2 — Vehicle', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, {
        timeout: 120000,
    });
    await expect(page.getByText('2 of 4 models fitted.', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Edit model setup', exact: true }).click();
    await step(page, 'Models').click();
    await expect(page.getByLabel('Model 1 name')).toBeDisabled();
    await expect(page.getByLabel('Model 1 type')).toBeDisabled();
    await expect(page.getByLabel('Model 2 name')).toBeDisabled();
    await expect(page.getByLabel('Model 2 type')).toBeDisabled();
});

test('rapidly naming then changing a model type persists both edits', async ({ page }) => {
    await chooseSequentialRoute(page);
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 name').fill('Vehicle CatBoost');
    await page.getByLabel('Model 2 name').press('Tab');
    await page.getByLabel('Model 2 type').selectOption('catboost');
    await expect(step(page, 'Models Saved')).toBeVisible();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Model stage').selectOption({ label: 'Model 2 of 2 · Vehicle CatBoost' });
    await expect(page.getByText(/Model 2 of 2 · Vehicle CatBoost · CatBoost/)).toBeVisible();
    await page.getByLabel('Use VehicleAge').check();
    await page.getByRole('button', { name: 'Save variables for Vehicle CatBoost and open Fit', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true })).toBeVisible();
});

test('a fresh single model carries its custom CatBoost name through treatment into Fit', async ({
    page,
}) => {
    await page.goto('/');
    await step(page, 'Model structure').click();
    await page.getByRole('button', { name: 'Single model', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to variables', exact: true }).click();
    await expect(page.getByText(/Model 1 of 1 · Main effects · GLM/)).toBeVisible();
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Continue to model details', exact: true }).click();
    await page.getByLabel('Model name').fill('Single age CatBoost');
    await page.getByLabel('Fitting method').selectOption('catboost');
    await page.getByRole('button', { name: 'Continue to variable treatment', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Model details', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Save Single age CatBoost setup and open Fit', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
    await expect(
        page.getByRole('button', { name: 'Fit Model 1 — Single age CatBoost', exact: true }),
    ).toBeVisible();
});

test('discarding a dirty stage navigation keeps the unsaved variables in place', async ({ page }) => {
    await chooseSequentialRoute(page);
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Use DriverAge').check();

    page.once('dialog', async (dialog) => {
        expect(dialog.message()).toMatch(/Discard.*variable changes/i);
        await dialog.dismiss();
    });
    await step(page, 'Models').click();
    await expect(page.getByRole('region', { name: 'Variables', exact: true })).toBeVisible();
    await expect(page.getByLabel('Use DriverAge')).toBeChecked();

    page.once('dialog', async (dialog) => {
        expect(dialog.message()).toMatch(/Discard.*variable changes/i);
        await dialog.accept();
    });
    await step(page, 'Models').click();
    await expect(page.getByRole('region', { name: 'Models' })).toBeVisible();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await expect(page.getByLabel('Use DriverAge')).not.toBeChecked();
});

test('browser Back cancel keeps the dirty Variables URL and stage selection', async ({ page }) => {
    await chooseSequentialRoute(page);
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await expect(page).toHaveURL(/setup=variables/);
    await page.getByLabel('Use DriverAge').check();

    page.once('dialog', async (dialog) => {
        expect(dialog.message()).toMatch(/Discard.*variable changes/i);
        await dialog.dismiss();
    });
    await page.goBack();
    await expect(page).toHaveURL(/setup=variables/);
    await expect(page.getByLabel('Model stage')).toHaveValue(/.+/);
    await expect(page.getByLabel('Use DriverAge')).toBeChecked();
});

test('browser Back cannot switch a dirty draft to another model stage', async ({ page }) => {
    await chooseSequentialRoute(page);
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 name').fill('Vehicle');
    await page.getByLabel('Model 2 name').press('Tab');
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Model stage').selectOption({ label: 'Model 2 of 2 · Vehicle' });
    const vehicleStageId = await page.getByLabel('Model stage').inputValue();
    await expect(page).toHaveURL(new RegExp(`setup=variables.*stage=${vehicleStageId}`));
    await page.getByLabel('Use VehicleAge').check();

    page.once('dialog', async (dialog) => {
        expect(dialog.message()).toMatch(/Discard.*variable changes/i);
        await dialog.dismiss();
    });
    await page.goBack();
    await expect(page).toHaveURL(new RegExp(`setup=variables.*stage=${vehicleStageId}`));
    await expect(page.getByLabel('Model stage')).toHaveValue(vehicleStageId);
    await expect(page.getByLabel('Use VehicleAge')).toBeChecked();
});

test('a legacy Variables URL remains on Model structure after replacing data and reload', async ({ page }) => {
    await page.goto('/?view=variables');
    await expect(page).toHaveURL(/view=project.*setup=structure/);
    await expect(step(page, 'Model structure')).toHaveAttribute('aria-current', 'step');

    await step(page, 'Data').click();
    await page.getByRole('radio', { name: 'Example dataset', exact: true }).check();
    const replacement = page.getByRole('checkbox', { name: /Replace this session/ });
    if (await replacement.count()) await replacement.check();
    const opened = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/project/open') && response.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Load example', exact: true }).click();
    expect((await opened).ok()).toBe(true);
    await expect(page).toHaveURL(/view=project.*setup=structure/);
    await expect(step(page, 'Model structure')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('heading', { name: 'Model structure', exact: true })).toBeVisible();

    await page.reload();
    await expect(page).toHaveURL(/view=project.*setup=structure/);
    await expect(step(page, 'Model structure')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('heading', { name: 'Model structure', exact: true })).toBeVisible();
});
