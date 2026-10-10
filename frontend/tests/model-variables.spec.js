import { test, expect } from '@playwright/test';

async function openBuilder(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
}

test('each model keeps its own variables and local definitions', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Age base');
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Use custom cuts', exact: true }).click();
    const cuts = page.getByLabel('Custom cuts for DriverAge');
    await cuts.fill('25, 35, 45, 55');
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Age base', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    const inherited = page.getByRole('region', { name: 'Inherited model definition' });
    await expect(inherited).toContainText('OutcomeClaims, poisson with log link');
    await expect(inherited).toContainText('ExposureExposure');
    await expect(inherited).toContainText('70% training (seed 42)');
    await expect(inherited).toContainText('Complete score from Age base');

    await page.getByLabel('Stacked model name').fill('Vehicle correction');
    await page.getByRole('tab', { name: 'JSON' }).click();
    const json = page.getByLabel('Model variables JSON');
    const protectedTarget = 'Claims';
    await json.fill(JSON.stringify({ variables: { [protectedTarget]: { kind: 'step' } } }, null, 2));
    await expect(page.getByRole('alert')).toContainText(`${protectedTarget} is not available`);
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeDisabled();
    await json.fill(JSON.stringify({
        defaults: { n_bins: 6 },
        variables: { VehicleAge: { kind: 'step', knots: [1, 3, 6, 10] } },
    }, null, 2));
    await page.getByRole('button', { name: 'Apply variable JSON', exact: true }).click();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
    await page.getByRole('tab', { name: 'Table' }).click();
    await expect(page.getByLabel('Use VehicleAge')).toBeChecked();
    await expect(page.getByLabel('Use DriverAge')).not.toBeChecked();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Vehicle correction', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
});

test('a catalogue stage uses uploaded attributes and keeps its join key out of the factors', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Catalogue base');
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Catalogue base', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await page.getByLabel('Stacked model name').fill('Vehicle file');
    await page.getByLabel('Model component type').selectOption('catalogue');
    const csv = ['PolicyID,VehicleGroup,Power']
        .concat(Array.from({ length: 12000 }, (_, index) => `${index},${index % 3 ? 'Family' : 'Sport'},${70 + index % 90}`))
        .join('\n');
    await page.getByLabel('Upload catalogue').setInputFiles({
        name: 'vehicle-file.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from(csv),
    });
    await page.getByLabel('Catalogue key').selectOption('PolicyID');
    await expect(page.getByLabel('Use PolicyID')).toHaveCount(0);
    await page.getByLabel('Use VehicleGroup').check();
    await page.getByLabel('CatBoost iterations').fill('20');
    await page.getByLabel('CatBoost time limit').fill('1');
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Vehicle file', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
    await page.getByRole('button', { name: 'Inspect and amend tables', exact: true }).click();
    await page.getByLabel('Inspect model component').selectOption({ label: 'Vehicle file' });
    await expect(page.getByLabel('Inspect model component')).toHaveValue(/.+/);
});
