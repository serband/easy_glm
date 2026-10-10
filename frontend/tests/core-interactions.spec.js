import { test, expect } from '@playwright/test';
import { useFreshCoreInteractionsProject } from './core-interactions-fixture.helpers.js';
import { saveAndFitLegacyGlm } from './legacy-models.helpers.js';

useFreshCoreInteractionsProject(test);

const steps = (page) => page.getByRole('navigation', { name: 'Project setup steps' });
const step = (page, name) => steps(page).getByRole('button', { name: new RegExp(name) });

async function createGlmThenVehiclePlan(page) {
    await page.goto('/');
    await step(page, 'Model structure').click();
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 name').fill('Vehicle');
    await page.getByLabel('Model 2 name').press('Tab');
    await page.getByLabel('Model 2 type').selectOption('catboost');
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Use DriverAge').check();
    await page.getByLabel('Use Region').check();
    await page
        .getByRole('button', { name: 'Save variables for Main effects and continue', exact: true })
        .click();
    await page.getByLabel('Use VehicleAge').check();
    await page
        .getByRole('button', { name: 'Save variables for Vehicle and open Fit', exact: true })
        .click();
    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
}

async function fitMainEffects(page) {
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({
        timeout: 120000,
    });
    await page.getByRole('button', { name: 'Manage interactions', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Interactions', exact: true })).toBeVisible();
}

async function addAndFitInteraction(page, first, second) {
    await page.getByText('Interaction fitting and table bands', { exact: true }).click();
    await page.getByLabel('Fitting approach').selectOption('manual');
    await page.getByLabel('Iterations').fill('12');
    await page.getByLabel('Depth').fill('2');
    await page.getByLabel('Learning rate').fill('0.1');
    await page.getByLabel('Pair first column').selectOption(first);
    await page.getByLabel('Pair second column').selectOption(second);
    await page.getByRole('button', { name: 'Add interaction', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Interaction candidate' })).toContainText(
        'Not included in score.',
    );
    await page.getByRole('button', { name: 'Fit interaction', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Keep interaction', exact: true })).toBeEnabled({
        timeout: 120000,
    });
    const candidate = page.getByRole('region', { name: 'Interaction candidate' });
    await candidate.getByText('Interaction table cells', { exact: true }).click();
    await expect(candidate.locator('tbody tr').first()).toBeVisible();
}

test('a planned GLM owns optional pair corrections rather than an Interaction model type', async ({
    page,
}) => {
    await page.goto('/');
    await step(page, 'Model structure').click();
    await expect(page.getByRole('button', { name: 'Interaction', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Build in stages', exact: true })).toBeVisible();

    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    await expect(page.getByRole('region', { name: 'Interactions', exact: true })).toContainText(
        'No interaction tables have been accepted.',
    );
    await expect(
        page.getByRole('button', { name: 'Continue to Vehicle', exact: true }),
    ).toBeVisible();
});

test('an interaction-only candidate remains isolated until Keep then advances the Vehicle offset', async ({
    page,
}) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);

    await addAndFitInteraction(page, 'AnnualMileage', 'DriverAge');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();
    const accepted = page.getByRole('list', { name: 'Accepted interaction tables' });
    await expect(accepted).toContainText('AnnualMileage × DriverAge');
    const session = await (await page.request.get('/api/session')).json();
    const plan = await (
        await page.request.get('/api/model-plan', { headers: { 'X-EasyGLM-Token': session.token } })
    ).json();
    expect(plan.plan.models[0].core.factors).not.toContain('AnnualMileage');
    await expect(
        page.getByRole('button', { name: 'Continue to Vehicle', exact: true }),
    ).toBeVisible();

    await page.getByRole('button', { name: 'Continue to Vehicle', exact: true }).click();
    const vehicleVariables = page.getByRole('region', { name: 'Variables', exact: true });
    await expect(vehicleVariables).toContainText('Variables for Vehicle');
    await expect(vehicleVariables).toContainText('AnnualMileage × DriverAge');
    await step(page, 'Fit').click();
    await expect(
        page.getByRole('heading', { name: 'Fit Model 2 — Vehicle', exact: true }),
    ).toBeVisible();
});

test('discarding a fitted candidate leaves the accepted scorer and downstream fit eligibility unchanged', async ({
    page,
}) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    await addAndFitInteraction(page, 'AnnualMileage', 'DriverAge');
    await page.getByRole('button', { name: 'Discard interaction', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Interactions', exact: true })).toContainText(
        'No interaction tables have been accepted.',
    );
    await expect(
        page.getByRole('button', { name: 'Continue to Vehicle', exact: true }),
    ).toBeVisible();
});

test('interaction search returns a usable training-data pair', async ({ page }) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    await page.getByRole('button', { name: 'Find interactions', exact: true }).click();
    const results = page.getByRole('region', {
        name: 'Interaction discovery results',
        exact: true,
    });
    await expect(results).toBeVisible({ timeout: 120000 });
    await expect(results.locator('tbody tr').first()).toBeVisible();
    await results.getByRole('button', { name: 'Use for interaction', exact: true }).first().click();
    await expect(page.getByLabel('Pair first column')).not.toHaveValue('');
    await expect(page.getByLabel('Pair second column')).not.toHaveValue('');
    await page.getByRole('button', { name: 'Add interaction', exact: true }).click();
    await expect(
        page.getByRole('region', { name: 'Interaction candidate', exact: true }),
    ).toContainText('Not included in score.');
});

test('two accepted corrections persist in their ordered parent list after reload', async ({
    page,
}) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    await addAndFitInteraction(page, 'AnnualMileage', 'DriverAge');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();
    await addAndFitInteraction(page, 'VehicleAge', 'Region');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();
    const accepted = page.getByRole('list', { name: 'Accepted interaction tables' });
    await expect(accepted).toContainText('1. AnnualMileage × DriverAge');
    await expect(accepted).toContainText('2. VehicleAge × Region');
    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Manage interactions', exact: true }).click();
    await expect(page.getByRole('list', { name: 'Accepted interaction tables' })).toContainText(
        '1. AnnualMileage × DriverAge',
    );
    await expect(page.getByRole('list', { name: 'Accepted interaction tables' })).toContainText(
        '2. VehicleAge × Region',
    );
});

test('amending an accepted pair previews the revision and marks its later correction for refit', async ({
    page,
}) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    await addAndFitInteraction(page, 'AnnualMileage', 'DriverAge');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();
    await addAndFitInteraction(page, 'VehicleAge', 'Region');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();

    const accepted = page.getByRole('list', { name: 'Accepted interaction tables' });
    await accepted.getByRole('button', { name: 'Amend table', exact: true }).first().click();
    const amendment = page.getByRole('region', { name: 'Amend interaction table', exact: true });
    await amendment.getByLabel('Lower cap').fill('0.9');
    await amendment.getByLabel('Upper cap').fill('1.1');
    await amendment.getByRole('button', { name: 'Preview table amendment', exact: true }).click();
    await expect(amendment).toContainText('Preview ready. Applying creates a new scorer');
    await page.once('dialog', (dialog) => dialog.accept());
    await amendment.getByRole('button', { name: 'Apply table amendment', exact: true }).click();
    await expect(page.getByRole('list', { name: 'Accepted interaction tables' })).toContainText(
        'VehicleAge × RegionNeeds refit',
    );
    await expect(
        page.getByText(/Vehicle needs refitting after this table amendment/),
    ).toBeVisible();
});

test('a fitted legacy scorer can be adopted through Project & data as an immutable v2 root', async ({
    page,
}) => {
    const savedName = `Adopted legacy ${Date.now()}`;
    await page.goto('/');
    await saveAndFitLegacyGlm(page, savedName, ['DriverAge']);
    await page.reload();
    await step(page, 'Model structure').click();
    await page.getByRole('button', { name: 'Use as current plan', exact: true }).click();
    await expect(
        page.getByRole('region', { name: 'Shared column assignments', exact: true }),
    ).toBeVisible();
    await step(page, 'Models').click();
    await expect(page.getByText('Saved starting model', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Saved model name')).toHaveValue(savedName);
    await expect(page.getByLabel('Saved model name')).toHaveAttribute('readonly', '');
    await expect(page.getByLabel('Saved model type')).toHaveValue('Saved fitted model');

    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 1 name').fill('After adopted scorer');
    await page.getByLabel('Model 1 name').press('Tab');
    await page.getByLabel('Model 1 type').selectOption('catboost');
    await expect(
        page
            .getByRole('region', { name: 'Model plan', exact: true })
            .getByText(`Offset: complete score from ${savedName}.`, { exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await expect(page.getByRole('region', { name: 'Variables', exact: true })).toContainText(
        `Offset: complete score from ${savedName}.`,
    );
});
