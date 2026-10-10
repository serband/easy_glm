import { test, expect } from '@playwright/test';
import { useFreshCoreInteractionsProject } from './core-interactions-fixture.helpers.js';
import { saveAndFitLegacyGlm } from './legacy-models.helpers.js';

useFreshCoreInteractionsProject(test);

const steps = (page) => page.getByRole('navigation', { name: 'Project setup steps' });
const step = (page, name) => steps(page).getByRole('button', { name: new RegExp(name) });
const modelRow = (page, index, name) =>
    page
        .locator('.fit-stages > li')
        .filter({ has: page.locator('.fit-stage-row strong', { hasText: `${index}. ${name}` }) });

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

async function createTwoGlmPlan(page) {
    await page.goto('/');
    await step(page, 'Model structure').click();
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 name').fill('Vehicle');
    await page.getByLabel('Model 2 name').press('Tab');
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
}

async function ensureInteractionsOpen(page, modelName = 'Main effects') {
    const review = page.getByRole('region', { name: `Review ${modelName}`, exact: true });
    if (!(await review.isVisible())) {
        await expect(review)
            .toBeVisible({ timeout: 5000 })
            .catch(() => undefined);
    }
    if (!(await review.isVisible())) {
        await page.getByRole('button', { name: 'Manage interactions', exact: true }).click();
    }
    await expect(review).toBeVisible();
    return review;
}

async function fitMainEffects(page) {
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({
        timeout: 120000,
    });
    const review = await ensureInteractionsOpen(page);
    await expect(review.getByRole('region', { name: 'Interactions', exact: true })).toBeVisible();
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

test('each fitted GLM keeps its review directly below its own row and preserves a hidden draft', async ({
    page,
}) => {
    await createTwoGlmPlan(page);
    await fitMainEffects(page);

    const mainReview = await ensureInteractionsOpen(page);
    // A new, unfitted candidate is the draft that must survive switching review rows.
    await mainReview.getByText('Interaction fitting and table bands', { exact: true }).click();
    await mainReview.getByLabel('Fitting approach').selectOption('manual');
    await mainReview.getByLabel('Pair first column').selectOption('AnnualMileage');
    await mainReview.getByLabel('Pair second column').selectOption('DriverAge');
    await mainReview.getByRole('button', { name: 'Add interaction', exact: true }).click();
    await expect(mainReview.getByRole('region', { name: 'Interaction candidate' })).toContainText(
        'Not included in score.',
    );

    await page.getByRole('button', { name: 'Continue to Vehicle', exact: true }).click();
    await page.getByRole('button', { name: 'Fit Model 2 — Vehicle', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, {
        timeout: 120000,
    });

    const vehicleReview = page.getByRole('region', { name: 'Review Vehicle', exact: true });
    await expect(mainReview).toBeVisible();
    await expect(vehicleReview).toBeHidden();
    await expect(
        mainReview.getByRole('heading', { name: 'Review Main effects', exact: true }),
    ).toBeVisible();
    await expect(
        mainReview.getByRole('button', { name: 'Continue to Vehicle', exact: true }),
    ).toBeVisible();
    await expect(
        vehicleReview.getByRole('button', { name: 'Continue to Vehicle', exact: true }),
    ).toHaveCount(0);

    await modelRow(page, 1, 'Main effects')
        .getByRole('button', { name: 'Hide interactions', exact: true })
        .click();
    await expect(mainReview).toBeHidden();
    await modelRow(page, 2, 'Vehicle')
        .getByRole('button', { name: 'Manage interactions', exact: true })
        .click();
    await expect(vehicleReview).toBeVisible();
    const rowLayout = await page.locator('.fit-stages > li').evaluateAll((rows) =>
        rows.map((row) => ({
            title: row.querySelector('.fit-stage-row strong')?.textContent,
            review: row
                .querySelector('[aria-label="Review Main effects"], [aria-label="Review Vehicle"]')
                ?.getAttribute('aria-label'),
        })),
    );
    expect(rowLayout).toEqual([
        { title: '1. Main effects', review: 'Review Main effects' },
        { title: '2. Vehicle', review: 'Review Vehicle' },
    ]);
    await modelRow(page, 1, 'Main effects')
        .getByRole('button', { name: 'Manage interactions', exact: true })
        .click();
    await expect(mainReview).toBeVisible();
    await expect(mainReview.getByRole('region', { name: 'Interaction candidate' })).toContainText(
        'Not included in score.',
    );
    const mainSession = await (await page.request.get('/api/session')).json();
    const mainPlan = await (
        await page.request.get('/api/model-plan', {
            headers: { 'X-EasyGLM-Token': mainSession.token },
        })
    ).json();
    const mainModelId = mainPlan.plan.models.find(
        (model) => model.name === 'Main effects',
    ).model_id;
    const mainInteractions = await (
        await page.request.get(`/api/core-models/${mainModelId}/interactions`, {
            headers: { 'X-EasyGLM-Token': mainSession.token },
        })
    ).json();
    expect(mainInteractions.candidates).toEqual(
        expect.arrayContaining([
            expect.objectContaining({ a: 'AnnualMileage', b: 'DriverAge', status: 'draft' }),
        ]),
    );
    await modelRow(page, 1, 'Main effects')
        .getByRole('button', { name: 'Hide interactions', exact: true })
        .click();
    await vehicleReview.getByRole('button', { name: 'Find interactions', exact: true }).click();
    await expect(
        vehicleReview.getByRole('region', { name: 'Interaction discovery results', exact: true }),
    ).toBeVisible({ timeout: 120000 });

    const session = await (await page.request.get('/api/session')).json();
    const plan = await (
        await page.request.get('/api/model-plan', { headers: { 'X-EasyGLM-Token': session.token } })
    ).json();
    const vehicleFit = plan.stage_statuses.find((status) => status.name === 'Vehicle').fit
        .model_name;
    await vehicleReview.getByRole('button', { name: 'Check A/E', exact: true }).click();
    await expect(page.getByLabel('Diagnostic saved model')).toHaveValue(vehicleFit);

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await ensureInteractionsOpen(page);
    await expect(mainReview).toBeVisible();
    await expect(mainReview.getByRole('region', { name: 'Interaction candidate' })).toContainText(
        'Not included in score.',
    );
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

test('starting a candidate clears a held factor search and a fresh search owns the result', async ({
    page,
}) => {
    await createGlmThenVehiclePlan(page);
    await fitMainEffects(page);
    const review = await ensureInteractionsOpen(page);
    await review.getByLabel('Pair first column').selectOption('AnnualMileage');
    await review.getByLabel('Pair second column').selectOption('DriverAge');

    let releaseOldSearch;
    let oldSearchStarted = false;
    await page.route('**/core-models/**/search', async (route) => {
        if (!oldSearchStarted && route.request().postDataJSON()?.action === 'factors') {
            oldSearchStarted = true;
            await new Promise((resolve) => {
                releaseOldSearch = resolve;
            });
        }
        await route.continue();
    });
    await review.getByRole('button', { name: 'Find missing variables', exact: true }).click();
    await expect.poll(() => oldSearchStarted).toBe(true);
    await expect(review.getByText('Searching training rows…', { exact: true })).toBeVisible();

    await review.getByRole('button', { name: 'Add interaction', exact: true }).click();
    await expect(review.getByText('Searching training rows…', { exact: true })).toHaveCount(0);
    await expect(review.getByRole('region', { name: 'Interaction candidate' })).toContainText(
        'Not included in score.',
    );

    await review.getByRole('button', { name: 'Find missing variables', exact: true }).click();
    await expect(
        review.getByRole('region', { name: 'Interaction discovery results', exact: true }),
    ).toBeVisible({ timeout: 120000 });
    releaseOldSearch();
    await expect(
        review.getByRole('region', { name: 'Interaction discovery results', exact: true }),
    ).toBeVisible();
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
    const reloadedReview = await ensureInteractionsOpen(page);
    await expect(
        reloadedReview.getByRole('list', { name: 'Accepted interaction tables' }),
    ).toContainText('1. AnnualMileage × DriverAge');
    await expect(
        reloadedReview.getByRole('list', { name: 'Accepted interaction tables' }),
    ).toContainText('2. VehicleAge × Region');
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
    await expect(modelRow(page, 2, 'Vehicle')).toContainText(
        "Refit the earlier model's remaining interaction tables first",
    );
});

test('an upstream score revision invalidates a hidden downstream GLM candidate', async ({
    page,
}) => {
    await createTwoGlmPlan(page);
    await fitMainEffects(page);
    await addAndFitInteraction(page, 'AnnualMileage', 'DriverAge');
    await page.getByRole('button', { name: 'Keep interaction', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to Vehicle', exact: true }).click();
    await step(page, 'Fit').click();
    await page.getByRole('button', { name: 'Fit Model 2 — Vehicle', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, {
        timeout: 120000,
    });

    const mainReview = await ensureInteractionsOpen(page);
    const vehicleReview = page.getByRole('region', { name: 'Review Vehicle', exact: true });
    await page.getByRole('button', { name: 'Manage interactions', exact: true }).click();
    await expect(vehicleReview).toBeVisible();
    await vehicleReview.getByLabel('Pair first column').selectOption('AnnualMileage');
    await vehicleReview.getByLabel('Pair second column').selectOption('VehicleAge');
    await vehicleReview.getByRole('button', { name: 'Add interaction', exact: true }).click();
    await expect(
        vehicleReview.getByRole('region', { name: 'Interaction candidate' }),
    ).toContainText('Not included in score.');
    let releaseOldSearch;
    let oldSearchStarted = false;
    await page.route('**/core-models/**/search', async (route) => {
        if (!oldSearchStarted && route.request().postDataJSON()?.action === 'factors') {
            oldSearchStarted = true;
            await new Promise((resolve) => {
                releaseOldSearch = resolve;
            });
        }
        await route.continue();
    });
    await vehicleReview
        .getByRole('button', { name: 'Find missing variables', exact: true })
        .click();
    await expect.poll(() => oldSearchStarted).toBe(true);
    await modelRow(page, 2, 'Vehicle')
        .getByRole('button', { name: 'Hide interactions', exact: true })
        .click();
    await expect(vehicleReview).toBeHidden();

    const accepted = mainReview.getByRole('list', { name: 'Accepted interaction tables' });
    await accepted.getByRole('button', { name: 'Amend table', exact: true }).click();
    const amendment = mainReview.getByRole('region', {
        name: 'Amend interaction table',
        exact: true,
    });
    await amendment.getByLabel('Lower cap').fill('0.9');
    await amendment.getByLabel('Upper cap').fill('1.1');
    await amendment.getByRole('button', { name: 'Preview table amendment', exact: true }).click();
    await page.once('dialog', (dialog) => dialog.accept());
    await amendment.getByRole('button', { name: 'Apply table amendment', exact: true }).click();

    await expect(modelRow(page, 2, 'Vehicle')).toContainText('Needs refit');
    releaseOldSearch();

    await expect(vehicleReview).toHaveCount(0);
    const postAmendSession = await (await page.request.get('/api/session')).json();
    const postAmendPlan = await (
        await page.request.get('/api/model-plan', {
            headers: { 'X-EasyGLM-Token': postAmendSession.token },
        })
    ).json();
    const vehicleModelId = postAmendPlan.plan.models.find(
        (model) => model.name === 'Vehicle',
    ).model_id;
    const staleCandidate = await (
        await page.request.get(`/api/core-models/${vehicleModelId}/interactions`, {
            headers: { 'X-EasyGLM-Token': postAmendSession.token },
        })
    ).json();
    expect(staleCandidate.candidates).toEqual(
        expect.arrayContaining([expect.objectContaining({ status: 'needs_rebuild' })]),
    );

    await page.getByRole('button', { name: 'Fit Model 2 — Vehicle', exact: true }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, {
        timeout: 120000,
    });
    await modelRow(page, 2, 'Vehicle')
        .getByRole('button', { name: 'Manage interactions', exact: true })
        .click();
    await expect(vehicleReview).toBeVisible();
    await expect(
        vehicleReview.getByRole('region', { name: 'Interaction candidate' }),
    ).toContainText('needs rebuild because its preceding accepted score has changed');
    await expect(
        vehicleReview.getByRole('button', { name: 'Keep interaction', exact: true }),
    ).toBeDisabled();
    await vehicleReview
        .getByRole('button', { name: 'Find missing variables', exact: true })
        .click();
    await expect(
        vehicleReview.getByRole('region', { name: 'Interaction discovery results', exact: true }),
    ).toBeVisible({ timeout: 120000 });
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
