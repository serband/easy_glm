import { test, expect } from '@playwright/test';

async function fitSingleModel(page) {
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Create model', exact: true }).click();
    await page.getByRole('button', { name: 'Fit model', exact: true }).first().click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 60000 });
}

async function fillDraft(page, { name, component, kind, variable }) {
    await page.getByLabel('Stacked model name').fill(name);
    await page.getByLabel('Model component type').selectOption(kind);
    await page.getByLabel(`Use ${variable}`).check();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
    if (kind === 'catboost') {
        await page.getByLabel('CatBoost iterations').fill('20');
        await page.getByLabel('CatBoost time limit').fill('1');
    }
}

test('model builder fits a model where its settings are edited', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize({ width: 1332, height: 1225 });
    await page.goto('/');
    await fitSingleModel(page);
    await page.getByRole('button', { name: 'Build a model in stages' }).click();

    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await expect(page.getByRole('heading', { name: 'First model in the chain', exact: true })).toBeVisible();
    await fillDraft(page, { name: 'Flow Root GLM', component: 'Root region', kind: 'glm', variable: 'Region' });
    let releaseJobsGate;
    const jobsGate = new Promise((resolve) => {
        releaseJobsGate = resolve;
    });
    let holdFirstJobsRead = true;
    const holdJobsRoute = async (route) => {
        if (holdFirstJobsRead) {
            holdFirstJobsRead = false;
            await jobsGate;
        }
        await route.continue();
    };
    await page.route('**/api/jobs', holdJobsRoute);
    await page.getByRole('button', { name: 'Fit model', exact: true }).first().click();
    await expect(page.getByRole('status').filter({ hasText: /Starting fit|Fitting:/ })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'First model in the chain', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Edit draft', exact: true })).toHaveCount(0);
    releaseJobsGate();
    await page.waitForTimeout(0);
    await page.unroute('**/api/jobs', holdJobsRoute);
    await expect(page.getByRole('heading', { name: 'Flow Root GLM', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.getByText('Fitted', { exact: true }).first()).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.model-list-item').filter({ has: page.locator('strong', { hasText: 'Flow Root GLM' }) })).toContainText('Fitted', { timeout: 60000 });
    const selectedHeader = page.locator('.selected-model-header');
    const identity = selectedHeader.locator('.selected-model-identity');
    const context = selectedHeader.locator('.selected-model-context');
    const actions = selectedHeader.locator('.selected-model-actions');
    const [identityBox, contextBox, actionBox] = await Promise.all([
        identity.boundingBox(), context.boundingBox(), actions.boundingBox(),
    ]);
    expect(identityBox.width).toBeGreaterThan(300);
    expect(contextBox.y).toBeGreaterThan(identityBox.y);
    expect(actionBox.y).toBeGreaterThan(contextBox.y);

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByLabel('Model', { exact: true })).toBeVisible();
    const [mobileIdentity, mobileContext, mobileActions] = await Promise.all([
        identity.boundingBox(), context.boundingBox(), actions.boundingBox(),
    ]);
    expect(mobileContext.y).toBeGreaterThan(mobileIdentity.y);
    expect(mobileActions.y).toBeGreaterThan(mobileContext.y);
    expect(await page.locator('.model-builder').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
    await page.setViewportSize({ width: 1332, height: 1225 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await expect(page.getByText('This model is fitted after Flow Root GLM.', { exact: true })).toBeVisible();
    await fillDraft(page, { name: 'Flow CatBoost', component: 'Vehicle correction', kind: 'catboost', variable: 'VehicleAge' });
    await page.getByRole('button', { name: 'Fit model', exact: true }).first().click();
    await expect(page.getByRole('heading', { name: 'Flow CatBoost', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.getByText('Built on Flow Root GLM. Its complete fitted score is the offset for this model.', { exact: true })).toBeVisible();
    await expect(page.locator('.model-list-item').filter({ has: page.locator('strong', { hasText: 'Flow CatBoost' }) })).toContainText('Built on Flow Root GLM');

    await page.getByRole('button', { name: 'Revise model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Revise Flow CatBoost', exact: true })).toBeVisible();
    await expect(page.getByLabel('Stacked model name')).toHaveValue('Flow CatBoost revised');
    await page.getByRole('button', { name: 'Save draft', exact: true }).first().click();
    await expect(page.getByText('Draft saved.', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();
    await page.getByLabel('CatBoost iterations').fill('21');
    await page.getByLabel('Search model variables').fill('Region');
    await page.getByLabel('Use Region').check();
    await page.locator('.model-list-item').filter({ has: page.locator('strong', { hasText: 'Flow Root GLM' }) }).click();
    await expect(page.getByRole('dialog', { name: 'Unsaved draft' })).toBeVisible();
    await page.getByRole('button', { name: 'Keep editing', exact: true }).click();
    await expect(page.getByLabel('Search model variables')).toHaveValue('Region');

    await page.setViewportSize({ width: 390, height: 844 });
    await expect(page.getByLabel('Model', { exact: true })).toBeVisible();
    expect(await page.locator('.model-builder').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);

    await page.getByLabel('Model', { exact: true }).selectOption('Flow Root GLM');
    await expect(page.getByRole('dialog', { name: 'Unsaved draft' })).toBeVisible();
    await page.getByRole('dialog', { name: 'Unsaved draft' }).getByRole('button', { name: 'Keep editing', exact: true }).click();
    await expect(page.getByLabel('Model', { exact: true })).toHaveValue('Flow CatBoost revised');

    await page.getByLabel('Page', { exact: true }).selectOption('variables');
    await expect(page.getByRole('dialog', { name: 'Unsaved draft' })).toBeVisible();
    await page.getByRole('dialog', { name: 'Unsaved draft' }).getByRole('button', { name: 'Keep editing', exact: true }).click();
    await expect(page.getByLabel('Page', { exact: true })).toHaveValue('model');

    await page.getByLabel('Page', { exact: true }).selectOption('variables');
    await expect(page.getByRole('dialog', { name: 'Unsaved draft' })).toBeVisible();
    await page.getByRole('dialog', { name: 'Unsaved draft' }).getByRole('button', { name: 'Discard changes', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();

    expect(errors).toEqual([]);
});
