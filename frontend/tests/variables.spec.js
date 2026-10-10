import { test, expect } from '@playwright/test';
import { useFreshModelPlanProject } from './model-plan-fixture.helpers.js';

useFreshModelPlanProject(test);

const setupSteps = (page) => page.getByRole('navigation', { name: 'Project setup steps' });
const setupStep = (page, name) => setupSteps(page).getByRole('button', { name: new RegExp(name) });

async function startPlan(page, route = 'Build in stages') {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Model structure').click();
    const routeButton = page.getByRole('button', { name: route, exact: true });
    await expect(routeButton).toBeEnabled({ timeout: 15000 });
    const saved = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/model-plan') && response.request().method() === 'POST',
        { timeout: 15000 },
    );
    await routeButton.click();
    const savedResponse = await saved;
    expect(savedResponse.ok()).toBe(true);
    const savedBody = await savedResponse.json();
    expect(savedBody.plan, JSON.stringify(savedBody)).toBeTruthy();
    expect(savedBody.revision, JSON.stringify(savedBody)).toEqual(expect.any(Number));
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible({
        timeout: 15000,
    });
}

async function modelPlan(page) {
    const session = await page.request.get('/api/session').then((response) => response.json());
    const response = await page.request.get('/api/model-plan', {
        headers: { 'X-EasyGLM-Token': session.token },
    });
    expect(response.ok()).toBe(true);
    return response.json();
}

test('a legacy Variables bookmark opens Project & data at the model setup', async ({ page }) => {
    await page.goto('/?view=variables');

    await expect(page).toHaveURL(/view=project.*setup=structure/);
    await expect(page.getByRole('heading', { name: 'Project & data', exact: true })).toBeVisible();
    await expect(page.getByRole('navigation', { name: 'Workbench' }).getByRole('button', {
        name: /^Variables/,
    })).toHaveCount(0);
    await expect(setupStep(page, 'Model structure')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Build in stages', exact: true })).toBeVisible();
});

test('starting a route wins over a delayed initial model-plan read', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    let initialResponse;
    let releaseInitialRead;
    await page.route('**/api/model-plan', async (route) => {
        if (route.request().method() !== 'GET' || initialResponse) {
            await route.continue();
            return;
        }
        const response = await route.fetch();
        initialResponse = {
            status: response.status(),
            headers: response.headers(),
            body: await response.body(),
        };
        await new Promise((resolve) => {
            releaseInitialRead = resolve;
        });
        await route.fulfill(initialResponse);
    });
    await page.goto('/');
    await expect.poll(() => Boolean(initialResponse)).toBe(true);
    await setupStep(page, 'Model structure').click();
    const routeButton = page.getByRole('button', { name: 'Build in stages', exact: true });
    await expect(routeButton).toBeEnabled();
    const saved = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/model-plan') && response.request().method() === 'POST',
    );
    await routeButton.click();
    expect((await saved).ok()).toBe(true);
    releaseInitialRead();
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible();
    expect(errors).toEqual([]);
});

test('a new plan deliberately chooses a different route before replacing the current one', async ({
    page,
}) => {
    await page.goto('/');
    await startPlan(page, 'Single model');
    await expect(setupStep(page, 'Variables')).toBeVisible();
    await setupStep(page, 'Model structure').click();
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Keep current plan', exact: true }).click();
    await expect(setupStep(page, 'Variables')).toBeVisible();
    await setupStep(page, 'Model structure').click();
    page.once('dialog', (dialog) => dialog.accept());
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await expect(setupStep(page, 'Shared columns')).toHaveAttribute('aria-current', 'step');
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Models' })).toBeHidden();
    await expect(setupStep(page, 'Models')).toBeVisible();

    await setupStep(page, 'Model structure').click();
    page.once('dialog', (dialog) => dialog.accept());
    await page.getByRole('button', { name: 'Single model', exact: true }).click();
    await expect(setupStep(page, 'Variables')).toBeVisible();
});

test('shared assignments use Table and JSON once, while stages keep independent factors', async ({
    page,
}) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    await startPlan(page);

    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    await shared.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const roles = shared.getByLabel('Variables JSON', { exact: true });
    const draft = JSON.parse(await roles.inputValue());
    expect(draft.target).toBe('Claims');
    expect(draft.weight).toBe('Exposure');
    expect(draft.split).toBeNull();
    await shared.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(shared.getByLabel('Role for Claims', { exact: true })).toHaveValue('target');
    await expect(shared.getByLabel('Role for Exposure', { exact: true })).toHaveValue('weight');

    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Save variables for Main effects and open Fit', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Models').click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Model stage').selectOption({ label: 'Model 2 of 2 · Model 2' });
    await expect(page.getByText(/complete score through Main effects as its offset/)).toBeVisible();
    await page.getByRole('tab', { name: 'JSON', exact: true }).click();
    await page.getByLabel('Model variables JSON').fill(JSON.stringify({
        variables: { VehicleAge: { kind: 'step', knots: [2, 5, 10] } },
    }));
    await page.getByRole('button', { name: 'Apply variable JSON', exact: true }).click();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Save variables for Model 2 and open Fit', exact: true }).click();

    const plan = await modelPlan(page);
    expect(plan.plan.stages[0].factors).toEqual(['DriverAge']);
    expect(plan.plan.stages[1].factors).toEqual(['VehicleAge']);
    expect(plan.plan.stages.every((stage) => !('target' in stage) && !('split' in stage))).toBe(
        true,
    );
    expect(errors).toEqual([]);
});

test('shared role drafts remain browser-local, validate atomically, and persist after Apply', async ({
    page,
}) => {
    const requests = [];
    page.on('request', (request) => requests.push(request.url()));
    await page.goto('/');
    await startPlan(page);
    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    const before = requests.length;

    await shared.getByLabel('Name for DriverAge', { exact: true }).fill('Age');
    await shared.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await shared.getByLabel('Role for VehicleAge', { exact: true }).selectOption('unassigned');
    await shared.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const editor = shared.getByLabel('Variables JSON', { exact: true });
    const draft = JSON.parse(await editor.inputValue());
    draft.predictor = draft.predictor.filter((name) => name !== 'Region');
    draft.ignore.push('Region');
    await editor.fill(JSON.stringify(draft, null, 2));
    await shared.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(shared.getByLabel('Role for Region', { exact: true })).toHaveValue('ignore');
    await expect(shared.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    expect(requests).toHaveLength(before);

    await shared.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(shared.locator('.preview-card')).toBeVisible();
    await shared.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(shared.getByRole('status')).toContainText('Settings applied');
    await page.reload();
    await expect(setupStep(page, 'Shared columns')).toHaveAttribute('aria-current', 'step');
    await expect(shared.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');

    await shared.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    await editor.fill('{broken');
    await shared.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await shared.getByRole('button', { name: 'Reset', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await shared.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(shared.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
});

test('shared assignments reject duplicate roles and freeze a draft while its preview is pending', async ({
    page,
}) => {
    await page.goto('/');
    await startPlan(page);
    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    await shared.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('target');
    await expect(shared.getByLabel('Role for Claims', { exact: true })).toHaveValue('unassigned');
    await shared.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const editor = shared.getByLabel('Variables JSON', { exact: true });
    const duplicate = JSON.parse(await editor.inputValue());
    duplicate.ignore.push('AnnualMileage');
    await editor.fill(JSON.stringify(duplicate));
    await shared.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText('appears more than once');
    await shared.getByRole('button', { name: 'Reset', exact: true }).click();
    await shared.getByRole('button', { name: 'Table', exact: true }).click();

    const regionRole = shared.getByLabel('Role for Region', { exact: true });
    await regionRole.selectOption('ignore');
    let release;
    await page.route('**/api/variables/preview', async (route) => {
        await new Promise((resolve) => {
            release = resolve;
        });
        await route.continue();
    });
    await shared.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(regionRole).toBeDisabled();
    await expect(shared.getByRole('button', { name: 'Reset', exact: true })).toBeDisabled();
    await expect.poll(() => Boolean(release)).toBe(true);
    release();
    await expect(regionRole).toBeEnabled();
    await expect(shared.getByRole('button', { name: 'Apply changes', exact: true })).toBeVisible();
});

test('an unapplied shared draft survives navigation away from Project & data', async ({ page }) => {
    await page.goto('/');
    await startPlan(page);
    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    await shared.getByLabel('Name for Claims', { exact: true }).fill('KeptDraft');
    await shared.getByLabel('Name for Claims', { exact: true }).press('Tab');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'One-way effects', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await expect(shared.getByLabel('Name for Claims', { exact: true })).toHaveValue('KeptDraft');
});

test('a fitted plan locks shared assignments until an explicit revision preserves history', async ({
    page,
}) => {
    await page.goto('/');
    await startPlan(page, 'Single model');
    await setupStep(page, 'Variables').click();
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Continue to model details', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to variable treatment', exact: true }).click();
    await page.getByRole('button', { name: 'Save Main effects setup and open Fit', exact: true }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: /^Fit / }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({ timeout: 120000 });

    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Shared columns').click();
    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    await expect(shared.getByText(/^Shared assignments are locked/)).toBeVisible();
    await expect(shared.getByLabel('Role for Claims', { exact: true })).toBeDisabled();
    await expect(shared.getByRole('button', { name: 'Revise shared assignments', exact: true })).toBeVisible();

    page.once('dialog', (dialog) => dialog.accept());
    await shared.getByRole('button', { name: 'Revise shared assignments', exact: true }).click();
    await expect(shared.getByLabel('Role for Claims', { exact: true })).toBeEnabled();
    expect((await modelPlan(page)).history).toHaveLength(1);
    await setupStep(page, 'Variables').click();
    await expect(page.getByText('Saved fitted versions', { exact: false })).toBeVisible();
});
