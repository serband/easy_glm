import { test, expect } from '@playwright/test';

const component = (name) => ({
    name,
    kind: 'glm',
    factors: ['Region'],
    settings: { alpha: 0.001 },
});

const models = {
    'Live parent': { components: [component('Available base')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
    'Historical child': { components: [component('Base region'), component('Historic addition')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
    'Missing own result': { components: [component('Lost result')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
    'Usable child': { components: [component('Base region'), component('Still usable addition')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
    'Running child': { components: [component('Base region'), component('Current addition')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
    'Normal draft': { components: [component('Unfitted region')], family: 'poisson', target: 'ClaimCount', weight: 'Exposure' },
};

const modelFlow = {
    nodes: [
        { name: 'Live parent', own_fit_available: true, parent_fit_available: true, append_available: true, revise_available: true },
        { name: 'Historical child', parent_name: 'Unavailable parent', edge_status: 'missing', full_refit_required: true, own_fit_available: false, parent_fit_available: false, append_available: false, revise_available: false },
        { name: 'Missing own result', parent_name: 'Live parent', edge_status: 'live', status: 'missing_fit', missing_own_fit: true, full_refit_required: true, own_fit_available: false, parent_fit_available: true, append_available: false, revise_available: false },
        { name: 'Usable child', parent_name: 'Unavailable parent', edge_status: 'historical', own_fit_available: true, parent_fit_available: false, append_available: true, revise_available: false },
        { name: 'Running child', parent_name: 'Unavailable parent', edge_status: 'live', own_fit_available: false, parent_fit_available: true, append_available: false, revise_available: false },
        { name: 'Normal draft', own_fit_available: false, parent_fit_available: true, append_available: false, revise_available: false },
    ],
    edges: [
        { from: 'Live parent', to: 'Missing own result', edge_status: 'live' },
        { from: 'Unavailable parent', to: 'Historical child', edge_status: 'missing' },
        { from: 'Unavailable parent', to: 'Usable child', edge_status: 'historical' },
        { from: 'Unavailable parent', to: 'Running child', edge_status: 'live' },
    ],
};

const jobs = {
    'Live parent': { status: 'complete', applicable: true, artifact_available: true },
    'Historical child': { status: 'not_fitted', applicable: false },
    'Missing own result': { status: 'complete', applicable: false, artifact_available: false },
    'Usable child': { status: 'complete', applicable: true, artifact_available: true },
    'Running child': { status: 'running', applicable: false, message: 'training component 2' },
    'Normal draft': { status: 'not_fitted', applicable: false },
};

async function checkVerticalLayout(page) {
    const header = page.locator('.selected-model-header');
    const [identity, context, actions] = await Promise.all([
        header.locator('.selected-model-identity').boundingBox(),
        header.locator('.selected-model-context').boundingBox(),
        header.locator('.selected-model-actions').boundingBox(),
    ]);
    expect(context.y).toBeGreaterThan(identity.y);
    expect(actions.y).toBeGreaterThan(context.y);
    expect(await page.locator('.model-builder').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
}

test('recovery cards render truthful actions for lost parents and artifacts', async ({ page }) => {
    await page.route('**/api/workbench', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({ response, json: { ...body, models: { ...body.models, ...models }, jobs, model_flow: modelFlow, stack_sources: {} } });
    });
    // This is an imported legacy-project fixture.  The current application decides
    // which legacy workbench to open from the saved model list, before it requests
    // the detailed workbench snapshot.
    await page.route('**/api/variables', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({ response, json: { ...body, models: Object.keys(models) } });
    });
    await page.route('**/api/jobs', async (route) => route.fulfill({ json: jobs }));
    await page.route('**/api/results/Usable%20child', async (route) => route.fulfill({ json: {
        components: models['Usable child'].components,
        summary: { components: 2, family: 'poisson', train_ae: 1 },
        review_variables: ['Region'],
    } }));

    await page.setViewportSize({ width: 1332, height: 1225 });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.locator('.model-list-item').filter({ has: page.locator('strong', { hasText: 'Historical child' }) })).toBeVisible();

    const choose = async (name) => page.getByRole('complementary', { name: 'Models' }).locator('.model-list-item').filter({ has: page.locator('strong', { hasText: name }) }).click();
    const selectedStatus = () => page.locator('.selected-model-header .model-state');
    await choose('Historical child');
    await expect(selectedStatus()).toHaveText('Needs a new fit');
    await expect(page.getByText('Original starting fit: Unavailable parent (unavailable).', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Fit saved recipe as new model', exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Saved recipe', exact: true })).toBeVisible();
    await checkVerticalLayout(page);

    await choose('Missing own result');
    await expect(selectedStatus()).toHaveText('Cannot score');
    await expect(page.getByText('This saved model cannot be scored because its fitted files are unavailable.', { exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Saved recipe', exact: true })).toBeVisible();
    const missingOwnListItem = page.locator('.model-list-item').filter({ has: page.locator('strong', { hasText: 'Missing own result' }) });
    await expect(missingOwnListItem).toContainText('Built on Live parent');
    await expect(missingOwnListItem).not.toContainText('unavailable');
    const modelMapSummary = page.locator('.model-map > summary');
    await modelMapSummary.click();
    await expect(page.getByText('Builds on Live parent · complete model predictions as offset', { exact: true })).toBeVisible();
    const missingOwnFlowCard = page.locator('article[aria-label="Missing own result, cannot score"]');
    await expect(missingOwnFlowCard.locator('.flow-select > small')).toHaveText('Builds on Live parent · complete model predictions as offset');
    await modelMapSummary.click();
    await checkVerticalLayout(page);

    await choose('Usable child');
    await expect(selectedStatus()).toHaveText('Fitted');
    await expect(page.getByText(/originally built on an earlier Unavailable parent fit/)).toBeVisible();
    await expect(page.getByRole('button', { name: 'Add model after this', exact: true })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Revise model', exact: true })).toBeDisabled();
    await checkVerticalLayout(page);

    await choose('Running child');
    await expect(page.getByText('Fitting: training component 2', { exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Cancel fit', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Edit draft', exact: true })).toHaveCount(0);
    await checkVerticalLayout(page);

    await choose('Normal draft');
    await expect(page.getByRole('heading', { name: 'Edit draft model', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Fit model', exact: true }).first()).toBeVisible();

    await page.setViewportSize({ width: 390, height: 844 });
    await page.getByLabel('Model', { exact: true }).selectOption('Historical child');
    await expect(selectedStatus()).toHaveText('Needs a new fit');
    await checkVerticalLayout(page);
    await page.getByLabel('Model', { exact: true }).selectOption('Missing own result');
    await expect(selectedStatus()).toHaveText('Cannot score');
    await checkVerticalLayout(page);
    await page.getByLabel('Model', { exact: true }).selectOption('Usable child');
    await expect(page.getByRole('button', { name: 'Revise model', exact: true })).toBeDisabled();
    await checkVerticalLayout(page);
    await page.getByLabel('Model', { exact: true }).selectOption('Running child');
    await expect(page.getByRole('status')).toHaveText('Fitting: training component 2');
    await checkVerticalLayout(page);
});
