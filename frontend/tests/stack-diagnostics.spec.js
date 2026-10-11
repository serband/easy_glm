import { test, expect } from '@playwright/test';

const components = [
    { component_id: 'base', name: 'Diagnostic base', kind: 'glm', factors: ['DriverAge'], settings: {} },
    { component_id: 'vehicle', name: 'Diagnostic vehicle', kind: 'catboost', factors: ['VehicleAge'], settings: {} },
    { component_id: 'region', name: 'Diagnostic region', kind: 'glm', factors: ['Region'], settings: {} },
];
const models = Object.fromEntries(components.map((component, index) => [
    component.name,
    {
        components: components.slice(0, index + 1),
        family: 'poisson', target: 'Claims', weight: 'Exposure',
    },
]));
const jobs = Object.fromEntries(Object.keys(models).map((name) => [
    name, { status: 'complete', applicable: true, artifact_available: true },
]));
const scopes = components.map((component, index) => ({
    id: component.component_id,
    label: index === components.length - 1
        ? 'Final fit — after Diagnostic region'
        : `After ${component.name}`,
    included_models: components.slice(0, index + 1).map((item) => item.name),
    component,
}));

function resultFor(name) {
    const last = models[name].components.at(-1);
    return {
        stack: true,
        components: models[name].components,
        summary: { components: models[name].components.length, family: 'poisson', train_ae: 1 },
        review_variables: ['DriverAge', 'VehicleAge', 'Region'],
        diagnostic_scopes: scopes.filter((scope) =>
            models[name].components.some((component) => component.component_id === scope.id),
        ),
        table_index: [],
        tables: {},
    };
}

function diagnosticsFor(scope) {
    return {
        metrics: { ae: 1, rows: 100, exposure: 80, actual: 10, expected: 10 },
        ae: [{ label: 'all', actual: 10, expected: 10, exposure: 80, ae: 1 }],
        lift: [],
        scope,
    };
}

test('diagnostics identify the saved stack, individual checkpoints, and ignore stale responses', async ({ page }) => {
    let delayBase = null;
    await page.route('**/api/variables', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({ response, json: { ...body, models: Object.keys(models) } });
    });
    await page.route('**/api/workbench', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({
            response,
            json: {
                ...body, models: { ...body.models, ...models }, jobs,
                model_flow: {
                    nodes: Object.keys(models).map((name) => ({ name, own_fit_available: true, append_available: true })),
                    edges: [
                        { from: 'Diagnostic base', to: 'Diagnostic vehicle', edge_status: 'live' },
                        { from: 'Diagnostic vehicle', to: 'Diagnostic region', edge_status: 'live' },
                    ],
                }, stack_sources: {},
            },
        });
    });
    await page.route('**/api/jobs', async (route) => route.fulfill({ json: jobs }));
    await page.route('**/api/results/**', async (route) => {
        const name = decodeURIComponent(route.request().url().split('/api/results/')[1].split('?')[0]);
        await route.fulfill({ json: resultFor(name) });
    });
    await page.route('**/api/stack/Diagnostic%20region/diagnostics*', async (route) => {
        const query = new URL(route.request().url()).searchParams;
        const scope = query.get('through') || 'region';
        if (scope === 'base' && delayBase) await delayBase;
        await route.fulfill({ json: diagnosticsFor(scope) });
    });
    await page.route('**/api/stack/Diagnostic%20region/component/*/table', async (route) => {
        await route.fulfill({ json: { tables: [], editable: false } });
    });

    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('complementary', { name: 'Models' })
        .getByRole('button', { name: /Diagnostic region/ }).click();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Diagnostics — Diagnostic region', exact: true })).toBeVisible();

    const saved = page.getByLabel('Diagnostic saved model');
    const scope = page.getByLabel('Diagnostic scope');
    await expect(saved).toHaveValue('Diagnostic region');
    await expect(scope).toContainText('Final fit — after Diagnostic region');
    await expect(scope).toContainText('After Diagnostic base');
    await scope.selectOption('vehicle');
    await expect(page.locator('.diagnostic-scope')).toContainText('After Diagnostic vehicle');
    await expect(page.locator('.diagnostic-scope')).toContainText('Diagnostic base → Diagnostic vehicle');
    await expect(page.locator('.diagnostic-scope')).toContainText('Earlier models are included as the offset');

    await page.getByLabel('A/E factor', { exact: true }).selectOption('DriverAge');
    await expect(page.locator('table caption').first()).toContainText('Diagnostic region: After Diagnostic vehicle');
    await page.getByLabel('Diagnostic rows').selectOption('holdout');
    await expect(page.getByLabel('Diagnostic rows')).toHaveValue('holdout');

    // A late response from an earlier checkpoint must not overwrite the final view.
    let releaseBase;
    delayBase = new Promise((resolve) => { releaseBase = resolve; });
    await scope.selectOption('base');
    await expect(page.getByText('Loading diagnostics…')).toBeVisible();
    await scope.selectOption('region');
    releaseBase();
    await expect(page.locator('.diagnostic-scope')).toContainText('Final fit — after Diagnostic region');
    await expect(page.getByText('Loading diagnostics…')).toHaveCount(0, { timeout: 60000 });

    // Leaving diagnostics while that request is outstanding must not replace tables.
    let releaseLeaving;
    delayBase = new Promise((resolve) => { releaseLeaving = resolve; });
    await scope.selectOption('base');
    await expect(page.getByText('Loading diagnostics…')).toBeVisible();
    await page.getByRole('button', { name: 'Rate tables', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Component table or lookup', exact: true })).toBeVisible();
    releaseLeaving();
    await page.waitForTimeout(100);
    await expect(page.getByRole('heading', { name: 'Component table or lookup', exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await saved.selectOption('Diagnostic base');
    await expect(page.getByRole('heading', { name: 'Diagnostics — Diagnostic base', exact: true })).toBeVisible();
    await expect(scope).toContainText('After Diagnostic base');

    await page.setViewportSize({ width: 797, height: 792 });
    expect(await page.locator('.stack-workbench').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
});
