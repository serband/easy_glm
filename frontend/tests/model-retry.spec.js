import { test, expect } from '@playwright/test';

const name = 'Retry after failure';
const component = {
    component_id: 'retry-root', name: 'Retry root', kind: 'glm',
    factors: ['Region'], settings: { alpha: 0.001 },
};
const recipe = {
    components: [component], family: 'poisson', target: 'Claims', weight: 'Exposure',
};

test('a delayed old failed job cannot unlock a newly accepted retry', async ({ page }) => {
    const oldFailed = {
        [name]: {
            id: 'previous-job', status: 'failed', applicable: false,
            message: 'Previous fit failed.',
        },
    };
    let jobs = oldFailed;
    await page.route('**/api/variables', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({ response, json: { ...body, models: [name] } });
    });
    await page.route('**/api/workbench', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({
            response,
            json: {
                ...body, models: { ...body.models, [name]: recipe },
                jobs, model_flow: {
                    nodes: [{ name, own_fit_available: false, parent_fit_available: true }],
                    edges: [],
                }, stack_sources: {},
            },
        });
    });
    await page.route('**/api/jobs', async (route) => route.fulfill({ json: jobs }));
    await page.route('**/api/models/save', async (route) => {
        const response = await route.fetch();
        const body = await response.json();
        return route.fulfill({ response, json: { ...body, models: [name] } });
    });
    await page.route(`**/api/models/${encodeURIComponent(name)}/fit`, async (route) => {
        jobs = { [name]: { id: 'new-job', status: 'running', applicable: false, message: 'Training' } };
        await route.fulfill({ status: 202, json: { id: 'new-job' } });
    });
    await page.route('**/api/results/Retry%20after%20failure?stack_view=true', async (route) => route.fulfill({
        json: { components: [component], summary: { components: 1, family: 'poisson', train_ae: 1 }, review_variables: ['Region'] },
    }));

    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Draft');

    let oldSnapshotRead = false;
    await page.unroute('**/api/jobs');
    await page.route('**/api/jobs', async (route) => {
        oldSnapshotRead = true;
        await route.fulfill({ json: oldFailed });
    });
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect.poll(() => oldSnapshotRead).toBe(true);
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitting');
    await expect(page.locator('.fit-progress')).toContainText(/Starting fit|Fitting:/);
    await expect(page.getByRole('heading', { name: 'Edit draft model', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Edit draft', exact: true })).toHaveCount(0);

    // The worker's eventual completion is covered by the real saved-model suite.
    // This regression is specifically about the old terminal reply: it must leave
    // the accepted retry locked while the new job is still the authoritative one.
});
