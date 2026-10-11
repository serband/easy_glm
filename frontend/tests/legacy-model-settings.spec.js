import { test, expect } from '@playwright/test';

async function saveLegacyCatBoost(page) {
    const session = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': session.token };
    const snapshot = await (await page.request.get('/api/variables', { headers })).json();
    const response = await page.request.post('/api/models/save', {
        headers,
        data: {
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            name: 'Legacy automatic CatBoost',
            create: true,
            fields: {
                family: 'poisson', link: 'log', target: 'Claims', weight: 'Exposure',
                components: [{
                    component_id: 'legacy-catboost', name: 'Legacy automatic CatBoost',
                    kind: 'catboost', factors: ['VehicleAge'],
                    settings: {
                        tune: true, trials: 4, time_limit_minutes: 12, cv: 3, seed: 42,
                        thread_count: 1, iterations: 20, depth: 3, learning_rate: 0.05,
                        l2_leaf_reg: 3,
                    },
                }],
            },
        },
    });
    expect(response.ok(), await response.text()).toBe(true);
    const workbench = await (await page.request.get('/api/workbench', { headers })).json();
    expect(workbench.models['Legacy automatic CatBoost'].components[0].settings).toMatchObject({
        tune: true,
        trials: 4,
        time_limit_minutes: 12,
        cv: 3,
        seed: 42,
        thread_count: 1,
    });
}

test('a saved legacy CatBoost recipe retains automatic tuning settings', async ({ page }) => {
    await page.goto('/');
    await saveLegacyCatBoost(page);
    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Legacy automatic CatBoost', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();
    await expect(page.getByText('Existing CatBoost fitting settings. Choose a method only if you want to change this draft.')).toBeVisible();
});
