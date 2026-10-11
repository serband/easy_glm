import { expect } from '@playwright/test';

async function authenticatedSnapshot(page) {
    const session = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': session.token };
    const snapshot = await (await page.request.get('/api/variables', { headers })).json();
    return { headers, snapshot };
}

export async function saveLegacyGlm(page, name, predictors = ['DriverAge']) {
    const { headers, snapshot } = await authenticatedSnapshot(page);
    const response = await page.request.post('/api/models/save', {
        headers,
        data: {
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            name,
            create: true,
            fields: {
                family: 'poisson', link: 'log', target: 'Claims', weight: 'Exposure',
                offset: null, divide_target_by_weight: false, predictors,
                interactions: [], pair_stages: [], pair_method: 'legacy_glm',
                pair_time_limit_minutes: 15, tweedie_power: 1.5, base: 'modal',
                penalty: { alpha: 0.001, cv: null, n_alphas: 20, l1_ratio: 1 },
            },
            designs: {}, n_bins: 8, min_level_share: 0.0025,
        },
    });
    expect(response.ok(), await response.text()).toBe(true);
    return response.json();
}

/**
 * Open the still-supported editor for an imported single-model project.
 * New projects deliberately begin on Project & data, so legacy regression
 * tests first create an ordinary persisted legacy model and then explicitly
 * select a blank draft.  This is the same path an imported pre-plan project
 * takes in the workbench; it is not a test-only UI route.
 */
export async function openNewLegacyDraft(page, predictors = ['DriverAge']) {
    await saveLegacyGlm(page, `Imported legacy ${Date.now()}`, predictors);
    // The save is made through the authenticated API so reload the client-side
    // snapshot before asking App to choose its compatibility workspace.
    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection', { exact: true }).selectOption('__new__');
}

export async function fitLegacyModel(page, name) {
    const { headers, snapshot } = await authenticatedSnapshot(page);
    const started = await page.request.post(`/api/models/${encodeURIComponent(name)}/fit`, {
        headers,
        data: { session_id: snapshot.session_id, revision: snapshot.revision },
    });
    expect(started.status(), await started.text()).toBe(202);
    await expect.poll(async () => {
        const jobs = await (await page.request.get('/api/jobs', { headers })).json();
        return jobs[name]?.status;
    }, { timeout: 60000 }).toBe('complete');
}

export async function saveAndFitLegacyGlm(page, name, predictors = ['DriverAge']) {
    await saveLegacyGlm(page, name, predictors);
    await fitLegacyModel(page, name);
}

export async function openSavedLegacyModel(page, name) {
    await page.goto('/?view=model');
    const selection = page.getByLabel('Model selection');
    await expect(selection).toBeVisible();
    await selection.selectOption(name);
}
