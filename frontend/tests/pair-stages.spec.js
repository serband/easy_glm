import { test, expect } from '@playwright/test';

test('residual interaction opens an unsaved automatic pair stage for review', async ({ page }) => {
    test.setTimeout(90000);
    const button = (name) => page.getByRole('button', { name, exact: true });
    const name = `Residual pair ${Date.now()}`;
    const includeRequests = [];
    page.on('request', (request) => {
        if (
            request.method() === 'POST' &&
            request.url().includes('/api/review/') &&
            request.postDataJSON()?.action === 'include_pair'
        )
            includeRequests.push(request);
    });
    await page.goto('/');
    await button('Model').click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill(name);
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const project = async () =>
        (
            await page.request.get('/api/project', {
                headers: { 'X-EasyGLM-Token': token },
            })
        ).json();
    const before = await project();
    async function search() {
        await button('Diagnostics').click();
        await page.getByRole('tab', { name: 'Residual factors', exact: true }).click();
        await button('Find missing interactions').click();
        await expect(
            page.getByRole('heading', { name: 'Missing interactions', exact: true }),
        ).toBeVisible();
    }
    await search();
    const candidate = page
        .getByRole('row')
        .filter({ has: button('Add and review model') })
        .first();
    const pairLabel = await candidate.locator('td').first().innerText();
    await candidate.getByRole('button', { name: 'Add and review model', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Model design and fit', exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel(`Stage 2 ${pairLabel}`, { exact: true })).toBeVisible();
    await expect(button('Save model settings')).toBeEnabled();
    expect((await project()).models[name]).toEqual(before.models[name]);
    expect(includeRequests).toHaveLength(0);
    // Reopening the same suggestion must not append another copy or lose the draft.
    await search();
    await candidate.getByRole('button', { name: 'Add and review model', exact: true }).click();
    await expect(page.locator('.pair-stages .stage-card')).toHaveCount(2);
    const save = page.waitForRequest('**/api/models/save');
    await button('Save model settings').click();
    const fields = (await save).postDataJSON().fields;
    expect(fields.pair_stages).toHaveLength(1);
    expect(fields.pair_stages[0].search.method).toBe('optuna');
    expect(fields.pair_stages[0]).not.toHaveProperty('search_limits');
    expect(fields.interactions).toEqual([]);
    await expect(
        page.getByText('Model settings saved. Fit when ready.', { exact: true }),
    ).toBeVisible();
    expect((await project()).models[name].pair_stages).toHaveLength(1);
});

test('sequential pair draft keeps stable IDs, order, and pair-only parents', async ({ page }) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill(`Pair draft ${Date.now()}`);
    await expect(page.getByRole('button', { name: 'Sequential CatBoost pairs' })).toHaveAttribute(
        'aria-pressed',
        'true',
    );
    const stages = page.getByRole('region', { name: 'Sequential pair stages' });
    await expect(stages.getByLabel('Stage 1 Main effects')).toBeVisible();
    await page.getByLabel('Include VehicleAge', { exact: true }).uncheck();
    await page.getByLabel('New pair first predictor').selectOption('DriverAge');
    await page.getByLabel('New pair second predictor').selectOption('VehicleAge');
    await page.getByRole('button', { name: 'Add pair correction' }).click();
    await page.getByLabel('New pair first predictor').selectOption('Region');
    await page.getByLabel('New pair second predictor').selectOption('DriverAge');
    await page.getByRole('button', { name: 'Add pair correction' }).click();
    await expect(stages.getByLabel('Stage 2 DriverAge × VehicleAge')).toContainText(
        'Baseline: Main effects',
    );
    await expect(stages.getByLabel('Stage 3 Region × DriverAge')).toContainText(
        'Baseline: Main effects + DriverAge × VehicleAge',
    );
    await page.getByRole('button', { name: 'Move Region × DriverAge up' }).click();
    await expect(stages.getByLabel('Stage 2 Region × DriverAge')).toBeVisible();
    await page.getByRole('button', { name: 'Create model' }).click();
    await expect(page.getByText('Model settings saved. Fit when ready.')).toBeVisible();
    const fields = saves.at(-1).fields;
    expect(fields.interactions).toEqual([]);
    expect(fields.predictors).not.toContain('VehicleAge');
    expect(fields.pair_stages.map((stage) => [stage.a, stage.b])).toEqual([
        ['Region', 'DriverAge'],
        ['DriverAge', 'VehicleAge'],
    ]);
    expect(new Set(fields.pair_stages.map((stage) => stage.stage_id)).size).toBe(2);
    expect(fields.pair_stages[0].candidates).toEqual([]);
    expect(fields.pair_stages[0]).not.toHaveProperty('search_limits');
    expect(fields.pair_stages[0].search).toEqual({
        method: 'optuna',
        trials: 8,
        prefix_trials: 4,
    });
    expect(fields.pair_method).toBe('sequential_catboost');
});

test('new eligible model keeps the staged editor even before a pair is added', async ({ page }) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    const name = `Empty staged ${Date.now()}`;
    await page.getByLabel('New model name').fill(name);
    await expect(page.getByRole('region', { name: 'Sequential pair stages' })).toBeVisible();
    await page.getByLabel('Model family').selectOption('gamma');
    await expect(page.getByRole('region', { name: 'Defined interactions' })).toBeVisible();
    await page.getByLabel('Model family').selectOption('poisson');
    await expect(page.getByRole('region', { name: 'Sequential pair stages' })).toBeVisible();
    await page.getByRole('button', { name: 'Create model' }).click();
    expect(saves.at(-1).fields.pair_method).toBe('sequential_catboost');
    expect(saves.at(-1).fields.pair_stages).toEqual([]);
    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption(name);
    await expect(page.getByRole('region', { name: 'Sequential pair stages' })).toBeVisible();
});

test('legacy interaction conversion creates a separate staged model', async ({ page }) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    const source = `Legacy pair source ${Date.now()}`;
    await page.getByLabel('New model name').fill(source);
    await page.getByRole('button', { name: 'Legacy GLM interactions' }).click();
    await page.getByLabel('Interaction first factor').selectOption('DriverAge');
    await page.getByLabel('Interaction second factor').selectOption('Region');
    await page.getByRole('button', { name: 'Add interaction' }).click();
    await page.getByRole('button', { name: 'Create model' }).click();
    await page.getByRole('button', { name: 'Copy as sequential pair stages' }).click();
    await expect(page.getByLabel('Model selection')).toHaveValue(`${source} pair stages`);
    expect(saves.at(-1).create).toBe(true);
    expect(saves.at(-1).fields.interactions).toEqual([]);
    expect(saves.at(-1).fields.pair_method).toBe('sequential_catboost');
    expect(saves.at(-1).fields.pair_stages).toHaveLength(1);
    expect(saves.at(-1).fields.pair_stages[0].search.method).toBe('optuna');
    expect(saves.at(-1).fields.pair_stages[0]).not.toHaveProperty('search_limits');
    await page.getByLabel('Model selection').selectOption(source);
    await expect(page.getByRole('region', { name: 'Defined interactions' })).toContainText(
        'DriverAge × Region',
    );
});

test('saved fixed settings remain fixed until explicitly switched to automatic tuning', async ({
    page,
}) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    const name = `Fixed pair ${Date.now()}`;
    await page.getByLabel('New model name').fill(name);
    await page.getByLabel('New pair first predictor').selectOption('DriverAge');
    await page.getByLabel('New pair second predictor').selectOption('Region');
    await page.getByRole('button', { name: 'Add pair correction' }).click();
    await page.getByRole('button', { name: 'Create model' }).click();
    const initial = saves.at(-1);
    const session = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': session.token };
    const snapshot = await (await page.request.get('/api/variables', { headers })).json();
    const fixed = await page.request.post('/api/models/save', {
        headers,
        data: {
            ...initial,
            create: false,
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            fields: {
                ...initial.fields,
                pair_stages: initial.fields.pair_stages.map((stage) => ({
                    ...stage,
                    search: null,
                    candidates: [{ depth: 2, iterations: 60, learning_rate: 0.08, l2_leaf_reg: 3 }],
                })),
            },
        },
    });
    expect(fixed.ok(), await fixed.text()).toBe(true);
    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption(name);
    const card = page.getByLabel('Stage 2 DriverAge × Region');
    await expect(card).toContainText('Saved fixed settings');
    await card.getByText('Stage settings').click();
    await expect(card.getByRole('button', { name: 'Switch to automatic tuning' })).toBeVisible();
    await expect(card.getByLabel('Tuning trials for stage 2')).toHaveCount(0);
    await card.getByRole('button', { name: 'Switch to automatic tuning' }).click();
    await expect(card).toContainText('Automatic tuning · 8 trials');
    await page.getByRole('button', { name: 'Save model settings' }).click();
    expect(saves.at(-1).fields.pair_stages[0].search.method).toBe('optuna');
    expect(saves.at(-1).fields.pair_stages[0].candidates).toEqual([]);
});
