import { test, expect } from '@playwright/test';

async function openBuilder(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
}

test('new CatBoost defaults to Optuna tuning and keeps its selected budget in a draft', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Tuned vehicle model');
    await page.getByLabel('Model component type').selectOption('catboost');
    await page.getByLabel('Use VehicleAge').check();

    await expect(page.getByLabel('CatBoost fitting method')).toHaveValue('tune');
    await expect(page.getByLabel('CatBoost tuning budget')).toHaveValue('trials');
    await expect(page.getByLabel('CatBoost trial count')).toHaveValue('8');
    await expect(page.getByText('CatBoost is fitted using training rows only. Holdout rows are not used for tuning.')).toBeVisible();

    await page.getByLabel('CatBoost trial count').fill('0');
    await expect(page.getByLabel('CatBoost trial count')).toHaveValue('0');
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText('CatBoost trials must be a positive integer');
    await expect(page.getByLabel('CatBoost trial count')).toHaveValue('0');
    await page.getByLabel('CatBoost trial count').fill('8');

    await page.getByLabel('CatBoost tuning budget').selectOption('time');
    await expect(page.getByLabel('CatBoost tuning time')).toHaveValue('15');
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    await expect(page.getByText('Draft saved.', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();
    await expect(page.getByLabel('CatBoost fitting method')).toHaveValue('tune');
    await expect(page.getByLabel('CatBoost tuning budget')).toHaveValue('time');
    await expect(page.getByLabel('CatBoost tuning time')).toHaveValue('15');

    await page.getByLabel('CatBoost fitting method').selectOption('manual');
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();
    await expect(page.getByLabel('CatBoost fitting method')).toHaveValue('manual');
    await page.getByLabel('CatBoost fitting method').selectOption('tune');
    await expect(page.getByLabel('CatBoost tuning budget')).toHaveValue('trials');
    await expect(page.getByLabel('CatBoost trial count')).toHaveValue('8');

    await page.setViewportSize({ width: 797, height: 792 });
    expect(await page.locator('.model-builder').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
});

test('a legacy automatic CatBoost draft keeps its saved settings until changed', async ({ page }) => {
    await openBuilder(page);
    const saved = page.waitForRequest('**/api/models/save');
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Legacy tuned draft');
    await page.getByLabel('Model component type').selectOption('catboost');
    await page.getByLabel('Use VehicleAge').check();
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    const original = (await saved).postDataJSON();
    const session = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': session.token };
    const snapshot = await (await page.request.get('/api/variables', { headers })).json();
    const legacySettings = {
        ...original.fields.components.at(-1).settings,
        tune: true,
        trials: 4,
        time_limit_minutes: 12,
    };
    delete legacySettings.tuning_budget;
    const update = await page.request.post('/api/models/save', {
        headers,
        data: {
            ...original,
            create: false,
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            fields: {
                ...original.fields,
                components: [{
                    ...original.fields.components.at(-1),
                    settings: legacySettings,
                }],
            },
        },
    });
    expect(update.ok(), await update.text()).toBe(true);
    const persisted = await (await page.request.get('/api/workbench', { headers })).json();
    expect(persisted.models['Legacy tuned draft'].components.at(-1).settings).toEqual(legacySettings);

    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
    await page.locator('.model-list-item').filter({ hasText: 'Legacy tuned draft' }).click();
    const editDraft = page.getByRole('button', { name: 'Edit draft', exact: true });
    if (await editDraft.count()) await editDraft.click();
    await expect(page.getByText('Existing CatBoost fitting settings. Choose a method only if you want to change this draft.')).toBeVisible();
    const preserved = page.waitForRequest('**/api/models/save');
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    expect((await preserved).postDataJSON().fields.components.at(-1).settings).toEqual(legacySettings);
});

test('CatBoost fits one Optuna trial, then a manual stage without tuning', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('One trial CatBoost');
    await page.getByLabel('Model component type').selectOption('catboost');
    await page.getByLabel('Use VehicleAge').check();
    await page.getByLabel('CatBoost trial count').fill('1');
    await page.getByText('Advanced tuning settings', { exact: true }).click();
    await page.getByLabel('CatBoost CV folds').fill('2');
    await page.getByLabel('CatBoost threads').fill('1');
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 120000 });
    await expect(page.getByText(/1 fitted model · poisson/)).toBeVisible();
    await expect(page.locator('.selected-model-header')).toContainText('Its score includes One trial CatBoost.');
    await expect(page.locator('.selected-model-header')).not.toContainText('Starting level');
    await expect(page.getByRole('heading', { name: /One trial CatBoost — (Optuna tuning|No change selected)/ })).toBeVisible();
    await expect(page.getByText(/1 completed of 1 trial/)).toBeVisible();
    await expect(page.getByText(/2 training folds/)).toBeVisible();
    const noChange = page.getByRole('heading', { name: 'One trial CatBoost — No change selected', exact: true });
    if (await noChange.count()) {
        await expect(page.getByText('No sampled CatBoost setting improved the preceding model on training folds')).toBeVisible();
    }

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await page.getByLabel('Stacked model name').fill('Manual correction');
    await page.getByLabel('Model component type').selectOption('catboost');
    await page.getByLabel('Use DriverAge').check();
    await page.getByLabel('CatBoost fitting method').selectOption('manual');
    await page.getByLabel('CatBoost iterations').fill('20');
    await page.getByLabel('CatBoost depth').fill('4');
    await page.getByLabel('CatBoost learning rate').fill('0.1');
    await page.getByLabel('CatBoost threads').fill('1');
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 120000 });
    await expect(page.getByRole('heading', { name: 'Manual correction — Manual CatBoost settings', exact: true })).toBeVisible();
    await expect(page.getByText('Automatic tuning is off; fitted the configured CatBoost settings once')).toBeVisible();
});

test('each model keeps its own variables and local definitions', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Age base');
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Use custom cuts', exact: true }).click();
    const cuts = page.getByLabel('Custom cuts for DriverAge');
    await cuts.fill('25, 35, 45, 55');
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Age base', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    const inherited = page.getByRole('region', { name: 'Inherited model definition' });
    await expect(inherited).toContainText('OutcomeClaims, poisson with log link');
    await expect(inherited).toContainText('ExposureExposure');
    await expect(inherited).toContainText('70% training (seed 42)');
    await expect(inherited).toContainText('Complete score from Age base');

    await page.getByLabel('Stacked model name').fill('Vehicle correction');
    await page.getByRole('tab', { name: 'JSON' }).click();
    const json = page.getByLabel('Model variables JSON');
    const protectedTarget = 'Claims';
    await json.fill(JSON.stringify({ variables: { [protectedTarget]: { kind: 'step' } } }, null, 2));
    await expect(page.getByRole('alert')).toContainText(`${protectedTarget} is not available`);
    await expect(page.getByRole('button', { name: 'Fit model', exact: true })).toBeDisabled();
    await json.fill(JSON.stringify({
        defaults: { n_bins: 6 },
        variables: { VehicleAge: { kind: 'step', knots: [1, 3, 6, 10] } },
    }, null, 2));
    await page.getByRole('button', { name: 'Apply variable JSON', exact: true }).click();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
    await page.getByRole('tab', { name: 'Table' }).click();
    await expect(page.getByLabel('Use VehicleAge')).toBeChecked();
    await expect(page.getByLabel('Use DriverAge')).not.toBeChecked();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Vehicle correction', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
});

test('a catalogue stage uses uploaded attributes and keeps its join key out of the factors', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Catalogue base');
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Catalogue base', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await page.getByLabel('Stacked model name').fill('Vehicle file');
    await page.getByLabel('Model component type').selectOption('catalogue');
    const csv = ['PolicyID,VehicleGroup,Power']
        .concat(Array.from({ length: 12000 }, (_, index) => `${index},${index % 3 ? 'Family' : 'Sport'},${70 + index % 90}`))
        .join('\n');
    await page.getByLabel('Upload catalogue').setInputFiles({
        name: 'vehicle-file.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from(csv),
    });
    await page.getByLabel('Catalogue key').selectOption('PolicyID');
    await expect(page.getByLabel('Use PolicyID')).toHaveCount(0);
    await page.getByLabel('Use VehicleGroup').check();
    await page.getByLabel('CatBoost fitting method').selectOption('manual');
    await page.getByLabel('CatBoost iterations').fill('20');
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Vehicle file', exact: true })).toBeVisible({ timeout: 60000 });
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
    await page.getByRole('button', { name: 'Inspect and amend tables', exact: true }).click();
    await page.getByLabel('Inspect model component').selectOption({ label: 'Vehicle file' });
    await expect(page.getByLabel('Inspect model component')).toHaveValue(/.+/);
});
