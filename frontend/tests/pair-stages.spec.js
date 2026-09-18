import { test, expect } from '@playwright/test';

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
    expect(fields.pair_stages[0].candidates).toHaveLength(2);
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
    await page.getByLabel('Model selection').selectOption(source);
    await expect(page.getByRole('region', { name: 'Defined interactions' })).toContainText(
        'DriverAge × Region',
    );
});
