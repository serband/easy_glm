import { test, expect } from '@playwright/test';

test('pair selector groups unassigned source variables without promoting them', async ({
    page,
}) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save'))
            saves.push(request.postDataJSON());
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill(`Unassigned pair ${Date.now()}`);
    const first = page.getByLabel('New pair first predictor');
    await expect(first.locator('optgroup[label="Predictors"]')).toBeAttached();
    await expect(first.locator('optgroup[label="Unassigned"] option[value="D"]')).toBeAttached();
    await first.selectOption('A');
    await page.getByLabel('New pair second predictor').selectOption('D');
    await page.getByRole('button', { name: 'Add interaction' }).click();
    await page.getByRole('button', { name: 'Create model' }).click();
    const fields = saves.at(-1).fields;
    expect(fields.predictors).not.toContain('D');
    expect(fields.pair_stages[0]).toMatchObject({ a: 'A', b: 'D' });
});

test('per-interaction time limit validates, persists, times out and retries without staling tables', async ({
    page,
}) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save'))
            saves.push(request.postDataJSON());
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    const modelName = `Pair timeout ${Date.now()}`;
    await page.getByLabel('New model name').fill(modelName);

    const limit = page.getByLabel('Time limit per interaction (minutes)');
    await expect(limit).toHaveValue('15');
    await expect(
        page.getByText(
            'Each interaction gets its own allowance for validation, tuning and its final table fit. The initial GLM is excluded. Checked between fitting steps.',
            { exact: true },
        ),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Legacy GLM interactions' }).click();
    await expect(limit).toBeHidden();
    await page.getByRole('button', { name: 'Sequential CatBoost pairs' }).click();
    await expect(limit).toHaveValue('15');

    await page.getByLabel('New pair first predictor').selectOption('A');
    await page.getByLabel('New pair second predictor').selectOption('B');
    await page.getByRole('button', { name: 'Add interaction' }).click();
    const stage = page.getByLabel('Stage 2 A × B');
    await stage.getByText('Stage settings').click();
    await stage.getByText('Tuning budget').click();
    await stage.getByLabel('Tuning trials for stage 2', { exact: true }).fill('1');
    await stage.getByLabel('Prefix tuning trials for stage 2', { exact: true }).fill('1');

    await limit.fill('0');
    await page.getByRole('button', { name: 'Create model' }).click();
    await expect(page.getByRole('alert')).toContainText(
        'pair_time_limit_minutes must be a positive finite number',
    );

    await limit.fill('0.000001');
    await page.getByRole('button', { name: 'Create model' }).click();
    await expect(page.getByText('Model settings saved. Fit when ready.')).toBeVisible();
    expect(saves.at(-1).fields.pair_time_limit_minutes).toBe(0.000001);

    const timeoutStarted = Date.now();
    await page.getByRole('button', { name: 'Fit all stages' }).click();
    await expect(page.getByText('Fit failed', { exact: true })).toBeVisible({ timeout: 30000 });
    await expect(
        page.getByText(
            'Interaction A × B reached its 1e-06-minute time limit. Increase Time limit per interaction in Model > Fit settings, save, and retry.',
            { exact: true },
        ),
    ).toBeVisible();
    expect(Date.now() - timeoutStarted).toBeLessThan(30000);

    await limit.fill('2.75');
    await page.getByRole('button', { name: 'Save model settings' }).click();
    await expect(page.getByText('Model settings saved. Fit when ready.')).toBeVisible();
    expect(saves.at(-1).fields.pair_time_limit_minutes).toBe(2.75);
    await page.getByRole('button', { name: 'Fit all stages' }).click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 180000 });
    await expect(stage).toContainText(/Up to date|No improvement/);

    await limit.fill('3.5');
    await page.getByRole('button', { name: 'Save model settings' }).click();
    await expect(page.getByText('Model settings saved. Fit when ready.')).toBeVisible();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible();
    await expect(stage).toContainText(/Up to date|No improvement/);

    await page.reload();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption(modelName);
    await expect(page.getByLabel('Time limit per interaction (minutes)')).toHaveValue('3.5');
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible();
    await expect(page.getByLabel('Stage 2 A × B')).toContainText(/Up to date|No improvement/);
});

test('fitted stages expose pair edits, suffix status, refit impact and narrow layout', async ({
    page,
}) => {
    const edits = [];
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
        if (request.method() !== 'POST' || !request.url().includes('/api/review/')) return;
        const payload = request.postDataJSON();
        if (payload.action === 'edit') edits.push(payload);
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill('Pair acceptance');
    for (const name of ['A', 'B', 'C']) {
        await page.getByLabel(`Include ${name}`, { exact: true }).uncheck();
    }
    await expect(page.getByRole('button', { name: 'Sequential CatBoost pairs' })).toHaveAttribute(
        'aria-pressed',
        'true',
    );
    await page.getByLabel('New pair first predictor').selectOption('A');
    await page.getByLabel('New pair second predictor').selectOption('B');
    await page.getByRole('button', { name: 'Add interaction' }).click();
    await page.getByLabel('New pair first predictor').selectOption('B');
    await page.getByLabel('New pair second predictor').selectOption('C');
    await page.getByRole('button', { name: 'Add interaction' }).click();
    for (const [number, label] of [
        [2, 'A × B'],
        [3, 'B × C'],
    ]) {
        const card = page.getByLabel(`Stage ${number} ${label}`);
        await card.getByText('Stage settings').click();
        await card.getByText('Tuning budget').click();
        await card.getByLabel(`Tuning trials for stage ${number}`, { exact: true }).fill('1');
        await card
            .getByLabel(`Prefix tuning trials for stage ${number}`, { exact: true })
            .fill('1');
    }
    await page.getByRole('button', { name: 'Create model' }).click();
    await page.getByRole('button', { name: 'Fit all stages' }).click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 180000 });
    await expect(page.getByLabel('Stage 2 A × B')).toContainText(/Up to date|No improvement/);
    await expect(page.getByLabel('Stage 3 B × C')).toContainText(/Up to date|No improvement/);

    await page.getByRole('button', { name: 'Rate tables', exact: true }).click();
    const firstStage = saves[0].fields.pair_stages[0].stage_id;
    await page.getByLabel('Rate table variable').selectOption(firstStage);
    await expect(page.getByRole('region', { name: 'Relativity chart' })).toContainText('A × B');
    await page.getByLabel('Pair chart values').selectOption('fitting_weight');
    await page.getByLabel('Pair chart values').selectOption('relativity');
    const firstRate = page.getByLabel('Pair relativity row 1', { exact: true });
    await expect(firstRate).toBeVisible();
    const initial = Number(await firstRate.inputValue());
    await firstRate.fill(String(initial * 1.2));
    await expect(page.getByRole('button', { name: 'Apply adjustment' })).toBeEnabled({
        timeout: 30000,
    });
    expect(edits.at(-1).stage_id).toBe(firstStage);
    await page.getByRole('button', { name: 'Apply adjustment' }).click();
    await expect(page.getByText('Adjustments applied.')).toBeVisible({ timeout: 30000 });

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(page.getByLabel('Stage 3 B × C')).toContainText('Needs refitting');
    await page.getByRole('button', { name: 'Move B × C up' }).click();
    await page.getByRole('button', { name: 'Save model settings' }).click();
    await expect(
        page.getByText(
            '1 manual cell edit in those stages will be replaced after the fit succeeds:',
        ),
    ).toBeVisible();
    await page.setViewportSize({ width: 390, height: 780 });
    await expect(page.getByLabel('Stage 2 B × C')).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
});
