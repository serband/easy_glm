import { test, expect } from '@playwright/test';

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
    await page.getByRole('button', { name: 'Add pair correction' }).click();
    await page.getByLabel('New pair first predictor').selectOption('B');
    await page.getByLabel('New pair second predictor').selectOption('C');
    await page.getByRole('button', { name: 'Add pair correction' }).click();
    for (const [number, label] of [
        [2, 'A × B'],
        [3, 'B × C'],
    ]) {
        const card = page.getByLabel(`Stage ${number} ${label}`);
        await card.locator('summary').click();
        await card.getByRole('button', { name: 'Remove setting' }).last().click();
        await card.getByLabel(`Iterations for stage ${number} candidate 1`).fill('5');
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
