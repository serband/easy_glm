import { test, expect } from '@playwright/test';

test('top N and manual pruning create a separate fitted challenger for holdout comparison', async ({
    page,
}, testInfo) => {
    const button = (name) => page.getByRole('button', { name, exact: true });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/');
    await button('Model').click();
    await page.getByLabel('Penalty mode').selectOption('fixed');
    await page.getByLabel('Fixed alpha').fill('0.01');
    await button('Create model').click();
    await page.getByLabel('Interaction first factor', { exact: true }).selectOption('DriverAge');
    await page.getByLabel('Interaction second factor', { exact: true }).selectOption('Region');
    await button('Add interaction').click();
    await button('Save model settings').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'X-EasyGLM-Token': token } })).json();
    const original = await get('project'),
        originalJobs = await get('jobs');
    await button('Diagnostics').click();
    await page.getByRole('tab', { name: 'Variable importance', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Build a smaller challenger', exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await page.getByLabel('Top N predictors').fill('2');
    await button('Keep top N').click();
    await expect(page.getByText('2 of 4 predictors kept', { exact: true })).toBeVisible();
    await button('Select all').click();
    await page.getByRole('checkbox', { name: 'Keep predictor Region', exact: true }).uncheck();
    await page.getByRole('checkbox', { name: 'Keep predictor VehicleAge', exact: true }).uncheck();
    await expect(
        page.getByText('1 interaction tables removed: DriverAge × Region.', { exact: true }),
    ).toBeVisible();
    await page.getByLabel('Reduced challenger name').fill('Lean frequency');
    await page.screenshot({ path: testInfo.outputPath('easyglm-reduced-selection.png'), fullPage: true });
    await button('Create and fit challenger').click();
    await expect(page.getByLabel('Model selection')).toHaveValue('Lean frequency');
    await expect(button('Compare with original')).toBeVisible({ timeout: 45000 });
    const saved = await get('project'),
        jobs = await get('jobs');
    expect(saved.models.Frequency).toEqual(original.models.Frequency);
    expect(saved.models['Lean frequency'].predictors).toEqual(['DriverAge', 'AnnualMileage']);
    expect(saved.models['Lean frequency'].interactions).toEqual([]);
    expect(jobs.Frequency.id).toBe(originalJobs.Frequency.id);
    expect(jobs.Frequency.applicable).toBe(true);
    await button('Compare with original').click();
    await expect(
        page.getByRole('heading', { name: 'Metrics side by side', exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await expect(page.getByLabel('Comparison subset')).toHaveValue('holdout');
    await expect(page.getByText('Total rate tables', { exact: true })).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('easyglm-reduced-comparison.png'), fullPage: true });
    await expect(page.getByRole('tab', { name: 'Double lift', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Make selected model champion' })).toHaveCount(0);
    await page.getByLabel('Model selection').selectOption('Lean frequency');
    await expect.poll(async () => (await get('project')).champion).toBe('Lean frequency');
    await button('Diagnostics').click();
    await expect(page.getByLabel('Compare with challenger', { exact: true })).toHaveCount(0);
    await expect(page.getByLabel('Default comparison model', { exact: true })).toHaveCount(0);
    await expect(page.getByRole('tab', { name: 'Double lift', exact: true })).toHaveCount(0);
    await expect(
        page.getByRole('tab', { name: 'Relativities that differ', exact: true }),
    ).toHaveCount(0);
    await button('Rate tables').click();
    await expect(page.getByLabel('Adjustment method')).toBeVisible();
    await expect(page.getByLabel('Compare with challenger', { exact: true })).toHaveCount(0);
    await expect(page.getByText('Fit complete', { exact: true })).toHaveCount(0);
    expect(errors).toEqual([]);
});
