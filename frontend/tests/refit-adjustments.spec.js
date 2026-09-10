import { test, expect } from '@playwright/test';

test('a fresh fit starts unadjusted and subsequent Apply uses the new revision', async ({
    page,
}) => {
    const button = (name) => page.getByRole('button', { name, exact: true });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((node) => node.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (path) =>
        (await page.request.get('/api/' + path, { headers: { 'x-easyglm-token': token } })).json();
    const firstFit = (await get('jobs')).Frequency.id;
    await button('Rate tables').click();
    const method = page.getByLabel('Adjustment method', { exact: true });
    await expect(method).toHaveValue('');
    await method.selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('1.71');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await button('Preview rebalance base rate').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await page.locator('.table-snapshots > summary').click();
    await page.getByLabel('Snapshot name').fill('Before refit');
    await button('Save snapshot').click();
    await expect(page.getByText('Snapshot saved in the project.', { exact: true })).toBeVisible();
    const before = await get('project');
    expect(before.models.Frequency.adjustments.length).toBeGreaterThan(0);
    expect(before.models.Frequency.base_rate_override).not.toBeNull();
    await button('Model').click();
    await expect(
        page.getByText('A new fit starts with unadjusted rates.', { exact: true }),
    ).toBeVisible();
    await button('Fit model').click();
    await expect
        .poll(
            async () => {
                const job = (await get('jobs')).Frequency;
                return job.id !== firstFit && job.applicable;
            },
            { timeout: 45000 },
        )
        .toBe(true);
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible();
    const after = await get('project');
    expect(after.models.Frequency.adjustments).toEqual([]);
    expect(after.models.Frequency.base_rate_override).toBeNull();
    expect(after.models.Frequency.snapshots).toEqual(before.models.Frequency.snapshots);
    await button('Rate tables').click();
    await expect(method).toHaveValue('');
    await expect(page.locator('.rate-preview-state')).toHaveText('Original fit');
    const legend = page.locator('.rate-chart-card .chart-legend');
    await expect(legend).toHaveText('● Original fit');
    const table = await get('results/Frequency/table?variable=DriverAge');
    expect(table.rows.every((row) => row.relativity === row.fitted)).toBe(true);
    const info = await get('review-info/Frequency');
    expect(info.undo).toBe(false);
    expect(info.redo).toBe(false);
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'Region · relativity', exact: true }),
    ).toBeVisible();
    await expect(legend).toHaveText('● Original fit');
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('DriverAge');
    await expect(
        page.getByRole('heading', { name: 'DriverAge · relativity', exact: true }),
    ).toBeVisible();
    // The successful refit changed revision. A new explicit edit must work immediately.
    await method.selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('1.83');
    await expect(button('Apply adjustment')).toBeEnabled();
    await expect(legend).toContainText('Original fit');
    await expect(legend).toContainText('Adjusted');
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect((await get('project')).models.Frequency.adjustments.length).toBeGreaterThan(0);
    expect(errors).toEqual([]);
});
