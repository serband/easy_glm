import { test, expect } from '@playwright/test';

async function project(page) {
    return page.evaluate(async () => {
        const { token } = await (await fetch('/api/session')).json();
        const response = await fetch('/api/project', {
            headers: { 'X-EasyGLM-Token': token },
        });
        return response.json();
    });
}

test('one-way screen uses training rows and stages reviewed role changes', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    const panel = page.getByRole('group', { name: 'One-way feature selection' });
    await panel.locator(':scope > summary').click();
    await expect(panel).toContainText('4 candidates');
    await expect(page.getByLabel('Divide target by weight for feature selection')).toBeChecked();
    await expect(panel).toContainText('4 candidate CV fits');
    await panel.getByText('Advanced settings', { exact: true }).click();
    await expect(page.getByLabel('Feature selection repeats')).toHaveValue('5');
    const before = await project(page);
    const request = page.waitForRequest('**/api/variables/feature-selection');
    await page.getByRole('button', { name: 'Run one-way feature selection' }).click();
    const submitted = (await request).postDataJSON();
    expect(submitted.options).toMatchObject({
        family: 'poisson',
        link: null,
        divide_target_by_weight: true,
        l1_ratio: 1,
        n_alphas: 20,
        repeats: 5,
        seed: 42,
        include_unassigned: true,
    });
    expect(submitted.setup.binning.default_bins).toBe(6);
    expect(submitted.setup.binning.overrides.Risk).toEqual({ method: 'cuts', cuts: [0.5] });
    await expect(panel.getByText(/tested of 4 candidates/)).toBeVisible({ timeout: 150000 });
    const resultRows = panel.locator('.selection-table-wrap tbody tr:not(.selection-detail)');
    await expect(resultRows).toHaveCount(4);
    await expect(panel.getByLabel('Details for Risk')).toBeVisible();
    await panel.getByLabel('Details for Risk').click();
    await expect(panel).toContainText('Shadow 4:');
    await expect(panel).toContainText('Random:');
    await page.setViewportSize({ width: 799, height: 820 });
    await panel.scrollIntoViewIfNeeded();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    const stickyColumns = await panel.locator('.selection-table-wrap').evaluate((wrapper) => {
        const select = wrapper.querySelector('tbody tr:not(.selection-detail) td:first-child');
        const variable = wrapper.querySelector('tbody tr:not(.selection-detail) td:nth-child(2)');
        const start = variable.getBoundingClientRect().left;
        wrapper.scrollLeft = wrapper.scrollWidth;
        return {
            scrollable: wrapper.scrollWidth > wrapper.clientWidth,
            selectVisible:
                select.getBoundingClientRect().left >= wrapper.getBoundingClientRect().left,
            variableVisible:
                variable.getBoundingClientRect().left >= wrapper.getBoundingClientRect().left &&
                Math.abs(variable.getBoundingClientRect().left - start) < 2,
        };
    });
    expect(stickyColumns).toEqual({
        scrollable: true,
        selectVisible: true,
        variableVisible: true,
    });
    await page.screenshot({
        path: testInfo.outputPath('feature-selection-results.png'),
        fullPage: true,
    });
    await page.setViewportSize({ width: 1440, height: 1000 });
    const afterRun = await project(page);
    expect(afterRun.data.roles).toEqual(before.data.roles);

    // Choose a completed, actionable row. Skipped/failed rows are never staged.
    const actionable = await panel
        .locator('input[aria-label^="Select feature "]:not([disabled])')
        .count();
    expect(actionable).toBeGreaterThan(0);
    const first = panel.locator('input[aria-label^="Select feature "]:not([disabled])').first();
    const displayName = (await first.getAttribute('aria-label')).replace('Select feature ', '');
    const rawName = displayName;
    const previousRole = before.data.roles[rawName];
    await first.check();
    const action = previousRole === 'predictor' ? 'Ignore selected' : 'Make selected predictors';
    await panel.getByRole('button', { name: action, exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply changes' })).toBeVisible();
    expect((await project(page)).data.roles).toEqual(before.data.roles);
    const intended = previousRole === 'predictor' ? 'ignore' : 'predictor';
    await expect(page.getByLabel('Role for ' + rawName)).toHaveValue(intended);
    await page.getByRole('button', { name: 'Variables JSON' }).click();
    const json = JSON.parse(await page.getByLabel('Variables JSON').inputValue());
    expect(json[intended]).toContain(rawName);
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes' }).click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    expect((await project(page)).data.roles[rawName]).toBe(intended);
    expect(errors).toEqual([]);
});

test('cancelled or changed drafts suppress late reports, and invalid JSON cannot start', async ({
    page,
}) => {
    await page.goto('/');
    const panel = page.getByRole('group', { name: 'One-way feature selection' });
    await panel.locator(':scope > summary').click();
    let release;
    const gate = new Promise((resolve) => {
        release = resolve;
    });
    await page.route('**/api/variables/feature-selection', async (route) => {
        const response = await route.fetch();
        await gate;
        await route.fulfill({ response });
    });
    const request = page.waitForRequest('**/api/variables/feature-selection');
    await panel.getByRole('button', { name: 'Run one-way feature selection' }).click();
    await request;
    await panel.getByRole('button', { name: 'Cancel feature selection' }).click();
    await page.getByLabel('Role for Noise').selectOption('ignore');
    const lateResponse = page.waitForResponse('**/api/variables/feature-selection');
    release();
    await lateResponse;
    await expect(panel.getByText(/tested of .* candidates/)).toHaveCount(0);
    await page.getByRole('button', { name: 'Variables JSON' }).click();
    await page.getByLabel('Variables JSON').fill('{unfinished');
    let starts = 0;
    page.on('request', (item) => {
        if (item.url().endsWith('/api/variables/feature-selection')) starts++;
    });
    await panel.getByRole('button', { name: 'Run one-way feature selection' }).click();
    await expect(page.getByRole('alert')).toContainText('JSON');
    expect(starts).toBe(0);
});
