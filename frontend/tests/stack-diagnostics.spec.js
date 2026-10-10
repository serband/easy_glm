import { test, expect } from '@playwright/test';

async function openBuilder(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
}

async function fitDraft(page, name, factor) {
    await page.getByLabel('Stacked model name').fill(name);
    await page.getByLabel(`Use ${factor}`).check();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
}

test('diagnostics identify the saved model and each point in its chain', async ({ page }) => {
    await openBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await fitDraft(page, 'Diagnostic base', 'DriverAge');

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await fitDraft(page, 'Diagnostic vehicle', 'VehicleAge');

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await fitDraft(page, 'Diagnostic region', 'Region');

    await page.getByRole('button', { name: 'Inspect actual versus expected', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Diagnostics — Diagnostic region', exact: true })).toBeVisible();
    const saved = page.getByLabel('Diagnostic saved model');
    const scope = page.getByLabel('Diagnostic scope');
    await expect(saved).toHaveValue('Diagnostic region');
    await expect(scope).toContainText('Final fit — after');
    await expect(scope).toContainText('After Diagnostic base');
    await scope.selectOption({ label: 'After Diagnostic vehicle' });
    await expect(page.locator('.diagnostic-scope')).toContainText('After Diagnostic vehicle');
    await expect(page.locator('.diagnostic-scope')).toContainText('Diagnostic base → Diagnostic vehicle');
    await expect(page.locator('.diagnostic-scope')).toContainText('Earlier models are included as the offset');

    await page.getByLabel('A/E factor', { exact: true }).selectOption('DriverAge');
    await expect(page.locator('table caption').first()).toContainText('Diagnostic region: After Diagnostic vehicle');
    await page.getByLabel('Diagnostic rows').selectOption('holdout');
    await expect(page.getByLabel('Diagnostic rows')).toHaveValue('holdout');

    // A late reply from the old checkpoint must not replace the newly selected one.
    let releaseOldRequest;
    const oldRequest = new Promise((resolve) => { releaseOldRequest = resolve; });
    let delayed = false;
    await page.route('**/api/stack/Diagnostic%20region/diagnostics*', async (route) => {
        if (!delayed && route.request().url().includes('through=')) {
            delayed = true;
            await oldRequest;
        }
        await route.continue();
    });
    await scope.selectOption({ label: 'After Diagnostic base' });
    await expect(page.getByText('Loading diagnostics…')).toBeVisible();
    await scope.selectOption(await scope.locator('option').filter({ hasText: 'Final fit' }).getAttribute('value'));
    releaseOldRequest();
    await expect(page.locator('.diagnostic-scope')).toContainText('Final fit — after Diagnostic region');
    await expect(page.getByText('Loading diagnostics…')).toHaveCount(0, { timeout: 60000 });
    await page.unroute('**/api/stack/Diagnostic%20region/diagnostics*');

    // Leaving Diagnostics while a request is delayed must not put its response in
    // the component-table view.
    let releaseLeavingDiagnostics;
    const leavingDiagnostics = new Promise((resolve) => { releaseLeavingDiagnostics = resolve; });
    await page.route('**/api/stack/Diagnostic%20region/diagnostics*', async (route) => {
        await leavingDiagnostics;
        await route.continue();
    });
    await scope.selectOption({ label: 'After Diagnostic base' });
    await expect(page.getByText('Loading diagnostics…')).toBeVisible();
    await page.getByRole('button', { name: 'Rate tables', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Component table or lookup', exact: true })).toBeVisible();
    await expect(page.getByLabel('Inspect model component')).toHaveValue(/.+/);
    releaseLeavingDiagnostics();
    await page.waitForTimeout(100);
    await expect(page.getByRole('heading', { name: 'Component table or lookup', exact: true })).toBeVisible();
    await page.unroute('**/api/stack/Diagnostic%20region/diagnostics*');

    // A saved model selector remains available, and makes the diagnostic target explicit.
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await saved.selectOption('Diagnostic base');
    await expect(page.getByRole('heading', { name: 'Diagnostics — Diagnostic base', exact: true })).toBeVisible();
    await expect(scope).toContainText('Final fit — after Diagnostic base');
    await expect(page.locator('.diagnostic-scope')).toContainText('This is the first model, using the project data.');

    await page.setViewportSize({ width: 797, height: 792 });
    expect(await page.locator('.stack-workbench').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
    await page.screenshot({ path: '/private/tmp/easyglm-stack-diagnostics-797.png', fullPage: true });
});
