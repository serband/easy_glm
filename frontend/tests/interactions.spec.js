import { test, expect } from '@playwright/test';

test('interaction definitions are visible, editable drafts and fitted only on request', async ({
    page,
}) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    const interactions = page.getByRole('region', { name: 'Defined interactions', exact: true });
    let saves = 0,
        fits = 0;
    const payloads = [];
    page.on('request', (request) => {
        if (request.method() !== 'POST') return;
        if (request.url().endsWith('/api/models/save')) {
            saves++;
            payloads.push(request.postDataJSON());
        }
        if (request.url().endsWith('/fit')) fits++;
    });
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((el) => el.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await expect(interactions).toContainText('No interactions defined.');
    await page.getByLabel('Interaction first factor', { exact: true }).selectOption('DriverAge');
    await page.getByLabel('Interaction second factor', { exact: true }).selectOption('Region');
    await page.getByLabel('New interaction minimum cell exposure').fill('0.1');
    await button('Add interaction').click();
    await expect(interactions.locator('.pair-name')).toHaveText('DriverAge × Region');
    await expect(page.getByRole('link', { name: 'Interactions (1)' })).toBeVisible();
    expect(saves).toBe(0);
    expect(fits).toBe(0);
    await button('Create model').click();
    await expect(
        page.getByText('Model settings saved. Fit when ready.', { exact: true }),
    ).toBeVisible();
    expect(payloads.at(-1).fields.interactions).toEqual([
        { a: 'DriverAge', b: 'Region', min_cell_exposure: 0.001, penalty_weight: 1 },
    ]);
    await expect(button('Fit model')).toBeEnabled();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    const fitCount = fits;

    await page.getByLabel('Interaction first factor', { exact: true }).selectOption('Region');
    await page.getByLabel('Interaction second factor', { exact: true }).selectOption('DriverAge');
    await expect(button('Add interaction')).toBeDisabled();
    await expect(interactions).toContainText('This pair is already defined.');
    await page.getByLabel('Include DriverAge', { exact: true }).uncheck();
    await expect(interactions).toContainText(
        'Select both main effects above, or remove this interaction.',
    );
    await expect(button('Save model settings')).toBeDisabled();
    await button('Reset model draft').click();
    await expect(interactions.locator('.pair-name')).toHaveText('DriverAge × Region');
    await expect(page.getByLabel('Include DriverAge', { exact: true })).toBeChecked();
    await page
        .getByLabel('Minimum cell exposure for DriverAge × Region', { exact: true })
        .fill('100');
    await expect(button('Save model settings')).toBeDisabled();
    await button('Reset model draft').click();

    await page.setViewportSize({ width: 884, height: 773 });
    await page.getByRole('link', { name: 'Interactions (1)' }).click();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await page.screenshot({ path: '/tmp/easyglm-interactions-editor.png', fullPage: true });
    await button('Remove interaction DriverAge × Region').click();
    await expect(interactions).toContainText('No interactions defined.');
    expect(fits).toBe(fitCount);
    await button('Reset model draft').click();
    await expect(interactions.locator('.pair-name')).toHaveText('DriverAge × Region');
    await button('Remove interaction DriverAge × Region').click();
    await button('Save model settings').click();
    await expect(
        page.getByText('Model settings saved. Fit when ready.', { exact: true }),
    ).toBeVisible();
    expect(payloads.at(-1).fields.interactions).toEqual([]);
    expect(fits).toBe(fitCount);
    await page.reload();
    await button('Model').click();
    await expect(interactions).toContainText('No interactions defined.');
    expect(errors).toEqual([]);
});
