import { test, expect } from '@playwright/test';

test('bidirectional roles, name/type preservation, invalid reset and explicit apply', async ({
    page,
}) => {
    const errors = [],
        requests = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => requests.push(r.url()));
    await page.goto('/');
    await expect(page.getByLabel('Role for DriverAge', { exact: true })).toHaveValue('predictor');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /Distribution of/ }).first()).toBeVisible();
    await page.getByRole('button', { name: /^Variables/ }).click();
    const base = requests.length;
    await page.getByLabel('Name for DriverAge', { exact: true }).fill('Age');
    await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await page.getByLabel('Type for DriverAge', { exact: true }).selectOption('categorical');
    await page.getByLabel('Role for VehicleAge', { exact: true }).selectOption('unassigned');
    await page.getByRole('button', { name: 'Role JSON', exact: true }).click();
    const text = page.getByLabel('Role JSON', { exact: true });
    const setup = JSON.parse(await text.inputValue());
    expect(Object.keys(setup)).toHaveLength(10);
    expect(setup.offset).toBeNull();
    expect(setup.ignore).toContain('InternalCode');
    expect(setup.unassigned).toContain('VehicleAge');
    setup.predictor = setup.predictor.filter((n) => n !== 'Region');
    setup.ignore.push('Region');
    await text.fill(JSON.stringify(setup, null, 2));
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByLabel('Role for Region', { exact: true })).toHaveValue('ignore');
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    await expect(page.getByLabel('Type for DriverAge', { exact: true })).toHaveValue('categorical');
    expect(requests.length).toBe(base); // draft edits and view changes are browser-local
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    await page.reload();
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    await page.getByRole('button', { name: 'Role JSON', exact: true }).click();
    await text.fill('{broken');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('alert')).toBeVisible();
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
    const reset = JSON.parse(await text.inputValue());
    expect(reset.ignore).toContain('Region');
    expect(reset.unassigned).toContain('VehicleAge');
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'Distribution of Region', exact: true }),
    ).toBeVisible();
    await page.getByLabel('Chart zoom', { exact: true }).fill('200');
    await expect(page.locator('.chart')).toHaveAttribute('style', /200%/);
    await page.screenshot({ path: 'test-results/variables.png', fullPage: true });
    expect(errors).toEqual([]);
    expect(requests.every((url) => url.startsWith('http://127.0.0.1:8770/'))).toBeTruthy();
});

test('singleton reassignment updates old row and invalid duplicate is refused', async ({
    page,
}) => {
    await page.goto('/');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('target');
    await expect(page.getByLabel('Role for Claims', { exact: true })).toHaveValue('unassigned');
    await page.getByRole('button', { name: 'Role JSON', exact: true }).click();
    const text = page.getByLabel('Role JSON', { exact: true });
    const draft = JSON.parse(await text.inputValue());
    draft.ignore.push('AnnualMileage');
    await text.fill(JSON.stringify(draft));
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText('appears more than once');
    await expect(text).toBeVisible();
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
});

test('pending preview freezes the draft until its result is reviewable', async ({ page }) => {
    await page.goto('/');
    await page.getByLabel('Role for Region', { exact: true }).selectOption('predictor');
    let release;
    await page.route('**/api/variables/preview', async (route) => {
        await new Promise((resolve) => {
            release = resolve;
        });
        await route.continue();
    });
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByLabel('Role for Region', { exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Reset', exact: true })).toBeDisabled();
    await expect.poll(() => Boolean(release)).toBe(true);
    release();
    await expect(page.getByLabel('Role for Region', { exact: true })).toBeEnabled();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeVisible();
});

test('workflow pages keep a Variables draft and expose honest project/export scope', async ({
    page,
}) => {
    await page.goto('/');
    await page.getByLabel('Name for Claims', { exact: true }).fill('KeptDraft');
    await page.getByLabel('Name for Claims', { exact: true }).press('Tab');
    for (const name of ['Project & data', 'Explore', 'Export']) {
        await page.getByRole('button', { name, exact: true }).click();
        await expect(page.getByRole('heading', { name, exact: true })).toBeVisible();
    }
    await expect(
        page.getByRole('button', { name: 'Download project JSON', exact: true }),
    ).toBeEnabled();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByLabel('Name for Claims', { exact: true })).toHaveValue('KeptDraft');
    await page.setViewportSize({ width: 884, height: 773 });
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
});
