import { test, expect } from '@playwright/test';

test('unchanged Variables preview continues to upfront model-plan setup', async ({ page }) => {
    await page.goto('/?view=variables');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('status').filter({ hasText: 'No variable changes to apply' })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Continue to Model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Define a model plan', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Go to Project & data', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Build in stages', exact: true })).toBeVisible();
});

test('saved stacked projects retain their model view and route new setup through Project', async ({ page }) => {
    await page.route('**/api/workbench', async (route) => {
        const response = await route.fetch();
        const workbench = await response.json();
        const template = Object.values(workbench.models)[0];
        workbench.models = {
            'Stack candidate': {
                ...template,
                components: [{ component_id: 'stage-1', name: 'First GLM', kind: 'glm', factors: ['DriverAge'], settings: {}, amendments: [] }],
            },
        };
        workbench.champion = 'Stack candidate';
        workbench.jobs = {};
        await route.fulfill({ response, json: workbench });
    });
    await page.goto('/?view=variables');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to Model', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Build a model in stages', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toHaveCount(0);
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Single model', exact: true })).toBeVisible();
    await expect(page.getByRole('button', { name: 'Build in stages', exact: true })).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
});

test('bidirectional roles, name/type preservation, invalid reset and explicit apply', async ({
    page,
}) => {
    const errors = [],
        requests = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => requests.push(r.url()));
    await page.goto('/?view=variables');
    await expect(page.getByLabel('Role for DriverAge', { exact: true })).toHaveValue('predictor');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /One-way effects of/ }).first()).toBeVisible();
    await page.getByRole('button', { name: /^Variables\s*\d*$/ }).click();
    const base = requests.length;
    await page.getByLabel('Name for DriverAge', { exact: true }).fill('Age');
    await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await page.getByLabel('Type for DriverAge', { exact: true }).selectOption('categorical');
    await page.getByLabel('Role for VehicleAge', { exact: true }).selectOption('unassigned');
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const text = page.getByLabel('Variables JSON', { exact: true });
    const setup = JSON.parse(await text.inputValue());
    expect(Object.keys(setup)).toHaveLength(12);
    expect(setup.binning).toEqual({ default_bins: 20, overrides: {} });
    expect(setup.time).toBeNull();
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
    await expect(page.locator('.editor-card > .preview-card')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeEnabled();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    await page.reload();
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
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
        page.getByRole('img', { name: 'One-way effects of Region', exact: true }),
    ).toBeVisible();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('VehicleAge');
    await page.getByLabel('Bands', { exact: true }).fill('10');
    await expect(page.getByLabel('Bands', { exact: true })).toHaveValue('10');
    await expect(
        page.getByRole('img', { name: 'One-way effects of VehicleAge', exact: true }),
    ).toBeVisible();
    await page.screenshot({ path: 'test-results/variables.png', fullPage: true });
    expect(errors).toEqual([]);
    expect(requests.every((url) => url.startsWith(new URL(page.url()).origin + '/'))).toBeTruthy();
});

test('singleton reassignment updates old row and invalid duplicate is refused', async ({
    page,
}) => {
    await page.goto('/?view=variables');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('target');
    await expect(page.getByLabel('Role for Claims', { exact: true })).toHaveValue('unassigned');
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const text = page.getByLabel('Variables JSON', { exact: true });
    const draft = JSON.parse(await text.inputValue());
    draft.ignore.push('AnnualMileage');
    await text.fill(JSON.stringify(draft));
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText('appears more than once');
    await expect(text).toBeVisible();
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
});

test('pending preview freezes the draft until its result is reviewable', async ({ page }) => {
    await page.goto('/?view=variables');
    const regionRole = page.getByLabel('Role for Region', { exact: true });
    const currentRole = await regionRole.inputValue();
    await regionRole.selectOption(currentRole === 'ignore' ? 'unassigned' : 'ignore');
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
    await page.goto('/?view=variables');
    await page.getByLabel('Name for Claims', { exact: true }).fill('KeptDraft');
    await page.getByLabel('Name for Claims', { exact: true }).press('Tab');
    for (const name of ['Project & data', 'Explore', 'Export']) {
        await page.getByRole('button', { name, exact: true }).click();
        await expect(
            page.getByRole('heading', {
                name: name === 'Explore' ? 'One-way effects' : name,
                exact: true,
            }),
        ).toBeVisible();
    }
    await expect(
        page.getByRole('button', { name: 'Download project JSON', exact: true }),
    ).toBeEnabled();
    await page.getByRole('button', { name: /^Variables\s*\d*$/ }).click();
    await expect(page.getByLabel('Name for Claims', { exact: true })).toHaveValue('KeptDraft');
    await page.setViewportSize({ width: 884, height: 773 });
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
});
