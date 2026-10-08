import { test, expect } from '@playwright/test';

async function fitLegacy(page) {
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Create model', exact: true }).click();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 60000 });
}
function flowCard(page, name) {
    return page.getByRole('article', { name: `${name}, fitted`, exact: true });
}
async function saveStage(page, { name, stage, kind, variable }) {
    await page.getByLabel('New stacked model name').fill(name);
    await page.getByLabel('New stage type').selectOption(kind);
    await page.getByLabel('New stage name').fill(stage);
    await page.getByRole('group', { name: 'Stage factors' }).getByText(variable, { exact: true }).click();
    // Search does not discard an existing checked variable.
    await page.getByLabel('Search stage factors').fill('no matching variable');
    await expect(page.getByText(/1 selected:/)).toBeVisible();
    await page.getByRole('button', { name: 'Show selected' }).click();
    await expect(page.getByRole('group', { name: 'Stage factors' }).getByText(variable, { exact: true })).toBeVisible();
    if (kind === 'catboost') {
        await page.getByLabel('CatBoost iterations').fill('20');
        await page.getByLabel('CatBoost time limit').fill('1');
    }
    await page.getByRole('button', { name: 'Save new candidate' }).click();
    await page.getByLabel('Stacked model', { exact: true }).selectOption(name);
    await page.getByRole('button', { name: 'Fit candidate' }).click();
    await expect(page.getByText('Review complete model', { exact: true })).toBeVisible({ timeout: 60000 });
}

test('visual flow builds standalone GLM, CatBoost, GLM and a sibling alternative', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto('/');
    await fitLegacy(page);
    await page.getByRole('button', { name: 'Build a stacked model' }).click();
    await expect(page.getByRole('region', { name: 'Model flow' })).toBeVisible();
    await page.getByRole('button', { name: 'Add standalone model' }).click();
    await saveStage(page, { name: 'Flow Root GLM', stage: 'Root region', kind: 'glm', variable: 'Region' });
    await flowCard(page, 'Flow Root GLM').getByRole('button', { name: 'Add model after this', exact: true }).click();
    await saveStage(page, { name: 'Flow CatBoost', stage: 'Vehicle correction', kind: 'catboost', variable: 'VehicleAge' });
    await flowCard(page, 'Flow CatBoost').getByRole('button', { name: 'Add model after this', exact: true }).click();
    await saveStage(page, { name: 'Flow GLM', stage: 'Region correction', kind: 'glm', variable: 'Region' });
    await expect(flowCard(page, 'Flow GLM')).toContainText('Builds on Flow CatBoost · complete model predictions as offset');
    await flowCard(page, 'Flow CatBoost').getByRole('button', { name: 'Create alternative', exact: true }).click();
    await expect(page.getByLabel('Start from fitted candidate')).toHaveValue('Flow Root GLM');
    await expect(page.getByLabel('New stage name')).toHaveValue(/Vehicle correction alternative/);
    await expect(page.getByText(/1 selected: VehicleAge/)).toBeVisible();
    await page.getByRole('group', { name: 'Stage factors' }).getByText('Region', { exact: true }).click();
    await page.getByLabel('New stacked model name').fill('Flow CatBoost alternative');
    await page.getByRole('button', { name: 'Save new candidate' }).click();
    await page.getByLabel('Stacked model', { exact: true }).selectOption('Flow CatBoost alternative');
    await page.getByRole('button', { name: 'Fit candidate' }).click();
    await expect(page.getByText('Review complete model', { exact: true })).toBeVisible({ timeout: 60000 });
    await expect(flowCard(page, 'Flow CatBoost alternative')).toContainText('Builds on Flow Root GLM · complete model predictions as offset');
    await page.getByRole('button', { name: 'Reset editor' }).click();
    await expect(page.getByText('No factors selected')).toBeVisible();
    for (const viewport of [{ width: 867, height: 850 }, { width: 390, height: 844 }]) {
        await page.setViewportSize(viewport);
        expect(await page.locator('.model-flow').evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
        expect(await page.locator('.flow-card').evaluateAll((cards) => cards.every((card) => card.scrollWidth <= card.clientWidth))).toBe(true);
        if (viewport.width === 867) await page.screenshot({ path: 'test-results/model-flow-desktop.png', fullPage: true });
    }
    const offsetCaption = flowCard(page, 'Flow CatBoost alternative').locator('.flow-select small');
    expect(await offsetCaption.evaluate((caption) => {
        const card = caption.closest('.flow-card');
        return caption.getBoundingClientRect().right <= card.getBoundingClientRect().right;
    })).toBe(true);
    await page.screenshot({ path: 'test-results/model-flow-mobile.png', fullPage: true });
    await flowCard(page, 'Frequency').locator('.flow-select').click();
    await expect(page.getByRole('button', { name: 'Build a stacked model' })).toBeVisible();
    expect(errors).toEqual([]);
});
