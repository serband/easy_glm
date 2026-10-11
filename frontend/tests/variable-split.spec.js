import { test, expect } from '@playwright/test';
test('explicit split maps table and JSON and reaches exploration and fit', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/?view=project');
    const shared = page.getByRole('region', { name: 'Shared column assignments' });
    const single = page.getByRole('button', { name: 'Single model', exact: true });
    await expect(single).toBeEnabled();
    await single.click();
    await expect(shared).toBeVisible();
    const button = (name) => page.getByRole('button', { name, exact: true });
    const train = page.getByLabel('Variable training value');
    const holdout = page.getByLabel('Variable holdout value');
    await expect(holdout).toHaveCount(0);
    await expect(train).toBeVisible();
    await expect(page.locator('.variable-split')).toHaveCount(0);
    await expect(page.getByText('Seeded random', { exact: true })).toHaveCount(0);
    await button('Preview changes').click();
    await expect(
        page.getByRole('alert').filter({ hasText: 'Choose a training value' }),
    ).toBeVisible();
    await train.selectOption('"TRAIN"');
    await expect(page.locator('.split-counts')).toContainText('9,000 training · 3,000 holdout');
    await button('Variables JSON').click();
    const editor = page.getByLabel('Variables JSON', { exact: true });
    const setup = JSON.parse(await editor.inputValue());
    expect(setup.split.column).toBe('train_test');
    expect(setup.split.train_value).toBe('TRAIN');
    expect(setup.split.holdout_value).toBeUndefined();
    setup.split.train_value = 'test';
    await editor.fill(JSON.stringify(setup, null, 2));
    await button('Table').click();
    await expect(page.locator('.split-counts')).toContainText('3,000 training · 9,000 holdout');
    await train.selectOption('"TRAIN"');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    await page.reload();
    await expect(train).toHaveValue('"TRAIN"');
    await expect(holdout).toHaveCount(0);
    await button('Explore').click();
    await expect(page.getByRole('img', { name: /One-way effects/ }).first()).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await button('Model').click();
    await expect(page.locator('.split-settings')).toHaveCount(0);
    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
    await button('Project & data').click();
    await page
        .getByRole('navigation', { name: 'Project setup steps' })
        .getByRole('button', { name: /Variables/ })
        .click();
    await page.getByLabel('Use DriverAge').check();
    await button('Continue to model details').click();
    await button('Continue to variable treatment').click();
    await button('Save Main effects setup and open Fit').click();
    await button('Model').click();
    await page.getByRole('button', { name: /^Fit / }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({ timeout: 45000 });
    await button('Project & data').click();
    await page
        .getByRole('navigation', { name: 'Project setup steps' })
        .getByRole('button', { name: /Shared columns/ })
        .click();
    await page.locator('.split-row-values').scrollIntoViewIfNeeded();
    await page.screenshot({
        path: testInfo.outputPath('variables-split.png'),
        fullPage: true,
    });
    await expect(page.locator('.split-counts')).toContainText('9,000 training · 3,000 holdout');
    page.once('dialog', (dialog) => dialog.accept());
    await button('Revise shared assignments').click();
    await expect(page.getByLabel('Role for DriverAge', { exact: true })).toBeEnabled();
    await page.getByLabel('Role for DriverAge', { exact: true }).selectOption('split');
    await expect(
        page.getByRole('alert').filter({ hasText: 'must have exactly two' }),
    ).toBeVisible();
    await expect(train).toBeDisabled();
    await button('Preview changes').click();
    await expect(
        page.getByRole('alert').filter({ hasText: 'must have exactly two' }).first(),
    ).toBeVisible();
    await page.getByLabel('Role for DriverAge', { exact: true }).selectOption('predictor');
    await expect(train).toHaveCount(0);
    await button('Preview changes').click();
    await button('Apply changes').click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    await button('Model').click();
    await expect(page.locator('.split-settings')).toHaveCount(0);
    await button('Project & data').click();
    await page.getByLabel('Training fraction', { exact: true }).fill('0.8');
    await page.getByLabel('Split seed', { exact: true }).fill('73');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await page.reload();
    await expect(page.getByLabel('Training fraction', { exact: true })).toHaveValue('0.8');
    await expect(page.getByLabel('Split seed', { exact: true })).toHaveValue('73');
    await button('Explore').click();
    await expect(page.getByRole('img', { name: /One-way effects/ }).first()).toBeVisible();
    await button('Model').click();
    await expect(page.getByRole('heading', { name: 'Fit models', exact: true })).toBeVisible();
    expect(errors).toEqual([]);
});
