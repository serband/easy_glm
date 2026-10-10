import { test, expect } from '@playwright/test';

test('a 500-column project keeps per-model JSON selections searchable and isolated', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await page.getByLabel('Stacked model name').fill('Wide base');
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });

    await page.getByRole('button', { name: 'Add model after this', exact: true }).click();
    await page.getByLabel('Stacked model name').fill('Wide child');
    await page.getByRole('tab', { name: 'JSON' }).click();
    const variables = Object.fromEntries(
        Array.from({ length: 100 }, (_, number) => [
            `OptionalFactor${String(number).padStart(3, '0')}`,
            { kind: 'categorical' },
        ])
    );
    await page.getByLabel('Model variables JSON').fill(JSON.stringify({ variables }, null, 2));
    await page.getByRole('button', { name: 'Apply variable JSON', exact: true }).click();
    await expect(page.getByText('100 selected', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Save draft', exact: true }).click();
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();
    await expect(page.getByText('100 selected', { exact: true })).toBeVisible();
    await page.getByLabel('Search model variables').fill('OptionalFactor499');
    await expect(page.getByLabel('Use OptionalFactor499')).toBeVisible();
    await expect(page.getByLabel('Use OptionalFactor499')).not.toBeChecked();
    await page.getByLabel('Search model variables').fill('OptionalFactor000');
    await expect(page.getByLabel('Use OptionalFactor000')).toBeChecked();
    await page.screenshot({ path: '/private/tmp/easyglm-model-variables-wide-desktop.png', fullPage: true });
    await page.setViewportSize({ width: 804, height: 1100 });
    expect(await page.locator('.variables-scroll').evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
    )).toBe(true);
    await page.screenshot({ path: '/private/tmp/easyglm-model-variables-wide-804.png', fullPage: true });
});
