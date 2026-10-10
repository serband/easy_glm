import { test, expect } from '@playwright/test';

async function openStageVariables(page, index) {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    const choose = page.getByRole('button', { name: 'Choose variables', exact: true });
    await choose.nth(index).click();
}

test('a 500-column project keeps each planned stage searchable and isolated', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();

    await openStageVariables(page, 0);
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Save model variables', exact: true }).click();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();

    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: 'Add stage', exact: true }).click();
    await openStageVariables(page, 1);
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
    await page.getByRole('button', { name: 'Save model variables', exact: true }).click();

    await page.getByRole('tab', { name: 'Table' }).click();
    await page.getByLabel('Search model variables').fill('OptionalFactor499');
    await expect(page.getByLabel('Use OptionalFactor499')).toBeVisible();
    await expect(page.getByLabel('Use OptionalFactor499')).not.toBeChecked();
    await page.getByLabel('Search model variables').fill('OptionalFactor000');
    await expect(page.getByLabel('Use OptionalFactor000')).toBeChecked();
    await page.setViewportSize({ width: 804, height: 1100 });
    expect(await page.locator('.variables-scroll').evaluate(
        (element) => element.scrollWidth <= element.clientWidth,
    )).toBe(true);
});
