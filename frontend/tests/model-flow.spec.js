import { test, expect } from '@playwright/test';

const setupStep = (page, name) =>
    page
        .getByRole('navigation', { name: 'Project setup steps' })
        .getByRole('button', { name: new RegExp(name) });

test('project setup chooses the staged workflow before its model variables and fit settings', async ({
    page,
}) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize({ width: 1332, height: 1000 });
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Define a model plan', exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Go to Project & data', exact: true }).click();

    await expect(page.getByRole('heading', { name: 'Project & data', exact: true })).toBeVisible();
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
    await expect(page.getByLabel('Model 1 name')).toBeVisible();
    await page.getByRole('button', { name: 'Choose variables', exact: true }).click();

    await expect(
        page.getByRole('heading', { name: 'Variables for Main effects', exact: true }),
    ).toBeVisible();
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Fit Model 1 — Main effects', exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(page.getByText('Fitting Main effects', { exact: false })).toBeVisible();
    await expect(page.getByText('All models are fitted.', { exact: false })).toBeVisible({
        timeout: 60000,
    });

    // A later model is defined from the same project data, but its fit starts
    // from the complete saved score of the preceding model.
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Models').click();
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await setupStep(page, 'Variables').click();
    await page.getByLabel('Model stage').selectOption({ index: 1 });
    await page.getByLabel('Use VehicleAge').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(
        page.getByText(/^Will use the complete score through Main effects as its offset/),
    ).toBeVisible();
    await page.getByRole('button', { name: /^Fit / }).click();
    await expect(page.getByText('All models are fitted.', { exact: false })).toBeVisible({
        timeout: 60000,
    });

    await page.setViewportSize({ width: 390, height: 844 });
    expect(
        await page
            .locator('.plan-fitting')
            .evaluate((element) => element.scrollWidth <= element.clientWidth),
    ).toBe(true);
    expect(errors).toEqual([]);
});
