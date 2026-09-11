import { test, expect } from '@playwright/test';
test('model definition controls preserve selected values without browser errors', async ({
    page,
}) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    await page.goto('/');
    await button('Model').click();
    await page.getByLabel('Model family', { exact: true }).selectOption('tweedie');
    await page.getByLabel('Tweedie power', { exact: true }).fill('1.7');
    await expect(page.getByLabel('Tweedie power', { exact: true })).toHaveValue('1.7');
    expect(errors).toEqual([]);
    for (const [field, value] of [
        ['Model family', 'gamma'],
        ['Model family', 'poisson'],
        ['Model link', 'log'],
        ['Model target', 'AnnualMileage'],
        ['Model target', 'Claims'],
        ['Model weight', 'AnnualMileage'],
        ['Model weight', 'Exposure'],
        ['Model offset', 'AnnualMileage'],
        ['Table base', 'reference'],
    ]) {
        await page.getByLabel(field, { exact: true }).selectOption(value);
        await expect(page.getByLabel(field, { exact: true })).toHaveValue(value);
        expect(errors, field).toEqual([]);
    }
    await page.getByLabel('Divide target by weight', { exact: true }).uncheck();
    expect(errors).toEqual([]);
    await button('Create model').click();
    await expect(
        page.getByText('Model settings saved. Fit when ready.', { exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel('Model offset', { exact: true })).toHaveValue('AnnualMileage');
    await expect(page.getByLabel('Table base', { exact: true })).toHaveValue('reference');
    await expect(page.getByLabel('Divide target by weight', { exact: true })).not.toBeChecked();
    expect(errors).toEqual([]);
});
