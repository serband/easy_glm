import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('fitted variable cache is immediate, invalidates after edits and rejects late misses', async ({
    page,
}) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    let starts = 0;
    page.on('request', (r) => {
        if (
            r.method() === 'POST' &&
            /\/api\/review\/[^/]+$/.test(r.url()) &&
            r.postDataJSON().action === 'variable'
        )
            starts++;
    });
    await page.goto('/');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('unassigned');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await button('Model').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Diagnostics').click();
    const select = page.getByLabel('Diagnostic variable', { exact: true });
    await expect(
        page.getByRole('heading', { name: 'DriverAge · Holdout', exact: true }),
    ).toBeVisible();
    const initial = starts;
    for (const name of ['VehicleAge', 'Region', 'DriverAge']) {
        const now = Date.now();
        await select.selectOption(name);
        await expect(
            page.getByRole('heading', { name: name + ' · Holdout', exact: true }),
        ).toBeVisible();
        expect(Date.now() - now).toBeLessThan(500);
    }
    await page.getByLabel('Diagnostic subset', { exact: true }).selectOption('all');
    await expect(
        page.getByRole('heading', { name: 'DriverAge · All rows', exact: true }),
    ).toBeVisible();
    expect(starts).toBe(initial);
    await page.getByLabel('Diagnostic subset', { exact: true }).selectOption('holdout');
    let delayedId, release, held;
    const gate = new Promise((r) => (release = r)),
        waiting = new Promise((r) => (held = r));
    await page.route('**/api/review/*', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        if (route.request().postDataJSON().variable === 'AnnualMileage') delayedId = data.id;
        await route.fulfill({ response, json: data });
    });
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch();
        if (
            route.request().method() === 'GET' &&
            route
                .request()
                .url()
                .endsWith('/' + delayedId)
        ) {
            const data = await response.json();
            if (data.status === 'complete') {
                held();
                await gate;
            }
            await route.fulfill({ response, json: data });
        } else await route.fulfill({ response });
    });
    await select.selectOption('AnnualMileage');
    await waiting;
    await select.selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'Region · Holdout', exact: true }),
    ).toBeVisible();
    release();
    await page.waitForTimeout(350);
    await expect(
        page.getByRole('heading', { name: 'AnnualMileage · Holdout', exact: true }),
    ).toHaveCount(0);
    await page.unrouteAll({ behavior: 'wait' });
    await button('Rate tables').click();
    await page.getByLabel('Adjustment method', { exact: true }).selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('2.5');
    await page.getByLabel('Relativity row 2', { exact: true }).press('Tab');
    const edited = starts;
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await button('Diagnostics').click();
    await expect(page.getByRole('heading', { name: / · Holdout$/ }).first()).toBeVisible();
    expect(starts).toBeGreaterThan(edited);
    await select.selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'Region · Holdout', exact: true }),
    ).toBeVisible();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
});
