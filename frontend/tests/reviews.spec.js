import { test, expect } from '@playwright/test';

test('diagnostic searches and table preview/apply/undo preserve the fit', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    await page.goto('/');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('unassigned');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((el) => el.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await expect(page.getByText('Split applied.', { exact: true })).toBeVisible();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible();
    await page.reload();
    await expect(button('Diagnostics')).toBeEnabled();
    await button('Diagnostics').click();
    await page.getByRole('tab', { name: 'A/E by variable', exact: true }).click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await page.getByRole('tab', { name: 'Residual factors', exact: true }).click();
    await button('Find missing factors').click();
    await expect(page.getByRole('heading', { name: 'Missing factors', exact: true })).toBeVisible();
    await expect(
        page.locator('.review-scroll').filter({ hasText: 'AnnualMileage' }).first(),
    ).toContainText('AnnualMileage');
    await button('Find missing interactions').click();
    await expect(
        page.getByRole('heading', { name: 'Missing interactions', exact: true }),
    ).toBeVisible();
    await page.getByRole('tab', { name: 'A/E by pair', exact: true }).click();
    await expect(page.locator('.ae-heatmap')).toBeVisible();
    await page.getByLabel('Diagnostic subset', { exact: true }).selectOption('train');
    await expect(page.getByRole('heading', { name: /Training/ })).toBeVisible();
    await page.getByRole('tab', { name: 'A/E by variable', exact: true }).click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 884, height: 773 });
    await page.evaluate(() => window.scrollTo(0, 0));
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await page.screenshot({ path: '/tmp/easyglm-workflow-diagnostics.png', fullPage: true });

    await button('Rate tables').click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 884, height: 773 });
    await page.evaluate(() => window.scrollTo(0, 0));
    await expect(page.locator('.rate-grid')).not.toBeVisible();
    const chart = await page.locator('.rate-chart-card').boundingBox();
    const tableSection = await page.locator('.rate-table-card').boundingBox();
    expect(chart.y).toBeLessThan(500);
    expect(chart.width).toBeGreaterThan(600);
    expect(tableSection.y).toBeGreaterThan(chart.y + chart.height - 1);
    await expect(page.locator('.numeric-curve')).toHaveCount(2);
    await expect(
        page.getByRole('img', { name: /^Fitted and current relativities for/ }),
    ).toBeVisible();
    await expect(page.getByRole('img', { name: /^Exposure for/ })).toBeVisible();
    await expect(button('View rate tables')).toHaveCount(0);
    await expect(button('Show variable A/E')).toHaveCount(0);
    await page.screenshot({ path: '/tmp/easyglm-rate-layout-884.png', fullPage: true });
    await page.locator('.rate-table-card > summary').click();
    await page.getByLabel('Relativity row 2', { exact: true }).fill('2.1');
    await page.getByLabel('Relativity row 2', { exact: true }).press('Tab');
    await page.locator('.rate-table-card > summary').click();
    await expect(page.locator('.rate-grid')).not.toBeVisible();
    await page.locator('.rate-table-card > summary').click();
    await expect(page.getByLabel('Relativity row 2', { exact: true })).toHaveValue('2.1');
    await button('Preview row edits (1)').click();
    await expect(button('Apply adjustment')).toBeVisible();
    await expect(
        page.getByText('Preview only; no settings have changed.', { exact: false }),
    ).toBeVisible();
    await button('Apply adjustment').click();
    await expect(page.getByLabel('Relativity row 2', { exact: true })).toHaveValue('2.1');
    await expect(button('Preview undo')).toBeEnabled();
    await button('Preview undo').click();
    await button('Apply adjustment').click();
    await expect(page.getByLabel('Relativity row 2', { exact: true })).not.toHaveValue('2.1');
    await button('Cap / floor').click();
    await page.getByLabel('Relativity cap').fill('1.1');
    await expect(button('Apply adjustment')).toBeEnabled();
    await expect(button('Apply adjustment')).toBeVisible();
    await page.locator('.preview-impact summary').first().click();
    await expect(page.locator('.preview-impact table')).toContainText('Before');
    await page.setViewportSize({ width: 884, height: 808 });
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await page.screenshot({ path: 'test-results/restored-adjustments.png', fullPage: true });
    await page.setViewportSize({ width: 1440, height: 1000 });
    await button('Discard preview').click();
    await expect(button('Preview rebalance base rate')).toBeEnabled();
    await button('Preview rebalance base rate').click();
    await expect(button('Apply adjustment')).toBeVisible();
    await button('Discard preview').click();
    await button('Diagnostics').click();
    await page.getByRole('tab', { name: 'A/E by variable', exact: true }).click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await page.getByRole('tab', { name: 'Residual factors', exact: true }).click();
    await button('Find missing factors').click();
    await page
        .getByRole('row')
        .filter({ hasText: 'AnnualMileage' })
        .getByRole('button', { name: 'Add and review model', exact: true })
        .click();
    await expect(page.getByLabel('Include AnnualMileage', { exact: true })).toBeChecked();
    await expect(page.getByText('Fit needs updating', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByLabel('Role for AnnualMileage', { exact: true })).toHaveValue(
        'predictor',
    );
    expect(errors).toEqual([]);
});
