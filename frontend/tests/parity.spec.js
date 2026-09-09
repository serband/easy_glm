import { test, expect } from '@playwright/test';
test('two-model diagnostics, paths, champion and search to refit', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    const tab = (name) => page.getByRole('tab', { name, exact: true });
    await page.goto('/');
    await page.getByLabel('Role for AnnualMileage', { exact: true }).selectOption('unassigned');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((el) => el.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await page.getByLabel('Penalty mode').selectOption('cv');
    await page.getByLabel('CV folds').fill('2');
    await page.getByLabel('Alpha path length').fill('5');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Diagnostics').click();
    await expect(
        page.getByText('Metrics and model facts side by side', { exact: true }),
    ).toHaveCount(0);
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await expect(page.getByRole('heading', { name: /DriverAge · train/ })).toBeVisible();
    await page.getByLabel('Diagnostic variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'Region · Holdout', exact: true }),
    ).toBeVisible();
    await expect(page.getByLabel('Diagnostic bins')).toHaveCount(0);
    const exposureX = await page
        .locator('.diagnostic-plot')
        .first()
        .locator('svg')
        .nth(1)
        .locator('rect')
        .evaluateAll((rects) => rects.map((rect) => Number(rect.getAttribute('x'))));
    expect(exposureX.length).toBe(4);
    expect(exposureX.at(-1) - exposureX[0]).toBeCloseTo(650);
    await tab('A/E by pair').click();
    await expect(button('Show pair A/E')).toHaveCount(0);
    await page.getByLabel('Pair first variable').selectOption('DriverAge');
    await page.getByLabel('Pair second variable').selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'DriverAge × Region · Holdout', exact: true }),
    ).toBeVisible();
    await tab('Regularisation path').click();
    await expect(page.getByRole('img', { name: /Stage 1.*regularisation path$/ })).toBeVisible();
    await page.setViewportSize({ width: 884, height: 773 });
    const path = page.locator('.path-chart').first();
    await expect(path.locator('.alpha-tick')).toHaveCount(5);
    await expect(path.locator('.selected-penalty')).toHaveCount(1);
    const displayed = await path.locator('svg text, svg title').allTextContents();
    expect(displayed.some((text) => /\d\.\d{4}/.test(text))).toBeFalsy();
    await expect(path).toContainText('Retained coefficients · right axis');
    const ticks = await path
        .locator('.alpha-tick')
        .evaluateAll((nodes) =>
            nodes.map((n) => ({ text: n.textContent, box: n.getBoundingClientRect().toJSON() })),
        );
    expect(ticks.every((t) => t.text.length <= 8)).toBeTruthy();
    for (let i = 1; i < ticks.length; i++)
        expect(ticks[i].box.x).toBeGreaterThan(ticks[i - 1].box.right);
    await expect(page.getByRole('img', { name: /retained coefficients$/ })).toHaveCount(0);
    await tab('Coefficients').click();
    await expect(page.locator('.diagnostic-table').last()).toContainText('exp coef');
    await tab('Double lift').click();
    await expect(page.getByText(/No challenger selected: the null benchmark/)).toBeVisible();
    await button('Model').click();
    await page.getByLabel('Model selection').selectOption('__new__');
    await page.getByLabel('New model name').fill('Challenger');
    await page.getByLabel('Penalty mode').selectOption('fixed');
    await page.getByLabel('Fixed alpha').fill('.1');
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Diagnostics').click();
    await tab('Regularisation path').click();
    await expect(page.locator('.path-chart .alpha-tick')).toHaveCount(1);
    await expect(page.locator('.path-chart .selected-penalty')).toHaveCount(1);
    await button('Model').click();
    await page.getByLabel('Model selection').selectOption('Frequency');
    await button('Diagnostics').click();
    await page.getByLabel('Compare with challenger').selectOption('Challenger');
    await expect(page.getByLabel('Default comparison model')).toHaveValue('Challenger');
    await tab('Lift').click();
    await expect(
        page.getByRole('img', { name: 'Challenger · holdout', exact: true }),
    ).toBeVisible();
    await tab('Double lift').click();
    await expect(
        page.getByRole('img', { name: 'Double lift · holdout', exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 884, height: 773 });
    await page.evaluate(() => window.scrollTo(0, 0));
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await page.screenshot({ path: '/tmp/easyglm-parity-double-lift.png', fullPage: true });
    await button('Compare').click();
    await expect(
        page.getByRole('heading', { name: 'Metrics side by side', exact: true }),
    ).toBeVisible();
    await expect(page.getByRole('tablist', { name: 'Diagnostics views' })).toHaveCount(0);
    await expect(page.getByText('Fit complete', { exact: true })).toHaveCount(0);
    await expect(page.locator('.metrics-grid')).toHaveCount(0);
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    await expect(page.getByLabel('Compare with challenger')).toHaveValue('Challenger');
    await button('Make selected model champion').click();
    await expect(
        page.getByText('Frequency is the project champion.', { exact: true }),
    ).toBeVisible();

    await expect(page.getByText(/Numeric factors use the union/)).toBeVisible();
    await button('Rate tables').click();
    await expect(page.getByLabel('Compare with challenger')).toHaveValue('Challenger');
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await button('Diagnostics').click();
    await tab('Residual factors').click();
    await button('Find missing factors').click();
    await page.getByRole('checkbox', { name: /AnnualMileage · signal/ }).check();
    await button('Add selected and review model').click();
    await expect(page.getByLabel('Include AnnualMileage', { exact: true })).toBeChecked();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Diagnostics').click();
    await tab('Residual factors').click();
    await button('Find missing interactions').click();
    await page.getByRole('button', { name: 'Add and review model', exact: true }).first().click();
    await expect(page.getByText(/Retained two-stage interactions/)).toBeVisible();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Diagnostics').click();
    await tab('Regularisation path').click();
    await expect(page.getByRole('img', { name: /Stage 2.*regularisation path$/ })).toBeVisible();
    await button('Rate tables').click();
    const interactionName = await page
        .getByLabel('Rate table variable', { exact: true })
        .locator('option')
        .evaluateAll((options) => options.map((o) => o.value).find((v) => v.includes('×')));
    expect(interactionName).toBeTruthy();
    await page.getByLabel('Rate table variable', { exact: true }).selectOption(interactionName);
    await expect(
        page.locator('.rate-relativities > .rate-chart-card .relativity-heatmap'),
    ).toBeVisible();
    await page.getByLabel('Adjustment method', { exact: true }).selectOption('manual');
    await expect(page.locator('.cell-edit-matrix')).toBeVisible();
    const cell = page.getByRole('spinbutton', { name: /^Relativity cell / }).first();
    await cell.fill('1.7');
    await cell.press('Tab');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await expect(page.locator('.rate-relativities .relativity-heatmap')).toBeVisible();
    const proposedAE = await page.locator('.ae-heatmap').innerText();
    await page.getByLabel('Heatmap model').selectOption('fitted_ae');
    await expect(page.locator('.ae-heatmap')).not.toHaveText(proposedAE);
    expect(errors).toEqual([]);
});
