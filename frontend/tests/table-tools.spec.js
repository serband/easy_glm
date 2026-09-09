import { test, expect } from '@playwright/test';
test.use({ actionTimeout: 15000 });

test('visible rate adjustment methods preview, apply and undo real changes', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((el) => el.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Rate tables').click();
    await expect(
        page.getByRole('img', { name: 'Actual fitted and adjusted by variable', exact: true }),
    ).toBeVisible();
    await page.setViewportSize({ width: 884, height: 773 });
    const chart = await page.locator('.rate-relativities > .rate-chart-card').boundingBox();
    const tools = page.locator('.adjustments-heading');
    const toolBox = await tools.boundingBox();
    const grid = await page.locator('.rate-table-card').boundingBox();
    const aeBox = await page.locator('.rate-ae').boundingBox();
    expect(toolBox.y).toBeGreaterThan(chart.y + chart.height - 2);
    expect(grid.y).toBeGreaterThan(toolBox.y);
    expect(aeBox.y).toBeGreaterThan(grid.y + grid.height);
    await expect(page.locator('.rate-relativities .rate-chart-card')).toHaveCount(1);
    await expect(
        page.locator('.rate-ae').getByRole('button', { name: 'Preview adjustment', exact: true }),
    ).toHaveCount(0);
    await expect(page.locator('.rate-grid')).not.toBeVisible();
    for (const name of [
        'Moving average',
        'Isotonic smoothing',
        'Cap / floor',
        'Round',
        'Edit individual or multiple rows',
    ])
        await expect(button(name)).toBeVisible();
    await tools.scrollIntoViewIfNeeded();
    await page.screenshot({ path: '/tmp/easyglm-visible-table-tools.png' });
    await button('Edit individual or multiple rows').click();
    await page.getByLabel('Relativity row 2', { exact: true }).fill('1.37');
    await page.getByLabel('Relativity row 2', { exact: true }).press('Tab');
    await page.getByLabel('Relativity row 3', { exact: true }).fill('2.41');
    await page.getByLabel('Relativity row 3', { exact: true }).press('Tab');
    await button('Preview row edits (2)').click();
    await expect(
        page.getByRole('img', {
            name: 'Original fit and adjusted preview for DriverAge',
            exact: true,
        }),
    ).toBeVisible();
    await expect(page.locator('.preview-impact')).toContainText('Training expected:');
    await expect(page.locator('.rate-ae')).toContainText('Adjusted');
    await expect(page.locator('.rate-relativities .rate-chart-card')).toHaveCount(1);
    await expect(
        page
            .locator('.rate-relativities')
            .getByRole('button', { name: 'Apply adjustment', exact: true }),
    ).toHaveCount(1);
    await page.screenshot({ path: '/tmp/easyglm-manual-rate-preview.png' });
    await button('Apply adjustment').click();
    await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
    await page.locator('.rate-table-card > summary').click();
    await page.locator('.table-snapshots > summary').click();
    await page.getByLabel('Snapshot name').fill('Manual starting point');
    await button('Save snapshot').click();
    await expect(page.getByText('Snapshot saved in the project.', { exact: true })).toBeVisible();
    const modes = [
        ['Moving average', async () => page.getByLabel('Smoothing window').fill('3')],
        [
            'Isotonic smoothing',
            async () => page.getByLabel('Smoothing direction').selectOption('increasing'),
        ],
        [
            'Isotonic smoothing',
            async () => page.getByLabel('Smoothing direction').selectOption('decreasing'),
        ],
        [
            'Cap / floor',
            async () => {
                await page.getByLabel('Relativity floor').fill('1.1');
                await page.getByLabel('Relativity cap').fill('');
            },
        ],
        [
            'Cap / floor',
            async () => {
                await page.getByLabel('Relativity floor').fill('');
                await page.getByLabel('Relativity cap').fill('1.2');
            },
        ],
        [
            'Cap / floor',
            async () => {
                await page.getByLabel('Relativity floor').fill('1.1');
                await page.getByLabel('Relativity cap').fill('1.2');
            },
        ],
        [
            'Round',
            async () => {
                await page.getByLabel('Rounding mode').selectOption('decimals');
                await page.getByLabel('Decimal places').fill('1');
            },
        ],
        [
            'Round',
            async () => {
                await page.getByLabel('Rounding mode').selectOption('step');
                await page.getByLabel('Rounding step').fill('.25');
            },
        ],
    ];
    for (const [mode, parameters] of modes) {
        await button(mode).click();
        await parameters();
        await expect(
            page.getByRole('img', {
                name: 'Original fit and adjusted preview for DriverAge',
                exact: true,
            }),
        ).toBeVisible();
        await expect(button('Apply adjustment')).toBeEnabled();
        await expect(page.locator('.preview-impact')).toContainText('changed rows');
        await button('Apply adjustment').click();
        await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
        await expect(button('Preview undo')).toBeEnabled();
        await button('Preview undo').click();
        await expect(button('Apply adjustment')).toBeEnabled();
        await button('Apply adjustment').click();
        await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
    }
    await button('Moving average').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await page.getByLabel('Smoothing window').fill('5');
    await expect(button('Apply adjustment')).toBeDisabled();
    await expect(button('Apply adjustment')).toBeEnabled();
    await expect(button('Discard preview')).toBeVisible();
    await button('Discard preview').click();
    await expect(button('Apply adjustment')).toHaveCount(0);
    await button('Preview rebalance base rate').click();
    await expect(page.locator('.preview-impact')).toContainText('Base rate:');
    await button('Apply adjustment').click();
    await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
    await button('Reset this variable').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
    await page.getByLabel('Saved snapshot').selectOption('Manual starting point');
    await button('Preview snapshot restore').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText(/Adjustments applied\./)).toBeVisible();
    await page.getByLabel('Second table version').selectOption('Manual starting point');
    await button('Compare table versions').click();
    await expect(page.getByText('Snapshot differences', { exact: true })).toBeVisible();
    await button('Delete snapshot').click();
    await expect(button('Confirm delete snapshot')).toBeVisible();
    await button('Keep snapshot').click();
    await button('Delete snapshot').click();
    await button('Confirm delete snapshot').click();
    await expect(page.getByText('Snapshot deleted.', { exact: true })).toBeVisible();
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('heading', { name: 'Region · Holdout', exact: true }),
    ).toBeVisible();
    await button('Moving average').click();
    await expect(button('Apply adjustment')).toBeDisabled();
    await page
        .getByRole('checkbox', {
            name: 'The levels of this factor are in a meaningful order',
            exact: true,
        })
        .check();
    await expect(
        page.getByRole('img', {
            name: 'Original fit and adjusted preview for Region',
            exact: true,
        }),
    ).toBeVisible();
    await button('Discard preview').click();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
    ).toBeTruthy();
    expect(errors).toEqual([]);
});
