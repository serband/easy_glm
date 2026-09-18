import { test, expect } from '@playwright/test';

function cutLines(panel) {
    return panel
        .locator('.model-cut')
        .evaluateAll((lines) => lines.map((line) => Number(line.getAttribute('data-cut'))));
}

test('numeric binning stays in sync and reaches fitted rate tables', async ({ page }, testInfo) => {
    const errors = [];
    let previews = 0;
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => {
        if (request.url().endsWith('/api/variables/binning-preview')) previews++;
    });
    await page.goto('/');
    const button = (name) => page.getByRole('button', { name, exact: true });
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    await expect(panel).toBeVisible();
    expect(previews).toBe(0);
    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    const cuts = page.getByLabel('Custom cuts for VehicleAge');
    await cuts.fill('0, 1, 2, 3, 4, 5');
    expect(previews).toBe(0);
    const responsePromise = page.waitForResponse('**/api/variables/binning-preview');
    await button('Preview bins for VehicleAge').click();
    const preview = await (await responsePromise).json();
    expect(preview.actual_bins).toBe(7);
    expect(preview.training_rows).toBe(1800);
    expect(preview.rows.filter((row) => row.lower === 0 && row.upper === 1)).toHaveLength(1);
    expect(preview.rows.find((row) => row.lower === 0 && row.upper === 1).rows).toBe(201);
    await expect(panel).toContainText('7 actual bins');
    await cuts.fill('0, 1,');
    await expect(button('Preview changes')).toBeDisabled();
    await expect(panel.getByRole('alert')).toBeVisible();
    await cuts.fill('0, 1, 2');
    await page.getByLabel('Default number of bins', { exact: true }).fill('');
    await cuts.fill('0, 1, 2, 3, 4, 5');
    await expect(button('Preview changes')).toBeDisabled();
    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await expect(button('Preview changes')).toBeEnabled();
    await button('Variables JSON').click();
    expect(
        JSON.parse(await page.getByLabel('Variables JSON', { exact: true }).inputValue()).binning
            .overrides.VehicleAge.cuts,
    ).toEqual([0, 1, 2, 3, 4, 5]);
    await button('Table').click();
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    // Method changes must survive while another field is invalid.
    await page.getByLabel('Default number of bins', { exact: true }).fill('');
    await page.getByLabel('Binning method for VehicleAge').selectOption('default');
    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await expect(page.getByLabel('Binning method for VehicleAge')).toHaveValue('default');
    await button('Variables JSON').click();
    expect(
        JSON.parse(await page.getByLabel('Variables JSON', { exact: true }).inputValue()).binning
            .overrides.VehicleAge,
    ).toBeUndefined();
    await button('Table').click();
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    // Leaving an invalid method must clear that method's stale validation too.
    await cuts.fill('0, 1,');
    await page.getByLabel('Binning method for VehicleAge').selectOption('default');
    await expect(button('Preview changes')).toBeEnabled();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    await cuts.fill('0, 1, 2, 3, 4, 5');
    await button('Variables JSON').click();
    const jsonEditor = page.getByLabel('Variables JSON', { exact: true });
    const setup = JSON.parse(await jsonEditor.inputValue());
    expect(setup.binning.default_bins).toBe(8);
    expect(setup.binning.overrides.VehicleAge).toEqual({
        method: 'cuts',
        cuts: [0, 1, 2, 3, 4, 5],
    });
    setup.binning.overrides.DriverAge = { method: 'quantile', bins: 6 };
    await jsonEditor.fill(JSON.stringify(setup, null, 2));
    await button('Table').click();
    await panel.getByRole('button', { name: /^DriverAge/ }).click();
    await expect(page.getByLabel('Bins for DriverAge', { exact: true })).toHaveValue('6');
    const changeResponse = page.waitForResponse('**/api/variables/preview');
    await button('Preview changes').click();
    const changePreview = await changeResponse;
    expect(changePreview.status(), JSON.stringify(await changePreview.json())).toBe(200);
    expect(errors).toEqual([]);
    await expect(button('Apply changes')).toBeVisible();
    await button('Apply changes').click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    await page.reload();
    await expect(page.getByLabel('Default number of bins', { exact: true })).toHaveValue('8');
    await button('Variables JSON').click();
    expect(JSON.parse(await jsonEditor.inputValue()).binning).toEqual(setup.binning);
    await button('Table').click();
    await button('Model').click();
    await expect(page.getByLabel('Default bins', { exact: true })).toHaveCount(0);
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    const api = async (path) =>
        page.evaluate(async (path) => {
            const { token } = await (await fetch('/api/session')).json();
            const response = await fetch(`/api/${path}`, { headers: { 'X-EasyGLM-Token': token } });
            return { status: response.status, body: await response.json() };
        }, path);
    const table = await api('results/Frequency/table?variable=VehicleAge');
    expect(table.status).toBe(200);
    expect(table.body.rows.some((row) => row.from === 0 && row.to === 1)).toBe(true);
    const exported = await api('project');
    expect(exported.body.design.defaults.n_bins).toBe(8);
    expect(exported.body.design.variables.VehicleAge.knots).toEqual([0, 1, 2, 3, 4, 5]);
    await page.getByRole('button', { name: /^Variables\s*\d*$/ }).click();
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await button('Preview bins for VehicleAge').click();
    await expect(panel).toContainText('7 actual bins');
    await page.setViewportSize({ width: 799, height: 803 });
    await panel.scrollIntoViewIfNeeded();
    expect(
        await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
    ).toBe(true);
    await page.screenshot({ path: testInfo.outputPath('numeric-binning.png'), fullPage: true });
    await page.getByLabel('Default number of bins', { exact: true }).fill('9');
    await button('Preview changes').click();
    await button('Apply changes').click();
    await expect(page.getByRole('status').filter({ hasText: 'Settings applied' })).toBeVisible();
    expect((await api('results/Frequency')).status).toBe(409);
    const downloadEvent = page.waitForEvent('download');
    await button('Export project').click();
    const download = await downloadEvent;
    const projectPath = testInfo.outputPath('binning-project.json');
    await download.saveAs(projectPath);
    await button('Project & data').click();
    await page.getByRole('radio', { name: 'Saved project', exact: true }).check();
    await page.getByLabel('Project file', { exact: true }).setInputFiles(projectPath);
    await page.getByLabel('Replace this session.', { exact: false }).check();
    await button('Open project').click();
    await expect(page.getByLabel('Default number of bins', { exact: true })).toHaveValue('9');
    await button('Variables JSON').click();
    const reopened = JSON.parse(await jsonEditor.inputValue()).binning;
    expect(reopened.overrides.VehicleAge.cuts).toEqual([0, 1, 2, 3, 4, 5]);
    expect(reopened.overrides.DriverAge.bins).toBe(6);
    expect(errors).toEqual([]);
});

test('training distribution stays fixed while default bins and custom cuts move', async ({
    page,
}) => {
    await page.goto('/');
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    await panel.getByRole('button', { name: /^Mileage/ }).click();
    const firstResponse = page.waitForResponse('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for Mileage' }).click();
    const first = await (await firstResponse).json();
    expect(first.histogram.bars).toHaveLength(40);
    expect(first.histogram.finite_rows).toBe(1800);
    expect(first.histogram.bars.reduce((total, bar) => total + bar.rows, 0)).toBe(1800);
    const figure = panel.locator('figure.binning-histogram');
    await expect(figure).toContainText('Equal-width bars show the distribution');
    await expect(figure.locator('.distribution-bar')).toHaveCount(40);
    const firstBar = first.histogram.bars[0];
    await expect(figure.locator('.distribution-bar').first().locator('title')).toHaveText(
        `Training values ${firstBar.lower} to ${firstBar.upper}: ${firstBar.rows} rows`,
    );
    const beforeCuts = await cutLines(panel);
    expect(beforeCuts.length).toBeGreaterThan(5);

    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await expect(figure).toHaveCount(0);
    const secondResponse = page.waitForResponse('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for Mileage' }).click();
    const second = await (await secondResponse).json();
    expect(second.histogram).toEqual(first.histogram);
    expect(second.actual_bins).toBeLessThan(first.actual_bins);
    await expect(figure.locator('.distribution-bar')).toHaveCount(40);
    expect(await cutLines(panel)).not.toEqual(beforeCuts);

    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 2, 4');
    const thirdResponse = page.waitForResponse('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for VehicleAge' }).click();
    const third = await (await thirdResponse).json();
    expect(third.histogram.finite_rows).toBeLessThan(third.training_rows);
    expect(await cutLines(panel)).toEqual([0, 2, 4]);
    await expect(figure.locator('.axis-label')).toHaveText('Rows');
    await expect(figure.locator('.value-tick').first()).toHaveAttribute('text-anchor', 'start');
    await expect(figure.locator('.value-tick').last()).toHaveAttribute('text-anchor', 'end');
    const emptyBar = third.histogram.bars.findIndex((bar) => bar.rows === 0);
    expect(emptyBar).toBeGreaterThanOrEqual(0);
    await expect(figure.locator('.distribution-bar').nth(emptyBar)).toHaveAttribute('height', '0');
    await expect(panel.getByText('Bin counts and intervals')).toBeVisible();
    await panel.getByText('Bin counts and intervals').click();
    await expect(panel.getByRole('table')).toBeVisible();

    await page.getByLabel('Custom cuts for VehicleAge').fill('-10, 0, 2, 4, 10');
    await expect(figure).toHaveCount(0);
    const outsideResponse = page.waitForResponse('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for VehicleAge' }).click();
    const outside = await (await outsideResponse).json();
    expect(outside.histogram).toEqual(third.histogram);
    expect(await cutLines(panel)).toEqual([0, 2, 4]);
    await expect(figure).toContainText('2 model boundaries outside the range are omitted.');

    await page.getByLabel('Type for VehicleAge').selectOption('categorical');
    await expect(figure).toHaveCount(0);
    const inactiveResponse = page.waitForResponse('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for VehicleAge' }).click();
    const inactive = await (await inactiveResponse).json();
    expect(inactive.active).toBe(false);
    await expect(panel).toContainText('Saved numeric setting is inactive');
    await expect(figure).toHaveCount(0);
});

test('a delayed preview cannot overwrite a newer draft', async ({ page }) => {
    await page.goto('/');
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 1, 2');
    let release;
    const gate = new Promise((resolve) => {
        release = resolve;
    });
    await page.route('**/api/variables/binning-preview', async (route) => {
        const response = await route.fetch();
        await gate;
        await route.fulfill({ response });
    });
    const request = page.waitForRequest('**/api/variables/binning-preview');
    await page.getByRole('button', { name: 'Preview bins for VehicleAge', exact: true }).click();
    await request;
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 2, 4');
    await expect(panel.locator('figure.binning-histogram')).toHaveCount(0);
    const response = page.waitForResponse('**/api/variables/binning-preview');
    release();
    await response;
    await expect(panel.locator('.binning-preview')).toHaveCount(0);
    await expect(panel.locator('figure.binning-histogram')).toHaveCount(0);
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
});
