import { test, expect } from '@playwright/test';

function displayedCuts(panel) {
    return panel.locator('.distribution-bar').evaluateAll((bars) =>
        [
            ...new Set(
                bars.flatMap((bar) =>
                    ['data-lower', 'data-upper']
                        .map((key) => bar.getAttribute(key))
                        .filter((value) => value != null && value !== '')
                        .map(Number),
                ),
            ),
        ].sort((a, b) => a - b),
    );
}

function previewResponse(page, column, accepts = () => true) {
    return page.waitForResponse((response) => {
        if (!response.url().endsWith('/api/variables/binning-preview')) return false;
        const request = response.request().postDataJSON();
        return request.column === column && accepts(request);
    });
}

test('numeric binning stays in sync and reaches fitted rate tables', async ({ page }, testInfo) => {
    const errors = [];
    let previews = 0;
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => {
        if (request.url().endsWith('/api/variables/binning-preview')) previews++;
    });
    const firstPreview = previewResponse(page, 'DriverAge');
    await page.goto('/');
    const button = (name) => page.getByRole('button', { name, exact: true });
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    await expect(panel).toBeVisible();
    await firstPreview;
    await expect(panel).toContainText('Training preview · DriverAge');
    await expect(panel.getByRole('button', { name: /Preview bins/ })).toHaveCount(0);
    expect(previews).toBeGreaterThan(0);
    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    const cuts = page.getByLabel('Custom cuts for VehicleAge');
    const responsePromise = previewResponse(
        page,
        'VehicleAge',
        (request) => request.setup.binning.overrides.VehicleAge?.cuts?.join(',') === '0,1,2,3,4,5',
    );
    await cuts.fill('0, 1, 2, 3, 4, 5');
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

test('distribution bars follow the selected model bins and their exact row counts', async ({
    page,
}) => {
    await page.goto('/');
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    const firstResponse = previewResponse(page, 'Mileage');
    await panel.getByRole('button', { name: /^Mileage/ }).click();
    const first = await (await firstResponse).json();
    expect(first.rows.reduce((total, bar) => total + bar.rows, 0)).toBe(1800);
    const figure = panel.locator('figure.binning-histogram');
    await expect(figure).toContainText('One bar per model bin');
    await expect(figure.locator('.distribution-bar')).toHaveCount(first.actual_bins);
    await expect(figure.locator('.model-cut')).toHaveCount(0);
    const firstBar = first.rows[0];
    await expect(figure.locator('.distribution-bar').first().locator('title')).toHaveText(
        `${firstBar.label}: ${firstBar.rows} rows`,
    );
    const beforeCuts = await displayedCuts(panel);
    expect(beforeCuts.length).toBeGreaterThan(5);

    const secondResponse = previewResponse(
        page,
        'Mileage',
        (request) => request.setup.binning.default_bins === 8,
    );
    await page.getByLabel('Default number of bins', { exact: true }).fill('8');
    await expect(figure).toHaveCount(0);
    const second = await (await secondResponse).json();
    expect(second.actual_bins).toBeLessThan(first.actual_bins);
    await expect(figure.locator('.distribution-bar')).toHaveCount(second.actual_bins);
    expect(await displayedCuts(panel)).not.toEqual(beforeCuts);

    const manyBinsResponse = previewResponse(
        page,
        'Mileage',
        (request) => request.setup.binning.default_bins === 50,
    );
    await page.getByLabel('Default number of bins', { exact: true }).fill('50');
    const manyBins = await (await manyBinsResponse).json();
    expect(manyBins.actual_bins).toBe(50);
    await expect(figure.locator('.distribution-bar')).toHaveCount(50);
    for (const width of [867, 390]) {
        await page.setViewportSize({ width, height: 782 });
        await expect
            .poll(() =>
                figure.locator('.chart-container').evaluate((chart) => {
                    const bounds = chart.getBoundingClientRect();
                    return (
                        chart.scrollWidth <= chart.clientWidth + 1 &&
                        Array.from(chart.querySelectorAll('.distribution-bar')).every((bar) => {
                            const rect = bar.getBoundingClientRect();
                            return (
                                rect.width > 0 &&
                                rect.left >= bounds.left &&
                                rect.right <= bounds.right
                            );
                        })
                    );
                }),
            )
            .toBe(true);
        expect(await figure.locator('.value-tick').count()).toBeLessThan(50);
        expect(await figure.locator('.distribution-bar title').count()).toBe(50);
    }
    await page.setViewportSize({ width: 1440, height: 1000 });

    await panel.getByRole('button', { name: /^VehicleAge/ }).click();
    await page.getByLabel('Binning method for VehicleAge').selectOption('cuts');
    const thirdResponse = previewResponse(
        page,
        'VehicleAge',
        (request) => request.setup.binning.overrides.VehicleAge?.cuts?.join(',') === '0,2,4',
    );
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 2, 4');
    const third = await (await thirdResponse).json();
    expect(third.missing_rows).toBeGreaterThan(0);
    await expect(figure.locator('.distribution-bar')).toHaveCount(4);
    expect(await displayedCuts(panel)).toEqual([0, 2, 4]);
    expect(
        await figure
            .locator('.distribution-bar')
            .evaluateAll((bars) => bars.map((bar) => Number(bar.getAttribute('data-rows')))),
    ).toEqual(third.rows.map((row) => row.rows));
    await expect(figure.locator('.value-tick')).toHaveText(third.rows.map((row) => row.label));
    await expect(figure.locator('.axis-label')).toHaveText('Rows');
    await expect(panel.getByText('Bin counts and intervals')).toBeVisible();
    await panel.getByText('Bin counts and intervals').click();
    await expect(panel.getByRole('table')).toBeVisible();

    const outsideResponse = previewResponse(
        page,
        'VehicleAge',
        (request) => request.setup.binning.overrides.VehicleAge?.cuts?.join(',') === '-10,0,2,4,10',
    );
    await page.getByLabel('Custom cuts for VehicleAge').fill('-10, 0, 2, 4, 10');
    await expect(figure).toHaveCount(0);
    const outside = await (await outsideResponse).json();
    await expect(figure.locator('.distribution-bar')).toHaveCount(6);
    expect(await displayedCuts(panel)).toEqual([-10, 0, 2, 4, 10]);
    expect(
        await figure
            .locator('.distribution-bar')
            .evaluateAll((bars) => bars.map((bar) => Number(bar.getAttribute('data-rows')))),
    ).toEqual(outside.rows.map((row) => row.rows));
    const emptyBar = outside.rows.findIndex((bar) => bar.rows === 0);
    expect(emptyBar).toBeGreaterThanOrEqual(0);
    await expect(figure.locator('.distribution-bar').nth(emptyBar)).toHaveAttribute('height', '0');
    await page.setViewportSize({ width: 390, height: 780 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );

    const inactiveResponse = previewResponse(page, 'VehicleAge', (request) =>
        request.setup.types.categorical?.includes('VehicleAge'),
    );
    await page.getByLabel('Type for VehicleAge').selectOption('categorical');
    await expect(figure).toHaveCount(0);
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
    let release;
    const gate = new Promise((resolve) => {
        release = resolve;
    });
    await page.route('**/api/variables/binning-preview', async (route) => {
        const request = route.request().postDataJSON();
        if (
            request.column !== 'VehicleAge' ||
            request.setup.binning.overrides.VehicleAge?.cuts?.join(',') !== '0,1,2'
        ) {
            await route.continue();
            return;
        }
        const response = await route.fetch();
        await gate;
        await route.fulfill({ response });
    });
    const oldRequest = page.waitForRequest(
        (request) =>
            request.url().endsWith('/api/variables/binning-preview') &&
            request.postDataJSON().setup.binning.overrides.VehicleAge?.cuts?.join(',') === '0,1,2',
    );
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 1, 2');
    await oldRequest;
    const newerResponse = previewResponse(
        page,
        'VehicleAge',
        (request) => request.setup.binning.overrides.VehicleAge?.cuts?.join(',') === '0,2,4',
    );
    await page.getByLabel('Custom cuts for VehicleAge').fill('0, 2, 4');
    await expect(panel.locator('figure.binning-histogram')).toHaveCount(0);
    await newerResponse;
    await expect.poll(() => displayedCuts(panel)).toEqual([0, 2, 4]);
    const oldResponse = previewResponse(
        page,
        'VehicleAge',
        (request) => request.setup.binning.overrides.VehicleAge?.cuts?.join(',') === '0,1,2',
    );
    release();
    await oldResponse;
    expect(await displayedCuts(panel)).toEqual([0, 2, 4]);
});

test('automatic previews debounce edits, skip invalid drafts, and stop off the Variables page', async ({
    page,
}) => {
    const requests = [];
    page.on('request', (request) => {
        if (request.url().endsWith('/api/variables/binning-preview'))
            requests.push(request.postDataJSON());
    });
    const initial = previewResponse(page, 'DriverAge');
    await page.goto('/');
    const panel = page.getByRole('region', { name: 'Numeric binning' });
    await initial;
    await expect(panel).toContainText('Training preview · DriverAge');
    await panel.getByRole('button', { name: /^DriverAge/ }).click();
    await expect(panel).toContainText('Training preview · DriverAge');
    const resetPreview = previewResponse(page, 'DriverAge');
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    await resetPreview;
    await expect(panel.locator('figure.binning-histogram')).toBeVisible();

    const defaultBins = page.getByLabel('Default number of bins', { exact: true });
    await defaultBins.fill('');
    await expect(panel.getByRole('alert')).toContainText('Default number of bins');
    await expect(panel.locator('.binning-preview')).toHaveCount(0);
    const invalidCount = requests.length;
    await page.waitForTimeout(450);
    expect(requests.length).toBe(invalidCount);

    const restored = previewResponse(
        page,
        'DriverAge',
        (request) => request.setup.binning.default_bins === 20,
    );
    await defaultBins.fill('20');
    await restored;
    await expect(panel.locator('figure.binning-histogram')).toBeVisible();

    const revised = previewResponse(
        page,
        'DriverAge',
        (request) => request.setup.binning.default_bins === 10,
    );
    const beforeBurst = requests.length;
    await defaultBins.evaluate((input) => {
        for (const value of ['8', '9', '10']) {
            input.value = value;
            input.dispatchEvent(new Event('input', { bubbles: true }));
        }
    });
    await revised;
    await expect(panel.locator('figure.binning-histogram')).toBeVisible();
    await page.waitForTimeout(450);
    expect(
        requests.slice(beforeBurst).map((request) => request.setup.binning.default_bins),
    ).toEqual([10]);

    const beforeNavigation = requests.length;
    await defaultBins.evaluate((input) => {
        input.value = '11';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        [...document.querySelectorAll('button')]
            .find((button) => button.textContent.trim() === 'Model')
            .click();
    });
    await expect(page.getByRole('button', { name: 'Fit model' })).toBeVisible();
    await page.waitForTimeout(450);
    expect(requests.length).toBe(beforeNavigation);
    const resumed = previewResponse(
        page,
        'DriverAge',
        (request) => request.setup.binning.default_bins === 11,
    );
    await page.getByRole('button', { name: /^Variables\s*\d*$/ }).click();
    await resumed;
    await expect(panel.locator('figure.binning-histogram')).toBeVisible();

    const beforeDestroy = requests.length;
    await defaultBins.evaluate((input) => {
        input.value = '12';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        [...document.querySelectorAll('button')]
            .find((button) => button.textContent.trim() === 'Variables JSON')
            .click();
    });
    await expect(page.getByLabel('Variables JSON', { exact: true })).toBeVisible();
    await page.waitForTimeout(450);
    expect(requests.length).toBe(beforeDestroy);
});
