import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

test.use({ viewport: { width: 919, height: 773 } });
test('downloads applied artifacts and excludes unsaved adjustment previews', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    async function download(name, suffix) {
        const [file] = await Promise.all([page.waitForEvent('download'), button(name).click()]);
        expect(file.suggestedFilename().endsWith(suffix)).toBe(true);
        const path = testInfo.outputPath(file.suggestedFilename());
        await file.saveAs(path);
        expect(await file.failure()).toBeNull();
        return readFile(path);
    }
    await page.goto('/');
    await button('Export').click();
    await expect(
        page.getByText('Fit a model to export its rate tables, scorer and report.', {
            exact: true,
        }),
    ).toBeVisible();
    expect(
        JSON.parse((await download('Download project JSON', '.easyglm-project.json')).toString())
            .models,
    ).toEqual({});
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((node) => node.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    await button('Rate tables').click();
    await page.getByLabel('Adjustment method', { exact: true }).selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('2.4');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Apply adjustment').click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    await page.getByLabel('Adjustment method', { exact: true }).selectOption('manual');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('3.7');
    await expect(button('Apply adjustment')).toBeEnabled();
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (path) =>
        (await page.request.get('/api/' + path, { headers: { 'x-easyglm-token': token } })).json();
    const before = await get('project');
    const jobs = await get('jobs');
    const history = await get('review-info/Frequency');
    await button('Export').click();
    await expect(page.getByLabel('Export model')).toHaveValue('Frequency');
    const scorer = JSON.parse(
        (await download('Download scorer (.easyglm)', '.easyglm')).toString(),
    );
    expect(scorer.variables.DriverAge.table[1].relativity).toBe(2.4);
    const excel = await download('Download Excel (.xlsx)', '.xlsx');
    expect(excel.subarray(0, 2).toString()).toBe('PK');
    const report = (await download('Download report (.html)', '.html')).toString();
    expect(report).toContain('Frequency');
    expect(report).toContain('<html');
    expect(report).toContain('<h2>2. Data summary</h2>');
    expect(report).toContain('training distribution');
    expect(report).toContain('Excess kurtosis');
    const script = (await download('Download script (.py)', '.py')).toString();
    expect(script).toContain('fit_glm');
    expect(
        JSON.parse((await download('Download project JSON', '.easyglm-project.json')).toString()),
    ).toEqual(before);
    expect(await get('project')).toEqual(before);
    expect(await get('jobs')).toEqual(jobs);
    expect(await get('review-info/Frequency')).toEqual(history);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    await page.screenshot({ path: testInfo.outputPath('exports-919.png'), fullPage: true });
    await page.route('**/api/exports/Frequency', (route) =>
        route.fulfill({
            status: 409,
            json: { detail: 'The applied model changed. Download again.' },
        }),
    );
    await button('Download scorer (.easyglm)').click();
    await expect(page.getByRole('alert')).toHaveText('The applied model changed. Download again.');
    expect(errors).toEqual([]);
});
