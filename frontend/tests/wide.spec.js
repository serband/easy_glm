import { test, expect } from '@playwright/test';
import fs from 'node:fs';

async function startSingleModel(page) {
    const saved = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/model-plan') && response.request().method() === 'POST',
    );
    await page.getByRole('button', { name: 'Single model', exact: true }).dispatchEvent('click');
    expect((await saved).ok()).toBe(true);
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible();
}

test('2000-column virtualization and local interaction timings', async ({ page }, testInfo) => {
    const requests = [],
        errors = [];
    page.on('request', (r) => requests.push(r.url()));
    page.on('pageerror', (e) => errors.push(e.message));
    const navigationStart = performance.now();
    await page.goto('/');
    await startSingleModel(page);
    await expect(page.locator('.data-row').first()).toBeVisible();
    const navigationReadyMs = performance.now() - navigationStart;
    expect(await page.locator('.data-row').count()).toBeLessThanOrEqual(28);
    const before = requests.length;
    const edits = [],
        switches = [],
        resets = [];
    for (let i = 0; i < 12; i++) {
        edits.push(
            await page.evaluate(async (i) => {
                const select = document.querySelector('[aria-label="Role for variable_0000"]');
                const start = performance.now();
                select.value = i % 2 ? 'unassigned' : 'predictor';
                select.dispatchEvent(new Event('change', { bubbles: true }));
                await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
                return performance.now() - start;
            }, i),
        );
        switches.push(
            await page.evaluate(async () => {
                const start = performance.now();
                [...document.querySelectorAll('button')]
                    .find((b) => b.textContent === 'Variables JSON')
                    .click();
                await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
                return performance.now() - start;
            }),
        );
        resets.push(
            await page.evaluate(async () => {
                const start = performance.now();
                [...document.querySelectorAll('button')]
                    .find((b) => b.textContent === 'Reset')
                    .click();
                await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
                return performance.now() - start;
            }),
        );
        await page.getByRole('button', { name: 'Table', exact: true }).click();
    }
    expect(requests.length).toBe(before);
    await page.getByLabel('Search variables', { exact: true }).fill('variable_1999');
    await expect(page.getByLabel('Role for variable_1999', { exact: true })).toHaveValue(
        'unassigned',
    );
    expect(await page.locator('.data-row').count()).toBe(1);
    await page.getByLabel('Role for variable_1999', { exact: true }).selectOption('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    await page.reload();
    await page.getByLabel('Search variables', { exact: true }).fill('variable_1999');
    await expect(page.getByLabel('Role for variable_1999', { exact: true })).toHaveValue('ignore');
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const bulk = JSON.parse(await page.getByLabel('Variables JSON', { exact: true }).inputValue());
    bulk.ignore = [...bulk.ignore, ...bulk.unassigned];
    bulk.unassigned = [];
    await page.getByLabel('Variables JSON', { exact: true }).fill(JSON.stringify(bulk));
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByText('1999 variable changes', { exact: true })).toBeVisible();
    expect(await page.locator('.preview-change').count()).toBeLessThanOrEqual(18);
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    const summary = (values) => ({
        median: values.toSorted((a, b) => a - b)[Math.floor(values.length / 2)],
        max: Math.max(...values),
        samples: values,
    });
    const result = {
        navigationReadyMs,
        editToPaintMs: summary(edits),
        tableToJsonToPaintMs: summary(switches),
        resetToPaintMs: summary(resets),
        maxRenderedRows: 28,
        rows: 2000,
        columns: 2000,
    };
    fs.writeFileSync(testInfo.outputPath('timings.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
    expect(errors).toEqual([]);
    expect(Math.max(...edits, ...switches, ...resets)).toBeLessThan(250);
});

test('per-model numeric bands stay local and searchable with 2000 numeric columns', async ({ page }) => {
    let binningRequests = 0;
    page.on('request', (request) => {
        if (request.url().includes('/api/variables/binning-preview')) binningRequests++;
    });
    await page.goto('/');
    await startSingleModel(page);
    await page.getByLabel('Search variables', { exact: true }).fill('variable_1999');
    await page.getByLabel('Role for variable_1999', { exact: true }).selectOption('predictor');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    await page.getByLabel('Search model variables').fill('variable_1999');
    await page.getByLabel('Use variable_1999').check();
    await page.getByRole('button', { name: 'Continue to model details', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to variable treatment', exact: true }).click();
    await page.getByLabel('Search model variables').fill('variable_1999');
    await expect(page.getByLabel('Use variable_1999')).toBeChecked();
    await expect(page.locator('.variables-scroll tbody tr')).toHaveCount(1);
    await page.getByLabel('Bands for variable_1999', { exact: true }).fill('12');
    expect(binningRequests).toBe(0);
});
