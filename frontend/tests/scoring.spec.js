import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { execFile as execFileCallback } from 'node:child_process';
import { resolve } from 'node:path';
import { promisify } from 'node:util';

const steps = (page) => page.getByRole('navigation', { name: 'Project setup steps' });
const execFile = promisify(execFileCallback);

test('separate CSV data is checked, previewed and downloaded without changing training data', async ({ page }) => {
    await page.goto('/');
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'Renewal extract' });
    await expect(page.getByRole('heading', { name: 'Score on new data', exact: true })).toBeVisible();
    await expect(page.getByText(/Score Renewal extract using .*Vehicle file/)).toBeVisible();
    await expect(page.getByText('Required columns: all present')).toBeVisible();
    await expect(page.getByText('Check values and keys before scoring.')).toBeVisible();
    await page.getByLabel('Prediction basis', { exact: true }).selectOption({ index: 1 });
    await page.getByText('Advanced cumulative checkpoints', { exact: true }).click();
    await page.getByLabel('Advanced cumulative checkpoint', { exact: true }).selectOption({ label: 'Through Vehicle file' });
    await expect(page.getByText(/Advanced checkpoint selected: Through Vehicle file/)).toBeVisible();
    await page.getByRole('radio', { name: /Overall prediction/ }).check();
    await expect(page.getByLabel('Advanced cumulative checkpoint', { exact: true })).toHaveValue('');
    await expect(page.getByLabel('Prediction basis', { exact: true })).toHaveValue('');
    await page.getByText('Advanced cumulative checkpoints', { exact: true }).click();
    await page.getByRole('button', { name: 'Check inputs', exact: true }).click();
    await expect(page.getByText(/Ready to score 4 rows/)).toBeVisible();
    await page.screenshot({ path: '/private/tmp/easyglm-scoring-overall-final.png', fullPage: true });
    await page.getByRole('button', { name: 'Preview first 20 rows', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Score preview', exact: true })).toBeVisible();
    const downloaded = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download scored data', exact: true }).click();
    const download = await downloaded;
    expect(download.suggestedFilename()).toMatch(/\.csv$/);
    const details = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download score details', exact: true }).click();
    expect((await details).suggestedFilename()).toMatch(/\.json$/);
});

test('missing predictors are a preflight failure and narrow scoring layout has no page overflow', async ({ page }) => {
    await page.goto('/');
    await steps(page).getByRole('button', { name: /Data/ }).click();
    await page.getByLabel('Scoring dataset name').fill('No rating inputs');
    await page.getByLabel('Scoring dataset file').setInputFiles({
        name: 'missing.csv', mimeType: 'text/csv', buffer: Buffer.from('PolicyID\nnew-policy\n'),
    });
    await page.getByRole('button', { name: 'Add scoring dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'No rating inputs' });
    await expect(page.getByText('Missing required columns')).toBeVisible();
    await expect(page.locator('.structural-check').getByText(/Region/)).toBeVisible();
    await page.getByRole('button', { name: 'Check inputs', exact: true }).click();
    await expect(page.getByText('Cannot score this dataset yet')).toBeVisible();
    await expect(page.getByRole('button', { name: 'Preview first 20 rows', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Download scored data', exact: true })).toHaveCount(0);
    await page.setViewportSize({ width: 797, height: 792 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.screenshot({ path: '/private/tmp/easyglm-scoring-797.png', fullPage: true });
});

test('a key-only vehicle extract scores an accepted component and downloads matching Parquet metadata', async ({ page }, testInfo) => {
    await page.goto('/');
    await steps(page).getByRole('button', { name: /Data/ }).click();
    await page.getByLabel('Scoring dataset name').fill('Vehicle key extract');
    await page.getByLabel('Scoring dataset file').setInputFiles({
        name: 'vehicle-keys.csv', mimeType: 'text/csv', buffer: Buffer.from('VehicleCode\n0001\n'),
    });
    await page.getByRole('button', { name: 'Add scoring dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'Vehicle key extract' });
    await page.getByRole('radio', { name: /One rating model/ }).check();
    await page.getByLabel('Rating model', { exact: true }).selectOption({ label: 'Vehicle file' });
    await expect(page.getByText(/Score Vehicle key extract using Vehicle file/)).toBeVisible();
    await page.getByRole('button', { name: 'Check inputs', exact: true }).click();
    await expect(page.getByText(/Ready to score 1 rows/)).toBeVisible();
    await page.screenshot({ path: '/private/tmp/easyglm-scoring-vehicle-final.png', fullPage: true });
    const previewResponse = page.waitForResponse((response) => response.url().includes('/api/scoring/preview') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Preview first 20 rows', exact: true }).click();
    const preview = await (await previewResponse).json();
    await page.locator('.score-preview input[type="radio"][value="parquet"]').check();
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download scored data', exact: true }).click();
    const parquetDownload = await download;
    expect(parquetDownload.suggestedFilename()).toMatch(/\.parquet$/);
    const parquetPath = testInfo.outputPath('vehicle-score.parquet');
    await parquetDownload.saveAs(parquetPath);
    const parquetBytes = await readFile(parquetPath);
    expect(parquetBytes.subarray(0, 4).toString()).toBe('PAR1');
    expect(parquetBytes.subarray(-4).toString()).toBe('PAR1');

    const scoreColumn = preview.columns.find((column) => column.startsWith('easyglm_'));
    expect(scoreColumn).toBeTruthy();
    const { stdout } = await execFile(resolve(process.cwd(), '../.venv/bin/python'), ['-c', "import json, polars as pl, sys; print(json.dumps(pl.read_parquet(sys.argv[1]).row(0, named=True)))", parquetPath]);
    const exported = JSON.parse(stdout);
    expect(exported[scoreColumn]).toBeCloseTo(preview.rows[0][scoreColumn]);

    const details = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download score details', exact: true }).click();
    const detailsDownload = await details;
    expect(detailsDownload.suggestedFilename()).toMatch(/\.json$/);
    const detailsPath = testInfo.outputPath('vehicle-score-details.json');
    await detailsDownload.saveAs(detailsPath);
    const metadata = JSON.parse(await readFile(detailsPath, 'utf8'));
    expect(metadata.dataset.name).toBe('Vehicle key extract');
    expect(metadata.selected_target.saved_version).toBeTruthy();
    expect(metadata.selected_target.label).toBe('Vehicle file');
    expect(metadata.output.request.component_ids).toHaveLength(1);
});

test('a refreshed vehicle master is a distinct candidate output', async ({ page }) => {
    await page.goto('/');
    await steps(page).getByRole('button', { name: /Data/ }).click();
    await page.getByLabel('Scoring dataset name').fill('Refreshed vehicle master');
    await page.getByLabel('Scoring dataset file').setInputFiles({
        name: 'vehicles.csv', mimeType: 'text/csv', buffer: Buffer.from('VehicleCode,Make,Power\nNEW001,New make,120\n'),
    });
    await page.getByRole('button', { name: 'Add scoring dataset', exact: true }).click();
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'Refreshed vehicle master' });
    await page.getByRole('radio', { name: /One rating model/ }).check();
    await page.getByLabel('Rating model', { exact: true }).selectOption({ label: 'Vehicle file' });
    await page.getByRole('button', { name: /Score refreshed catalogue master: Vehicle file catalogue candidate/ }).click();
    await expect(page.getByText('Candidate catalogue output selected.')).toBeVisible();
    await page.getByRole('button', { name: 'Check inputs', exact: true }).click();
    await expect(page.getByText(/Ready to score 1 rows/)).toBeVisible();
    await page.getByRole('button', { name: 'Preview first 20 rows', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Candidate catalogue output', exact: true })).toBeVisible();
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download scored data', exact: true }).click();
    expect((await download).suggestedFilename()).toMatch(/\.csv$/);
});

test('a delayed preflight cannot replace a changed scope or reset selection', async ({ page }) => {
    let releaseFirstCheck;
    let delayed = false;
    await page.route('**/api/scoring/check', async (route) => {
        if (!delayed) {
            delayed = true;
            await new Promise((resolve) => { releaseFirstCheck = resolve; });
        }
        await route.continue();
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'Renewal extract' });
    await page.getByRole('button', { name: 'Check inputs', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Checking inputs…', exact: true })).toBeVisible();
    await page.getByRole('radio', { name: /One rating model/ }).check();
    await page.getByLabel('Scoring dataset', { exact: true }).selectOption({ label: 'Missing rating factor' });
    releaseFirstCheck();
    await expect(page.getByText(/Ready to score 4 rows/)).toHaveCount(0);
    await expect(page.getByText('Cannot score this dataset yet')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Preview first 20 rows', exact: true })).toHaveCount(0);
});

test('same-minute saved rating models are distinguished by their human model names', async ({ page }) => {
    const target = (id, createdAt, modelName = 'Vehicle pricing') => ({
        id, purpose: 'rating', label: 'Vehicle file', model_name: modelName, saved_version: id,
        created_at: createdAt, included_models: ['Vehicle file'], output_kind: 'multiplier', output_label: 'Rating multiplier', expected_label: 'Expected multiplier', required_columns: ['VehicleCode'],
        request: { model_id: id, scope: 'model', component_ids: ['vehicle'], through: null }, exposure_column: null, allowed_exposure_columns: [], candidate_for: null, native_available: false, external_offset_column: null,
    });
    await page.route('**/api/scoring/models', (route) => route.fulfill({ json: {
        targets: [
            { ...target('fit-overall', '2026-10-10T08:45:00Z'), id: 'fit-overall/overall', purpose: 'overall', label: 'Complete accepted score', included_models: ['Main effects', 'Vehicle file'], request: { model_id: 'fit-overall', scope: 'full', component_ids: [], through: null } },
            target('fit-old/vehicle', '2026-10-10T08:45:00Z', 'Vehicle baseline'),
            target('fit-new/vehicle', '2026-10-10T08:45:30Z', 'Vehicle refresh'),
        ], session_id: 'test', revision: 1,
    } }));
    await page.goto('/');
    await page.getByRole('button', { name: 'Score on new data', exact: true }).click();
    await page.getByRole('radio', { name: /One rating model/ }).check();
    const labels = await page.getByLabel('Rating model', { exact: true }).locator('option').allTextContents();
    expect(labels).toEqual(expect.arrayContaining([expect.stringMatching(/Vehicle file · saved as Vehicle baseline/)]));
    expect(new Set(labels).size).toBe(2);
    expect(labels).toEqual(expect.arrayContaining([expect.stringMatching(/Vehicle refresh/)]));
    await expect(page.locator('main')).not.toContainText('fit-old');
    await expect(page.locator('main')).not.toContainText('fit-new');
});
