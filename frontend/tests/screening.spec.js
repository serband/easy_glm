import { test, expect } from '@playwright/test';
const button = (page, name) => page.getByRole('button', { name, exact: true });
const panel = (page) => page.getByRole('region', { name: 'Predictor screening', exact: true });
async function session(page) {
    const { token } = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': token };
    return {
        headers,
        snapshot: await (await page.request.get('/api/variables', { headers })).json(),
    };
}
async function mockScan(page, { status = 'complete', holdStart = false, holdPoll = false } = {}) {
    const { snapshot } = await session(page);
    const calls = { starts: [], polls: 0, cancellations: 0, releaseStart: null, releasePoll: null };
    let packet;
    await page.route('**/api/variables/screen', async (route) => {
        const body = route.request().postDataJSON();
        calls.starts.push(body);
        const display = (raw) => body.setup.renames[raw] || raw;
        const variable = (raw, association) => ({
            variable: display(raw),
            raw_name: raw,
            association,
            method: 'Weighted Pearson',
            reason: 'Very strong target association',
            observations: 1400,
        });
        packet = {
            id: 'scan-test',
            session_id: body.session_id,
            project_id: snapshot.project_id,
            revision: body.revision,
            fingerprint: 'test-fingerprint',
            status,
            progress: { phase: 'pairs', completed: 1, total: 4 },
            message: 'Checking pairs…',
            elapsed: 0.1,
            result: {
                target: 'Claims',
                weight: 'Exposure',
                divide_target_by_weight: true,
                rows: 1400,
                training_rows: 1400,
                predictor_count: 4,
                excluded_target_rows: 2,
                leakage: [
                    variable('DriverAge', 0.99),
                    variable('VehicleAge', 0.98),
                    variable('Claims', 1),
                    {
                        variable: 'Derived only',
                        raw_name: null,
                        association: 0.99,
                        method: 'Grouped prediction',
                        reason: 'No source column',
                        observations: 1400,
                    },
                ],
                correlated: [
                    {
                        first: display('DriverAge'),
                        second: display('VehicleAge'),
                        first_raw: 'DriverAge',
                        second_raw: 'VehicleAge',
                        association: 1,
                        method: 'Weighted Pearson',
                        observations: 1400,
                    },
                ],
                correlated_count: 2001,
                missing: [
                    {
                        variable: 'Region',
                        raw_name: 'Region',
                        missing_share: 0.8,
                        observations: 280,
                    },
                ],
                unsupported: [
                    { variable: 'Derived only', raw_name: null, reason: 'Too many levels.' },
                ],
                notes: ['Pair results are limited to the strongest matches.'],
                columns: [],
            },
        };
        if (holdStart)
            await new Promise((resolve) => {
                calls.releaseStart = resolve;
            });
        await route.fulfill({
            status: 202,
            json: status === 'complete' ? packet : { ...packet, result: undefined },
        });
    });
    await page.route('**/api/screenings/scan-test', async (route) => {
        calls.polls++;
        if (holdPoll)
            await new Promise((resolve) => {
                calls.releasePoll = resolve;
            });
        await route.fulfill({ json: { ...packet, status: 'complete' } });
    });
    await page.route('**/api/screenings/scan-test/cancel', (route) => {
        calls.cancellations++;
        return route.fulfill({ json: { ...packet, status: 'cancelled', result: undefined } });
    });
    return calls;
}
test.beforeEach(async ({ page }) => {
    await page.goto('/');
    await expect(page.getByLabel('Role for Claims', { exact: true })).toHaveValue('target');
});
test('validated full draft is screened; explicit bulk removal synchronises JSON and previews once', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const calls = await mockScan(page);
    let previews = 0,
        applies = 0;
    page.on('request', (r) => {
        if (r.url().endsWith('/api/variables/preview')) previews++;
        if (r.url().endsWith('/api/variables/apply')) applies++;
    });
    await page.getByLabel('Name for DriverAge', { exact: true }).fill('Age');
    await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await page.getByLabel('Type for DriverAge', { exact: true }).selectOption('categorical');
    await button(page, 'Role JSON').click();
    const editor = page.getByLabel('Role JSON', { exact: true }),
        raw = await editor.inputValue();
    await editor.fill(raw + '\n');
    await button(page, 'Check selected predictors').click();
    await expect(panel(page)).toContainText('1,400 of 1,400 training rows');
    expect(calls.starts[0].setup.renames.DriverAge).toBe('Age');
    expect(calls.starts[0].setup.types.categorical).toContain('DriverAge');
    expect(calls.starts[0].options).toEqual({
        sample_rows: 10000,
        seed: 42,
        missing_threshold: 0.7,
        correlation_threshold: 0.95,
        leakage_threshold: 0.9,
        divide_target_by_weight: true,
    });
    await expect(editor).toHaveValue(raw + '\n');
    await expect(button(page, 'Remove selected predictors (0)')).toBeDisabled();
    await expect(page.getByLabel('Select Claims in possible target leakage')).toBeDisabled();
    await expect(page.getByLabel('Select Derived only in possible target leakage')).toBeDisabled();
    await expect(panel(page)).toContainText('280 observed rows');
    await expect(panel(page)).toContainText('Showing the strongest 1 of 2,001 flagged pairs.');
    await button(page, 'Select all possible target leakage').click();
    await expect(button(page, 'Remove selected predictors (2)')).toBeEnabled();
    await expect(panel(page)).toContainText('Both predictors are selected in 1 related pair');
    await expect(page.getByLabel('Removal choice for Age and VehicleAge')).toHaveValue('__both__');
    await page.getByLabel('Search screening results').fill('Region');
    await expect(page.getByLabel('Select Age in possible target leakage')).toHaveCount(0);
    await page.getByLabel('Search screening results').fill('');
    await page.getByLabel('Removal choice for Age and VehicleAge').selectOption('second');
    await button(page, 'Select all mostly missing').click();
    await expect(button(page, 'Remove selected predictors (2)')).toBeEnabled();
    await panel(page).screenshot({ path: testInfo.outputPath('screening-919.png') });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    await button(page, 'Remove selected predictors (2)').click();
    await expect(button(page, 'Apply changes')).toBeEnabled();
    expect(previews).toBe(1);
    expect(applies).toBe(0);
    const draft = JSON.parse(await editor.inputValue());
    expect(draft.ignore).toEqual(expect.arrayContaining(['VehicleAge', 'Region']));
    expect(draft.predictor).toContain('DriverAge');
    await button(page, 'Table').click();
    await expect(page.getByLabel('Role for Region', { exact: true })).toHaveValue('ignore');
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('Age');
    expect(errors).toEqual([]);
});
test('invalid raw role JSON is preserved and never submitted for screening', async ({ page }) => {
    const calls = await mockScan(page);
    await button(page, 'Role JSON').click();
    await page.getByLabel('Role JSON', { exact: true }).fill('{unfinished');
    await button(page, 'Check selected predictors').click();
    await expect(page.getByRole('alert')).toBeVisible();
    await expect(page.getByLabel('Role JSON', { exact: true })).toHaveValue('{unfinished');
    expect(calls.starts).toHaveLength(0);
});
for (const change of ['name', 'type', 'role', 'raw JSON', 'options', 'navigation']) {
    test(`late scan start is discarded and cancelled after a ${change} change`, async ({
        page,
    }) => {
        const calls = await mockScan(page, { status: 'running', holdStart: true });
        await button(page, 'Check selected predictors').click();
        await expect.poll(() => Boolean(calls.releaseStart)).toBe(true);
        if (change === 'name') {
            await page.getByLabel('Name for DriverAge', { exact: true }).fill('DraftAge');
            await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
        } else if (change === 'type')
            await page
                .getByLabel('Type for DriverAge', { exact: true })
                .selectOption('categorical');
        else if (change === 'role')
            await page.getByLabel('Role for DriverAge', { exact: true }).selectOption('ignore');
        else if (change === 'raw JSON') {
            await button(page, 'Role JSON').click();
            await page.getByLabel('Role JSON', { exact: true }).fill('{unfinished');
        } else if (change === 'options') {
            await panel(page).locator('.screening-settings > summary').click();
            await page.getByLabel('Screening sample rows').fill('5000');
        } else await button(page, 'Project & data').click();
        calls.releaseStart();
        await expect.poll(() => calls.cancellations).toBe(1);
        if (change === 'navigation') await page.getByRole('button', { name: /^Variables/ }).click();
        await expect(
            page.getByRole('region', { name: 'Possible target leakage', exact: true }),
        ).toHaveCount(0);
        expect(calls.polls).toBe(0);
    });
}
test('cancel prevents a late polled report from becoming removable', async ({ page }) => {
    const calls = await mockScan(page, { status: 'running', holdPoll: true });
    await button(page, 'Check selected predictors').click();
    await expect.poll(() => Boolean(calls.releasePoll)).toBe(true);
    await button(page, 'Cancel scan').click();
    await expect.poll(() => calls.cancellations).toBe(1);
    calls.releasePoll();
    await expect(panel(page)).toContainText('Scan cancelled.');
    await expect(
        page.getByRole('region', { name: 'Possible target leakage', exact: true }),
    ).toHaveCount(0);
});
test('actual training scan is read-only; applying flagged removals cleans model predictors and interactions', async ({
    page,
}) => {
    await button(page, 'Model').click();
    await page.getByLabel('Interaction first factor').selectOption('DriverAge');
    await page.getByLabel('Interaction second factor').selectOption('AnnualMileage');
    await button(page, 'Add interaction').click();
    await button(page, 'Create model').click();
    await expect(
        page.getByText('Model settings saved. Fit when ready.', { exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: /^Variables/ }).click();
    const { headers } = await session(page);
    const readProject = async () => (await page.request.get('/api/project', { headers })).json();
    const before = await readProject();
    await button(page, 'Check selected predictors').click();
    await expect(page.getByLabel('Select AnnualMileage in possible target leakage')).toBeVisible({
        timeout: 20000,
    });
    await expect(page.getByLabel('Select Region in mostly missing')).toBeVisible();
    expect(await readProject()).toEqual(before);
    expect(await (await page.request.get('/api/jobs', { headers })).json()).toEqual({});
    await page.getByLabel('Select AnnualMileage in possible target leakage').check();
    await page.getByLabel('Select Region in mostly missing').check();
    await button(page, 'Remove selected predictors (2)').click();
    await expect(button(page, 'Apply changes')).toBeEnabled();
    expect(await readProject()).toEqual(before);
    await button(page, 'Apply changes').click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    const after = await readProject();
    expect(after.data.roles.AnnualMileage).toBe('ignore');
    expect(after.data.roles.Region).toBe('ignore');
    expect(after.models.Frequency.predictors).not.toContain('AnnualMileage');
    expect(after.models.Frequency.predictors).not.toContain('Region');
    expect(after.models.Frequency.interactions).toEqual([]);
    expect(await (await page.request.get('/api/jobs', { headers })).json()).toEqual({});
});
