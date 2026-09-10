import { test, expect } from '@playwright/test';
import { mkdtemp, writeFile, rm, readFile, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

async function snapshot(page, endpoint = 'variables') {
    const session = await (await page.request.get('/api/session')).json();
    const response = await page.request.get('/api/' + endpoint, {
        headers: { 'X-EasyGLM-Token': session.token },
    });
    expect(response.ok()).toBe(true);
    return response.json();
}

test('opening data is visible, explicit, and supports files, projects and both examples', async ({
    page,
}, info) => {
    const folder = await realpath(await mkdtemp(join(tmpdir(), 'easyglm_onboarding_')));
    const csv = join(folder, 'portfolio.csv');
    await writeFile(csv, 'Claims,Exposure,Age\n0,1,20\n1,0.5,30\n0,1,40\n');
    const errors = [];
    const openings = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (r.method() === 'POST' && /\/api\/project\/(open|upload)/.test(r.url()))
            openings.push(r);
    });
    const button = (name) => page.getByRole('button', { name, exact: true });
    const nav = (name) =>
        page.getByRole('navigation', { name: 'Workbench' }).getByRole('button', {
            name: name === 'Variables' ? /^Variables/ : name,
            exact: name !== 'Variables',
        });
    const replace = () => page.getByLabel('Replace this session.', { exact: false }).check();
    try {
        await page.setViewportSize({ width: 884, height: 773 });
        await page.goto('/');
        await expect(
            page.getByRole('heading', { name: 'Project & data', exact: true }),
        ).toBeVisible();
        await expect(page.getByRole('heading', { name: 'Open data', exact: true })).toBeVisible();
        await expect(page.getByLabel('Data file', { exact: true })).toBeVisible();
        for (const name of ['Own data', 'Saved project', 'Example dataset']) {
            await expect(page.getByRole('radio', { name, exact: true })).toBeVisible();
        }
        await expect(page.getByLabel('Data format')).toHaveValue('auto');
        await page.screenshot({ path: info.outputPath('empty-start-884.png'), fullPage: true });
        expect(await page.getByLabel('Data format').locator('option').allTextContents()).toEqual([
            'Automatic — from file name',
            'CSV',
            'Parquet',
            'Excel',
            'Arrow / Feather',
            'SAS',
        ]);
        await page.getByRole('radio', { name: 'Example dataset', exact: true }).check();
        await page
            .getByRole('combobox', { name: 'Example dataset', exact: true })
            .selectOption('swedish_motorcycle');
        await expect(page.getByText(/Total claim cost.*including zero claims/)).toBeVisible();
        expect(openings).toHaveLength(0);
        expect((await snapshot(page)).row_count).toBe(0);
        await page.getByRole('radio', { name: 'Own data', exact: true }).check();
        await page.getByLabel('File selection').selectOption('path');
        await page.getByLabel('File path', { exact: true }).fill(join(folder, 'missing.csv'));
        await button('Open data').click();
        await expect(page.locator('.open-error')).toBeVisible();
        expect((await snapshot(page)).row_count).toBe(0);
        await page.getByLabel('File selection').selectOption('upload');
        await page.getByLabel('Data format').selectOption('csv');
        await page.getByLabel('Data file', { exact: true }).setInputFiles(csv);
        expect(openings).toHaveLength(1);
        // Reauthentication in the same session must retain the selected file bytes.
        let uploadAttempts = 0;
        await page.route('**/api/project/upload?**', async (route) => {
            uploadAttempts += 1;
            if (uploadAttempts === 1) {
                await route.fulfill({
                    status: 401,
                    json: { code: 'session_expired', detail: 'Reauthenticate' },
                });
            } else {
                await route.continue();
            }
        });
        await button('Open data').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        expect(uploadAttempts).toBe(2);
        await page.unroute('**/api/project/upload?**');
        const raw = await snapshot(page);
        expect(raw.row_count).toBe(3);
        expect(raw.models).toHaveLength(0);
        expect(raw.columns.map((c) => c.name)).toEqual(['Claims', 'Exposure', 'Age']);
        expect(
            await page
                .getByLabel('Type for Age', { exact: true })
                .locator('option')
                .allTextContents(),
        ).toEqual(['Infer from data', 'Numeric', 'Categorical']);
        await page.getByLabel('Type for Age', { exact: true }).selectOption('categorical');
        await nav('Model').click();
        await expect(page.getByText('No predictors assigned yet.', { exact: true })).toBeVisible();
        await button('Assign predictors in Variables').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        await expect(page.getByLabel('Type for Age', { exact: true })).toHaveValue('categorical');
        expect((await snapshot(page)).models).toHaveLength(0);
        await nav('Project & data').click();
        await expect(page.getByText('portfolio.csv', { exact: true })).toBeVisible();
        const loaded = page.getByRole('region', { name: 'Current dataset' });
        await expect(loaded).toContainText('Loaded:');
        await expect(loaded).toContainText('3 rows');
        await expect(loaded).toContainText('3 columns');
        await page.screenshot({ path: info.outputPath('own-data-884.png'), fullPage: true });

        // Export the uploaded source-backed project, give it a real user name, then reopen by upload.
        await nav('Export').click();
        const [download] = await Promise.all([
            page.waitForEvent('download'),
            button('Download project JSON').click(),
        ]);
        const projectPath = join(folder, 'renewal.easyglm-project.json');
        await download.saveAs(projectPath);
        const project = JSON.parse(await readFile(projectPath, 'utf8'));
        project.name = 'Fleet renewal 2027';
        await writeFile(projectPath, JSON.stringify(project));
        await nav('Project & data').click();
        await page.getByRole('radio', { name: 'Saved project', exact: true }).check();
        await page.getByLabel('Project file', { exact: true }).setInputFiles(projectPath);
        await expect(button('Open project')).toBeDisabled();
        expect((await snapshot(page)).project_id).toBe(raw.project_id);
        await replace();
        await button('Open project').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        expect((await snapshot(page)).name).toBe('Fleet renewal 2027');
        await expect(page.getByLabel('Type for Age', { exact: true })).toHaveValue('auto');
        await nav('Project & data').click();
        await expect(page.getByRole('complementary')).toContainText('Fleet renewal 2027');
        await expect(page.getByText('portfolio.csv', { exact: true })).toBeVisible();

        // The path alternative reads the same saved project; field changes alone never replace it.
        await page.getByLabel('File selection').selectOption('path');
        await page.getByLabel('File path', { exact: true }).fill(projectPath);
        const beforePath = await snapshot(page);
        await expect(button('Open project')).toBeDisabled();
        await replace();
        await button('Open project').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        expect((await snapshot(page)).project_id).not.toBe(beforePath.project_id);

        for (const [id, family, target] of [
            ['french_motor', 'poisson', 'ClaimNb'],
            ['swedish_motorcycle', 'tweedie', 'ClaimAmount'],
        ]) {
            await nav('Project & data').click();
            await page.getByRole('radio', { name: 'Example dataset', exact: true }).check();
            await page
                .getByRole('combobox', { name: 'Example dataset', exact: true })
                .selectOption(id);
            const beforeExample = await snapshot(page);
            const requestCount = openings.length;
            await expect(button('Load example')).toBeDisabled();
            expect(openings).toHaveLength(requestCount);
            expect((await snapshot(page)).project_id).toBe(beforeExample.project_id);
            await replace();
            await button('Load example').click();
            await expect(
                page.getByRole('heading', { name: 'Model design and fit', exact: true }),
            ).toBeVisible();
            await expect(page.getByLabel('Model family', { exact: true })).toHaveValue(family);
            const loaded = await snapshot(page);
            expect(loaded.setup.assignments.target).toBe(target);
            expect(loaded.models).toHaveLength(1);
            expect((await snapshot(page, 'workbench')).jobs).toEqual({});
            await nav('Project & data').click();
            await expect(page.getByRole('region', { name: 'Current dataset' })).toContainText(
                'Loaded:',
            );
            expect(
                await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
            ).toBe(true);
            await page.screenshot({ path: info.outputPath(id + '-884.png'), fullPage: true });
        }
        // Initial-page hints are honoured once, and never trap subsequent sidebar navigation.
        await page.goto('/?view=project');
        await expect(
            page.getByRole('heading', { name: 'Project & data', exact: true }),
        ).toBeVisible();
        await nav('Variables').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        expect(errors).toEqual([]);
    } finally {
        await rm(folder, { recursive: true, force: true });
    }
});

async function openCsv(page, name, contents) {
    await page.goto('/?view=project');
    await page.getByLabel('Data file', { exact: true }).setInputFiles({
        name,
        mimeType: 'text/csv',
        buffer: Buffer.from(contents),
    });
    const replace = page.getByLabel('Replace this session.', { exact: false });
    if (await replace.count()) await replace.check();
    await page.getByRole('button', { name: 'Open data', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
}

for (const failedRead of ['session', 'variables']) {
    test(`successful upload keeps the confirmed new project when ${failedRead} refresh fails`, async ({
        page,
    }, info) => {
        const errors = [];
        page.on('pageerror', (e) => errors.push(e.message));
        await openCsv(page, 'prior.csv', 'OldTarget,OldFactor\n0,20\n1,30\n');
        await page.getByLabel('Type for OldFactor', { exact: true }).selectOption('categorical');
        await page.getByRole('button', { name: 'Project & data', exact: true }).click();
        await page.getByLabel('Data file', { exact: true }).setInputFiles({
            name: 'confirmed.csv',
            mimeType: 'text/csv',
            buffer: Buffer.from('NewTarget,NewFactor\n0,50\n1,60\n0,70\n'),
        });
        await page.getByLabel('Replace this session.', { exact: false }).check();
        let failRefresh = false,
            opened = null,
            openRequests = 0;
        await page.route('**/api/' + failedRead, async (route) => {
            if (failRefresh)
                await route.fulfill({
                    status: 503,
                    json: { detail: 'Connection refresh unavailable' },
                });
            else await route.continue();
        });
        await page.route('**/api/project/upload?**', async (route) => {
            openRequests += 1;
            const response = await route.fetch();
            expect(response.ok()).toBe(true);
            opened = await response.json();
            failRefresh = true;
            await route.fulfill({ response });
        });
        await page.getByRole('button', { name: 'Open data', exact: true }).click();
        const failure = page.locator('.message.error');
        await expect(failure).toContainText('Data loaded. Reconnect to continue.');
        await expect(page.getByLabel('Type for NewFactor', { exact: true })).toHaveValue('auto');
        await expect(page.getByLabel('Type for OldFactor', { exact: true })).toHaveCount(0);
        await expect(page.locator('.portfolio')).toHaveText('confirmed');
        const confirmed = await snapshot(page);
        expect(confirmed.project_id).toBe(opened.project_id);
        expect(confirmed.row_count).toBe(3);
        expect(confirmed.models).toEqual([]);
        expect(openRequests).toBe(1);
        await page.screenshot({
            path: info.outputPath('confirmed-' + failedRead + '-failure.png'),
            fullPage: true,
        });
        failRefresh = false;
        await failure.getByRole('button', { name: 'Reconnect', exact: true }).click();
        await expect(failure).toBeHidden();
        await expect(page.getByLabel('Type for NewFactor', { exact: true })).toHaveValue('auto');
        expect((await snapshot(page)).project_id).toBe(opened.project_id);
        expect(openRequests).toBe(1);
        expect(errors).toEqual([]);
    });
}

test('a stale opener reconnects without losing its draft or replaying Open', async ({
    page,
    context,
}) => {
    await openCsv(page, 'two-tabs.csv', 'Response,Exposure,Rating\n0,1,20\n1,1,30\n');
    await page.getByLabel('Type for Rating', { exact: true }).selectOption('categorical');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByLabel('Data file', { exact: true }).setInputFiles({
        name: 'after-reconnect.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('NewTarget,NewFactor\n0,1\n1,2\n'),
    });
    const before = await snapshot(page);
    const other = await context.newPage();
    try {
        await other.goto('/');
        await other.getByLabel('Role for Response', { exact: true }).selectOption('target');
        await other.getByRole('button', { name: 'Preview changes', exact: true }).click();
        await other.getByRole('button', { name: 'Apply changes', exact: true }).click();
        await expect(other.getByRole('status')).toContainText('Settings applied');
        expect((await snapshot(page)).revision).toBeGreaterThan(before.revision);
        let attempts = 0;
        page.on('request', (r) => {
            if (r.method() === 'POST' && r.url().includes('/api/project/upload?')) attempts += 1;
        });
        await page.getByLabel('Replace this session.', { exact: false }).check();
        const stale = page.waitForResponse((r) => r.url().includes('/api/project/upload?'));
        await page.getByRole('button', { name: 'Open data', exact: true }).click();
        expect((await stale).status()).toBe(409);
        const opener = page.getByRole('region', { name: 'Open data', exact: true });
        await expect(opener.getByRole('button', { name: 'Reconnect', exact: true })).toBeVisible();
        await opener.getByRole('button', { name: 'Reconnect', exact: true }).click();
        await expect(opener.locator('.open-error')).toBeHidden();
        await expect(page.getByLabel('Replace this session.', { exact: false })).not.toBeChecked();
        await expect(page.getByRole('button', { name: 'Open data', exact: true })).toBeDisabled();
        expect(attempts).toBe(1);
        expect((await snapshot(page)).project_id).toBe(before.project_id);
        await page
            .getByRole('navigation', { name: 'Workbench' })
            .getByRole('button', { name: /^Variables/ })
            .click();
        await expect(page.getByLabel('Type for Rating', { exact: true })).toHaveValue(
            'categorical',
        );
        await page.getByRole('button', { name: 'Project & data', exact: true }).click();
        await page.getByLabel('Replace this session.', { exact: false }).check();
        await page.getByRole('button', { name: 'Open data', exact: true }).click();
        await expect(page.getByLabel('Type for NewFactor', { exact: true })).toHaveValue('auto');
        expect((await snapshot(page)).project_id).not.toBe(before.project_id);
        expect(attempts).toBe(2);
    } finally {
        await other.close();
    }
});
