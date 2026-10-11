import { test, expect } from '@playwright/test';
import { execFileSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
const root = path.resolve('..');
const port = Number(process.env.EASYGLM_SESSION_TEST_PORT || '8773');
const base = `http://127.0.0.1:${port}`;
let server;
let fixtureFolder;
let projectPath;
test.beforeAll(async () => {
    fixtureFolder = await mkdtemp(path.join(tmpdir(), 'easyglm-session-test-'));
    projectPath = path.join(fixtureFolder, 'project.json');
    // Load the complete legacy project before accepting browser requests. Seeding
    // it over HTTP after startup lets an open tab reconnect to a partial fixture.
    execFileSync(
        path.join(root, '.venv/bin/python'),
        [
            '-c',
            `
from pathlib import Path
import sys
import numpy as np
import polars as pl
from easy_glm.workflow import DataSource, Project

folder = Path(sys.argv[1])
rng = np.random.default_rng(42)
raw = pl.DataFrame({
    "PolicyID": np.arange(12000),
    "Claims": rng.poisson(0.12, 12000),
    "Exposure": rng.uniform(0.1, 1, 12000),
    "DriverAge": rng.integers(18, 90, 12000),
    "VehicleAge": rng.integers(0, 25, 12000),
    "Region": rng.choice(["North", "South", "East", "West"], 12000),
    "AnnualMileage": rng.gamma(4, 3000, 12000),
    "InternalCode": rng.integers(0, 50, 12000),
})
raw.write_parquet(folder / "data.parquet")
project = Project(name="Motor portfolio · synthetic sample")
project.data.source = DataSource(type="parquet", path=str(folder / "data.parquet"))
project.data.split.mode = "random"
project.data.roles = {
    "PolicyID": "id", "Claims": "target", "Exposure": "weight",
    "DriverAge": "predictor", "VehicleAge": "predictor", "Region": "predictor",
    "AnnualMileage": "predictor", "InternalCode": "ignore",
}
project.new_model("Imported legacy", family="poisson", link="log", predictors=["DriverAge"])
project.to_json(folder / "project.json")
`,
            fixtureFolder,
        ],
        {
            cwd: root,
            env: {
                ...process.env,
                PYTHONPATH: process.env.EASYGLM_TEST_SOURCE || path.join(root, 'src'),
            },
            stdio: 'pipe',
        },
    );
});
test.afterAll(async () => {
    if (fixtureFolder) await rm(fixtureFolder, { recursive: true, force: true });
});
async function start() {
    const id = randomUUID();
    server = spawn(
        path.join(root, '.venv/bin/python'),
        [
            '-m',
            'easy_glm.desktop',
            '--project',
            projectPath,
            '--port',
            String(port),
            '--launch-id',
            id,
        ],
        {
            cwd: root,
            env: {
                ...process.env,
                PYTHONPATH: process.env.EASYGLM_TEST_SOURCE || path.join(root, 'src'),
            },
            stdio: 'ignore',
        },
    );
    await expect
        .poll(
            async () => {
                try {
                    return (await (await fetch(base + '/health')).json()).launch_id;
                } catch {
                    return null;
                }
            },
            { timeout: 30000 },
        )
        .toBe(id);
}
async function stop() {
    if (server && server.exitCode === null) {
        const exited = once(server, 'exit');
        server.kill();
        await exited;
    }
}
async function readProject() {
    const session = await (await fetch(base + '/api/session', { cache: 'no-store' })).json();
    return (
        await fetch(base + '/api/project', { headers: { 'X-EasyGLM-Token': session.token } })
    ).json();
}
async function openSharedColumns(page) {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    const steps = page.getByRole('navigation', { name: 'Project setup steps' });
    await steps.getByRole('button', { name: /Shared columns/ }).click();
    await expect(page.getByRole('region', { name: 'Shared column assignments' })).toBeVisible();
}

test.beforeEach(start);
test.afterEach(stop);

test('bare URL, refresh and new tabs use independent bootstrap without rotating token', async ({
    page,
    context,
}) => {
    await page.goto('/');
    await expect(page.getByLabel('Role for Claims', { exact: true })).toHaveValue('target');
    const second = await context.newPage();
    await second.goto(base);
    await expect(second.getByLabel('Role for Claims', { exact: true })).toHaveValue('target');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'One-way effects of Region', exact: true }),
    ).toBeVisible();
    await page.reload();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /One-way effects of/ }).first()).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await second.close();
});

test('actual server restart reconnects plot while retaining names, roles and raw JSON draft', async ({
    page,
}) => {
    await page.goto('/?view=variables');
    await openSharedColumns(page);
    await page.getByLabel('Name for DriverAge', { exact: true }).fill('DraftAge');
    await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await page.getByLabel('Role for VehicleAge', { exact: true }).selectOption('unassigned');
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const editor = page.getByLabel('Variables JSON', { exact: true });
    const raw = await editor.inputValue();
    await editor.fill(raw + '\n');
    await stop();
    await start();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'One-way effects of Region', exact: true }),
    ).toBeVisible();
    await openSharedColumns(page);
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    await expect(editor).toHaveValue(raw + '\n');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('DraftAge');
    await expect(page.getByLabel('Role for VehicleAge', { exact: true })).toHaveValue('unassigned');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.locator('.message[role="status"]')).toContainText('Settings applied');
    expect((await readProject()).data.renames.DriverAge).toBe('DraftAge');
});

test('restart clears a previously previewed Apply even when revision is again zero', async ({
    page,
}) => {
    await page.goto('/?view=variables');
    await openSharedColumns(page);
    await page.getByLabel('Role for Region', { exact: true }).selectOption('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeVisible();
    await stop();
    await start();
    // Background polling reconnects this open page and invalidates its old
    // preview. Keep the draft, but require a new preview before applying it.
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toHaveCount(0);
    expect((await readProject()).data.roles.Region).toBe('predictor');
    await expect(page.getByLabel('Role for Region', { exact: true })).toHaveValue('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.locator('.message[role="status"]')).toContainText('Settings applied');
    expect((await readProject()).data.roles.Region).toBe('ignore');
});

test('host and origin rejection does not trigger a bootstrap retry', async ({ page }) => {
    let bootstraps = 0;
    page.on('request', (r) => {
        if (r.url().endsWith('/api/session')) bootstraps++;
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /One-way effects of/ }).first()).toBeVisible();
    const before = bootstraps;
    await page.route(/\/api\/explore\?.*column=Region(?:&|$)/, (route) =>
        route.fulfill({
            status: 403,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Cross-origin access is refused.' }),
        }),
    );
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(page.getByText('Cross-origin access is refused.', { exact: true })).toBeVisible();
    expect(bootstraps).toBe(before);
});

test('incomplete JSON survives reconnection and a changed project cannot receive the old draft', async ({
    page,
}) => {
    await page.goto('/?view=variables');
    await openSharedColumns(page);
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    const editor = page.getByLabel('Variables JSON', { exact: true });
    await editor.fill('{unfinished');
    await stop();
    await start();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'One-way effects of Region', exact: true }),
    ).toBeVisible();
    await openSharedColumns(page);
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    await expect(editor).toHaveValue('{unfinished');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    const kept = await editor.inputValue();
    // Install the replacement-project response before restarting: background
    // polling may be the first request to notice the new server session.
    await page.route('**/api/variables', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        await route.fulfill({
            response,
            json: { ...data, project_id: 'another-project', name: 'Another project' },
        });
    });
    await stop();
    await start();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('DriverAge');
    await openSharedColumns(page);
    await expect(
        page.getByRole('alert').filter({ visible: true, hasText: 'different project' }).first(),
    ).toBeVisible();
    await openSharedColumns(page);
    await page.getByRole('button', { name: 'Variables JSON', exact: true }).click();
    await expect(editor).toHaveValue(kept);
    await expect(page.getByRole('button', { name: 'Preview changes', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Download draft', exact: true })).toBeVisible();
    await page.unroute('**/api/variables');
    await page.getByRole('button', { name: 'Discard draft and reload', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await openSharedColumns(page);
    await expect(page.getByRole('button', { name: 'Preview changes', exact: true })).toBeEnabled();
});
