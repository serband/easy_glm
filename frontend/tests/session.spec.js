import { test, expect } from '@playwright/test';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
const root = path.resolve('..');
const base = 'http://127.0.0.1:8773';
let server;
async function start() {
    const id = randomUUID();
    server = spawn(
        path.join(root, '.venv/bin/python'),
        ['-m', 'easy_glm.desktop', '--port', '8773', '--launch-id', id],
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
            { timeout: 15000 },
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
        page.getByRole('img', { name: 'Distribution of Region', exact: true }),
    ).toBeVisible();
    await page.reload();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /Distribution of/ }).first()).toBeVisible();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await second.close();
});

test('actual server restart reconnects plot while retaining names, types, roles and raw JSON draft', async ({
    page,
}) => {
    await page.goto('/');
    await page.getByLabel('Name for DriverAge', { exact: true }).fill('DraftAge');
    await page.getByLabel('Name for DriverAge', { exact: true }).press('Tab');
    await page.getByLabel('Type for DriverAge', { exact: true }).selectOption('categorical');
    await page.getByLabel('Role for VehicleAge', { exact: true }).selectOption('unassigned');
    await page.getByRole('button', { name: 'Role JSON', exact: true }).click();
    const editor = page.getByLabel('Role JSON', { exact: true });
    const raw = await editor.inputValue();
    await editor.fill(raw + '\n');
    await stop();
    await start();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'Distribution of Region', exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(editor).toHaveValue(raw + '\n');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'Table', exact: true }).click();
    await expect(page.getByLabel('Name for DriverAge', { exact: true })).toHaveValue('DraftAge');
    await expect(page.getByLabel('Type for DriverAge', { exact: true })).toHaveValue('categorical');
    await expect(page.getByLabel('Role for VehicleAge', { exact: true })).toHaveValue('unassigned');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    expect((await readProject()).data.renames.DriverAge).toBe('DraftAge');
});

test('restart refuses a previously previewed Apply even when revision is again zero', async ({
    page,
}) => {
    await page.goto('/');
    await page.getByLabel('Role for Region', { exact: true }).selectOption('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Apply changes', exact: true })).toBeVisible();
    await stop();
    await start();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByRole('alert')).toContainText('Preview changes again');
    expect((await readProject()).data.roles.Region).toBe('predictor');
    await expect(page.getByLabel('Role for Region', { exact: true })).toHaveValue('ignore');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
    await expect(page.getByRole('status')).toContainText('Settings applied');
    expect((await readProject()).data.roles.Region).toBe('ignore');
});

test('host and origin rejection does not trigger a bootstrap retry', async ({ page }) => {
    let bootstraps = 0;
    page.on('request', (r) => {
        if (r.url().endsWith('/api/session')) bootstraps++;
    });
    await page.goto('/');
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await expect(page.getByRole('img', { name: /Distribution of/ }).first()).toBeVisible();
    const before = bootstraps;
    await page.route('**/api/plot?column=Region', (route) =>
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
    await page.goto('/');
    await page.getByRole('button', { name: 'Role JSON', exact: true }).click();
    const editor = page.getByLabel('Role JSON', { exact: true });
    await editor.fill('{unfinished');
    await stop();
    await start();
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('Region');
    await expect(
        page.getByRole('img', { name: 'Distribution of Region', exact: true }),
    ).toBeVisible();
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(editor).toHaveValue('{unfinished');
    await expect(page.getByRole('alert')).toHaveCount(0);
    await page.getByRole('button', { name: 'Reset', exact: true }).click();
    const kept = await editor.inputValue();
    await stop();
    await start();
    await page.route('**/api/variables', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        await route.fulfill({
            response,
            json: { ...data, project_id: 'another-project', name: 'Another project' },
        });
    });
    await page.getByRole('button', { name: 'Explore', exact: true }).click();
    await page.getByLabel('Plot variable', { exact: true }).selectOption('DriverAge');
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByRole('alert')).toContainText('different project');
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(editor).toHaveValue(kept);
    await expect(page.getByRole('button', { name: 'Preview changes', exact: true })).toBeDisabled();
    await expect(page.getByRole('button', { name: 'Download draft', exact: true })).toBeVisible();
    await page.unroute('**/api/variables');
    await page.getByRole('button', { name: 'Discard draft and reload', exact: true }).click();
    await expect(page.getByRole('alert')).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Preview changes', exact: true })).toBeEnabled();
});
