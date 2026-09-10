import { test, expect } from '@playwright/test';
import { mkdtemp, writeFile, rm, readFile, realpath } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

test('empty startup loads local data and reopens a project without stale drafts', async ({
    page,
}, info) => {
    const folder = await realpath(await mkdtemp(join(tmpdir(), 'easyglm_onboarding_')));
    const csv = join(folder, 'portfolio.csv');
    await writeFile(csv, 'Claims,Exposure,Age\n0,1,20\n1,0.5,30\n0,1,40\n');
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    const button = (name) => page.getByRole('button', { name, exact: true });
    try {
        await page.goto('/');
        await expect(
            page.getByRole('heading', { name: 'Project & data', exact: true }),
        ).toBeVisible();
        await expect(page.getByText('EasyGLM 0.460', { exact: true })).toBeVisible();
        await page.getByLabel('File path', { exact: true }).fill(join(folder, 'missing.csv'));
        await button('Open data').click();
        await expect(page.locator('.open-error')).toBeVisible();
        await page.getByLabel('File path', { exact: true }).fill(csv);
        await button('Open data').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        await expect(page.getByText('Claims', { exact: true }).first()).toBeVisible();
        await button('Export').click();
        const [download] = await Promise.all([
            page.waitForEvent('download'),
            button('Download project JSON').click(),
        ]);
        const project = join(folder, 'saved.easyglm-project.json');
        await download.saveAs(project);
        expect(JSON.parse(await readFile(project, 'utf8')).data.source.path).toBe(csv);
        await button('Project & data').click();
        await page.locator('.project-open > summary').click();
        await page
            .getByRole('combobox', { name: 'Open file type', exact: true })
            .selectOption('project');
        await page.getByLabel('File path', { exact: true }).fill(project);
        await expect(button('Open project')).toBeDisabled();
        await page.getByLabel('Replace this session.', { exact: false }).check();
        await button('Open project').click();
        await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
        await button('Project & data').click();
        await expect(page.getByText(csv, { exact: true })).toBeVisible();
        expect(
            await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1),
        ).toBe(true);
        await page.screenshot({ path: info.outputPath('onboarding.png'), fullPage: true });
        expect(errors).toEqual([]);
    } finally {
        await rm(folder, { recursive: true, force: true });
    }
});
