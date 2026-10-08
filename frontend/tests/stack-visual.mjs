// Run against a local acceptance project with a fitted catalogue stack.
// EASYGLM_STACK_URL=http://127.0.0.1:8871 EASYGLM_STACK_MODEL='Reviewed vehicle smoke' \
// EASYGLM_STACK_FACTOR=Power node tests/stack-visual.mjs
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';

const url = process.env.EASYGLM_STACK_URL || 'http://127.0.0.1:8871';
const model = process.env.EASYGLM_STACK_MODEL;
const factor = process.env.EASYGLM_STACK_FACTOR || 'Power';
if (!model) throw new Error('Set EASYGLM_STACK_MODEL to a fitted catalogue stack.');
const browser = await chromium.launch({ headless: true, channel: 'chrome' });
let passed = false;
try {
    const page = await browser.newPage({ viewport: { width: 867, height: 850 }, acceptDownloads: true });
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(url);
    const nav = page.getByRole('navigation', { name: 'Workbench' });
    await nav.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
    await page.getByRole('button', { name: model, exact: true }).click();
    assert.ok(await page.locator('.stack-components li').count());
    await page.screenshot({ path: '/private/tmp/easyglm-stack-model-867.png', fullPage: true });
    await nav.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await page.getByLabel('A/E factor', { exact: true }).selectOption(factor);
    await page.locator('.stack-metrics').waitFor();
    assert.ok(await page.locator('.stack-table table').count());
    await page.screenshot({ path: '/private/tmp/easyglm-stack-diagnostics-867.png', fullPage: true });
    await nav.getByRole('button', { name: 'Rate tables', exact: true }).click();
    await page.getByLabel('Inspect model component').selectOption({ index: 0 });
    await page.getByRole('button', { name: 'Preview adjustment' }).waitFor();
    await page.screenshot({ path: '/private/tmp/easyglm-stack-lookup-867.png', fullPage: true });
    await nav.getByRole('button', { name: 'Export', exact: true }).click();
    await page.getByRole('button', { name: 'Download full lookup CSV' }).waitFor();
    await page.screenshot({ path: '/private/tmp/easyglm-stack-export-867.png', fullPage: true });
    const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.getByRole('button', { name: 'Download full lookup CSV' }).click(),
    ]);
    assert.match(download.suggestedFilename(), /\.csv$/);
    assert.deepEqual(errors, []);
    console.log('Stack browser review passed; screenshots are in /private/tmp/easyglm-stack-*-867.png');
    passed = true;
} catch (error) {
    console.error(error);
} finally {
    await Promise.race([
        browser.close(),
        new Promise((resolve) => setTimeout(resolve, 2000)),
    ]);
    process.exit(passed ? 0 : 1);
}
