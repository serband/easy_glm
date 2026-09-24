// Fresh demo server: PYTHONPATH=src .venv/bin/python -m easy_glm.desktop --demo --port 8872
// Then: EASYGLM_STACK_URL=http://127.0.0.1:8872 node tests/stack-legacy-append.mjs
import assert from 'node:assert/strict';
import { chromium } from '@playwright/test';

const url = process.env.EASYGLM_STACK_URL || 'http://127.0.0.1:8872';
const browser = await chromium.launch({ headless: true, channel: 'chrome' });
let passed = false;
try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(url);
    await page.getByRole('navigation', { name: 'Workbench' }).getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Create model', exact: true }).click();
    await page.getByRole('button', { name: 'Fit model', exact: true }).click();
    await page.getByText('Fit complete', { exact: true }).waitFor({ timeout: 30000 });
    const session = await (await page.request.get(`${url}/api/session`)).json();
    async function workbench() {
        const response = await page.request.get(`${url}/api/workbench`, { headers: { 'X-EasyGLM-Token': session.token } });
        assert.equal(response.status(), 200);
        return response.json();
    }
    const original = await workbench();
    const sourceId = original.jobs.Frequency.id;
    await page.getByRole('button', { name: 'Build a stacked model' }).click();
    await page.getByLabel('Start from fitted candidate').selectOption('Frequency');
    await page.getByLabel('New stacked model name').fill('Legacy then CatBoost');
    await page.getByLabel('New stage type').selectOption('catboost');
    await page.getByLabel('New stage name').fill('Vehicle correction');
    await page.getByRole('group', { name: 'Stage factors' }).getByText('VehicleAge', { exact: true }).click();
    await page.getByLabel('CatBoost iterations').fill('20');
    await page.getByLabel('CatBoost time limit').fill('1');
    await page.getByRole('button', { name: 'Save new candidate' }).click();
    await page.getByLabel('Stacked model', { exact: true }).selectOption('Legacy then CatBoost');
    await page.getByRole('button', { name: 'Fit candidate' }).click();
    await page.getByText('Review complete model', { exact: true }).waitFor({ timeout: 30000 });
    const after = await workbench();
    assert.equal(after.jobs.Frequency.id, sourceId, 'source fit ID changed during append');
    assert.equal(after.jobs['Legacy then CatBoost'].status, 'complete');
    assert.equal(await page.locator('.stack-components li').count(), 2);
    assert.deepEqual(errors, []);
    console.log('Legacy GLM → CatBoost browser append passed with unchanged source fit ID.');
    passed = true;
} catch (error) {
    console.error(error);
} finally {
    await Promise.race([browser.close(), new Promise((resolve) => setTimeout(resolve, 2000))]);
    process.exit(passed ? 0 : 1);
}
