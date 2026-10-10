import { test, expect } from '@playwright/test';

async function startPlan(page, mode) {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: mode, exact: true }).click();
}

async function openFrenchMotorExample(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('radio', { name: 'Example dataset', exact: true }).check();
    if (await page.getByRole('checkbox', { name: /Replace this session/ }).count()) {
        await page.getByRole('checkbox', { name: /Replace this session/ }).check();
    }
    const loaded = page.waitForResponse((response) => response.url().endsWith('/api/project/open') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Load example', exact: true }).click();
    expect((await loaded).ok()).toBe(true);
    await expect(page.getByRole('button', { name: 'Load example', exact: true })).toBeDisabled();
    await expect(page.getByRole('heading', { name: 'Project model plan' })).toBeVisible();
}

async function planFitRefs(page) {
    return page.evaluate(async () => {
        const session = await fetch('/api/session').then((response) => response.json());
        const response = await fetch('/api/model-plan', {
            headers: { 'X-EasyGLM-Token': session.token },
        });
        const current = await response.json();
        return current.fit_refs.map((reference) => reference.model_name);
    });
}

async function chooseAndFitFirstGlm(page, variable = 'DriverAge') {
    await page.getByRole('button', { name: 'Choose variables', exact: true }).first().click();
    await page.getByLabel(`Use ${variable}`).check();
    await page.getByRole('button', { name: 'Save stage setup', exact: true }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: /Fit .+/ }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toBeVisible({ timeout: 120000 });
}

async function openBinaryData(page) {
    const rows = ['Claim,DriverAge'];
    for (let index = 0; index < 180; index += 1)
        rows.push(`${index % 3 === 0 ? 1 : 0},${20 + (index % 55)}`);
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByText('Own data', { exact: true }).click();
    await page.locator('input[type="file"]').setInputFiles({
        name: 'binary_claims.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from(rows.join('\n')),
    });
    await page.getByRole('checkbox', { name: 'Replace this session. I have saved any work I want to keep.' }).check();
    const loaded = page.waitForResponse((response) => response.url().includes('/api/project/upload?') && response.request().method() === 'POST');
    await page.getByRole('button', { name: 'Open data', exact: true }).click();
    expect((await loaded).ok()).toBe(true);
    await expect(page.getByRole('button', { name: 'Open data', exact: true })).toBeDisabled();
    await page.getByRole('button', { name: /^Variables\s*\d*$/ }).click();
    await page.getByLabel('Role for Claim').selectOption('target');
    await page.getByLabel('Role for DriverAge').selectOption('predictor');
    await page.getByRole('button', { name: 'Preview changes', exact: true }).click();
    await page.getByRole('button', { name: 'Apply changes', exact: true }).click();
}

test('single-model links are restricted to scorers that can be diagnosed and exported', async ({ page }) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));

    await openFrenchMotorExample(page);
    await startPlan(page, 'Single model');
    await page.getByLabel('Model family').selectOption('gaussian');
    await expect(page.getByLabel('Model link').locator('option[value="identity"]')).toHaveCount(0);

    // The API applies the same rule.  A stale client must not be able to create
    // a plan that can never produce a rate-table scorer.
    const rejected = await page.evaluate(async () => {
        const session = await fetch('/api/session').then((response) => response.json());
        const headers = {
            'content-type': 'application/json',
            'X-EasyGLM-Token': session.token,
        };
        const current = await fetch('/api/model-plan', { headers }).then((response) => response.json());
        const response = await fetch('/api/model-plan', {
            method: 'POST',
            headers,
            body: JSON.stringify({
                session_id: current.session_id,
                revision: current.revision,
                operation: 'save',
                plan: { ...current.plan, family: 'gaussian', link: 'identity' },
            }),
        });
        return response.status;
    });
    expect(rejected).toBe(422);

    await chooseAndFitFirstGlm(page, 'DrivAge');

    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Actual versus expected' })).toBeVisible();
    await page.getByRole('button', { name: 'Export', exact: true }).click();
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Download scoring script (.py)', exact: true }).click();
    expect((await download).suggestedFilename()).toMatch(/\.py$/);
    expect(errors).toEqual([]);
});

test('a single binomial logit model can be inspected and exported', async ({ page }) => {
    await openBinaryData(page);
    await startPlan(page, 'Single model');
    await page.getByLabel('Model family').selectOption('binomial');
    await page.getByLabel('Model link').selectOption('logit');
    await chooseAndFitFirstGlm(page);
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Actual versus expected' })).toBeVisible();
    await page.getByRole('button', { name: 'Export', exact: true }).click();
    await expect(page.getByRole('button', { name: 'Download scorer (.easyglm)', exact: true })).toBeEnabled();
});

test('diagnostics opens the latest staged fit and lets the actuary inspect an earlier prefix', async ({ page }) => {
    await openFrenchMotorExample(page);
    await startPlan(page, 'Build in stages');
    await chooseAndFitFirstGlm(page, 'DrivAge');
    const [first] = await planFitRefs(page);

    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await page.getByRole('button', { name: /Add .*stage/ }).click();
    await page.getByLabel('Stage 2 name').locator('xpath=ancestor::li').getByRole('button', { name: 'Choose variables', exact: true }).click();
    await page.getByLabel('Use DrivAge').check();
    await page.getByRole('button', { name: 'Save stage setup', exact: true }).click();
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: /Fit .+/ }).click();
    await expect(page.getByText('Fitted and kept as a saved version.')).toHaveCount(2, { timeout: 120000 });
    const [savedFirst, latest] = await planFitRefs(page);
    expect(savedFirst).toBe(first);
    expect(latest).toBeTruthy();

    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    const modelPicker = page.getByLabel('Diagnostic saved model');
    await expect(modelPicker).toHaveValue(latest);
    await modelPicker.selectOption(savedFirst);
    await expect(modelPicker).toHaveValue(savedFirst);
    await expect(page.getByText('This is the first model, using the project data.')).toBeVisible();

    await page.reload();
    await page.getByRole('button', { name: 'Diagnostics', exact: true }).click();
    await expect(page.getByRole('heading', { name: /Diagnostics/ })).toBeVisible();
    await expect(page.getByLabel('Diagnostic saved model')).toHaveValue(latest);
});
