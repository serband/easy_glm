import { test, expect } from '@playwright/test';
import { openNewLegacyDraft } from './legacy-models.helpers.js';

const button = (page, name) => page.getByRole('button', { name, exact: true });

async function newModel(page, name) {
    await page.goto('/');
    await openNewLegacyDraft(page);
    await page.getByLabel('New model name').fill(name);
    await page.getByLabel('Model family').selectOption('gaussian');
}

async function choosePair(page) {
    await page.getByLabel('New pair first predictor').selectOption('DriverAge');
    await page.getByLabel('New pair second predictor').selectOption('Region');
}

test('Gaussian default and explicit log allow sequential pairs; non-log preserves the draft but blocks actions', async ({ page }) => {
    const saves = [];
    page.on('request', (request) => {
        if (request.method() === 'POST' && request.url().endsWith('/api/models/save')) {
            saves.push(request.postDataJSON());
        }
    });
    await newModel(page, `Gaussian pairs ${Date.now()}`);
    await expect(page.getByLabel('Model link')).toHaveValue('');
    await expect(page.getByRole('region', { name: 'Sequential pair stages' })).toBeVisible();
    await choosePair(page);
    await button(page, 'Add interaction').click();
    const stage = page.getByLabel('Stage 2 DriverAge × Region');
    await expect(stage).toBeVisible();
    await page.getByLabel('Model link').selectOption('logit');
    await expect(page.getByLabel('Model link')).toHaveValue('logit');
    await expect(stage).toBeVisible();
    await expect(page.getByText('Gaussian interactions require a log link.')).toBeVisible();
    await choosePair(page);
    await expect(button(page, 'Add interaction')).toBeDisabled();
    await expect(button(page, 'Create model')).toBeDisabled();
    await page.getByLabel('Model link').selectOption('log');
    await expect(stage).toBeVisible();
    await expect(button(page, 'Create model')).toBeEnabled();
    await button(page, 'Create model').click();
    await expect(page.getByText('Model settings saved. Fit when ready.')).toBeVisible();
    expect(saves.at(-1).fields.link).toBe('log');
    expect(saves.at(-1).fields.pair_stages).toHaveLength(1);
    await expect(page.getByRole('button', { name: /^Fit (all|remaining) stages$/ })).toBeEnabled();
    await page.getByLabel('Model link').selectOption('logit');
    await expect(stage).toBeVisible();
    await expect(button(page, 'Save model settings')).toBeDisabled();
    await expect(page.getByRole('button', { name: /^Fit (all|remaining) stages$/ })).toBeDisabled();
    await page.getByLabel('Remove pair stage DriverAge × Region').click();
    await expect(button(page, 'Save model settings')).toBeEnabled();
    await page.getByLabel('Model link').selectOption('log');
    await expect(button(page, 'Add interaction')).toBeEnabled();
});

test('Gaussian non-log links cannot add legacy interactions; other families retain legacy controls', async ({ page }) => {
    await newModel(page, `Gaussian legacy ${Date.now()}`);
    await button(page, 'Legacy GLM interactions').click();
    await page.getByLabel('Interaction first factor').selectOption('DriverAge');
    await page.getByLabel('Interaction second factor').selectOption('Region');
    await button(page, 'Add interaction').click();
    await page.getByLabel('Model link').selectOption('logit');
    await expect(page.getByRole('region', { name: 'Defined interactions' })).toBeVisible();
    await expect(page.getByText('DriverAge × Region')).toBeVisible();
    await expect(button(page, 'Create model')).toBeDisabled();
    await page.getByLabel('Remove interaction DriverAge × Region').click();
    await page.getByLabel('Interaction first factor').selectOption('DriverAge');
    await page.getByLabel('Interaction second factor').selectOption('Region');
    await expect(button(page, 'Add interaction')).toBeDisabled();
    await expect(button(page, 'Create model')).toBeEnabled();
    await button(page, 'Sequential CatBoost pairs').click();
    await choosePair(page);
    await expect(button(page, 'Add interaction')).toBeDisabled();
    await page.getByLabel('Model family').selectOption('gamma');
    await button(page, 'Legacy GLM interactions').click();
    await page.getByLabel('Interaction first factor').selectOption('DriverAge');
    await page.getByLabel('Interaction second factor').selectOption('Region');
    await expect(button(page, 'Add interaction')).toBeEnabled();
});
