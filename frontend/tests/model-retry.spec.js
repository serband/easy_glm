import { test, expect } from '@playwright/test';

async function openStackBuilder(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Build a model in stages' }).click();
}

async function fillDraft(page, name) {
    await page.getByLabel('Stacked model name').fill(name);
    await page.getByLabel('Use Region').check();
}

test('a delayed old failed job cannot unlock a newly accepted retry', async ({ page }) => {
    const name = 'Retry after failure';
    await openStackBuilder(page);
    await page.getByRole('button', { name: 'Start a new model chain', exact: true }).first().click();
    await fillDraft(page, name);
    await page.getByRole('button', { name: 'Save draft', exact: true }).first().click();
    await expect(page.getByText('Draft saved.', { exact: true })).toBeVisible();

    const oldFailed = { [name]: { id: 'previous-job', status: 'failed', applicable: false, message: 'Previous fit failed.' } };
    await page.route('**/api/jobs', async (route) => route.fulfill({ json: oldFailed }));
    await expect(page.getByRole('alert')).toHaveText('Previous fit failed.', { timeout: 5000 });
    await page.getByRole('button', { name: 'Edit draft', exact: true }).click();

    let oldSnapshotRead = false;
    const delayedOldSnapshot = async (route) => {
        oldSnapshotRead = true;
        await route.fulfill({ json: oldFailed });
    };
    await page.unroute('**/api/jobs');
    await page.route('**/api/jobs', delayedOldSnapshot);
    await page.getByRole('button', { name: 'Fit model', exact: true }).first().click();
    await expect.poll(() => oldSnapshotRead).toBe(true);
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitting');
    await expect(page.locator('.fit-progress')).toContainText(/Starting fit|Fitting:/);
    await expect(page.getByRole('heading', { name: 'Edit draft model', exact: true })).toHaveCount(0);
    await expect(page.getByRole('button', { name: 'Edit draft', exact: true })).toHaveCount(0);

    await page.unroute('**/api/jobs', delayedOldSnapshot);
    await expect(page.locator('.selected-model-header .model-state')).toHaveText('Fitted', { timeout: 60000 });
});
