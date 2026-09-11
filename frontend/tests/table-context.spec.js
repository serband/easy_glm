import { test, expect } from '@playwright/test';

test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });

test('late table and variable-review responses cannot replace the latest factor or tool', async ({
    page,
}) => {
    const errors = [];
    const commits = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => {
        if (request.method() === 'POST' && /\/api\/reviews\/[^/]+\/apply$/.test(request.url()))
            commits.push(request.url());
    });
    const button = (name) => page.getByRole('button', { name, exact: true });
    const variable = page.getByLabel('Rate table variable', { exact: true });
    const method = page.getByLabel('Adjustment method', { exact: true });
    const chart = page.locator('.relativity-chart');
    const chartHeading = (name) =>
        page.getByRole('heading', { name: `${name} · relativity`, exact: true });
    await page.goto('/');
    await button('Model').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (path) =>
        (await page.request.get('/api/' + path, { headers: { 'x-easyglm-token': token } })).json();
    const original = await get('project');
    const jobs = await get('jobs');

    // Hold the old factor's initial A/E response while the next factor gets a preview.
    let releaseReview, signalReview;
    const reviewGate = new Promise((resolve) => (releaseReview = resolve));
    const reviewHeld = new Promise((resolve) => (signalReview = resolve));
    let delayedReview = false;
    await page.route('**/api/review/Frequency', async (route) => {
        const response = await route.fetch();
        if (!delayedReview && route.request().postDataJSON().action === 'variable') {
            delayedReview = true;
            signalReview();
            await reviewGate;
        }
        await route.fulfill({ response });
    });
    await button('Rate tables').click();
    await reviewHeld;
    await variable.selectOption('AnnualMileage');
    await expect(chartHeading('AnnualMileage')).toBeVisible();
    await method.selectOption('moving');
    await expect(button('Apply adjustment')).toBeEnabled();
    const chosenPreview = await chart.innerHTML();
    releaseReview();
    await page.waitForTimeout(500);
    await expect(method).toHaveValue('moving');
    await expect(button('Apply adjustment')).toBeEnabled();
    expect(await chart.innerHTML()).toBe(chosenPreview);
    await page.unroute('**/api/review/Frequency');
    await button('Discard preview').click();

    async function holdRegion({ staleMarker = false, failure = false } = {}) {
        let release, signal;
        const gate = new Promise((resolve) => (release = resolve));
        const held = new Promise((resolve) => (signal = resolve));
        let delayed = false;
        const handler = async (route) => {
            const response = await route.fetch();
            const data = await response.json();
            if (
                !delayed &&
                new URL(route.request().url()).searchParams.get('variable') === 'Region'
            ) {
                delayed = true;
                signal();
                await gate;
                if (failure) {
                    await route.fulfill({ status: 500, json: { detail: 'Stale table error' } });
                    return;
                }
                if (staleMarker)
                    data.rows = data.rows.map((row) => ({ ...row, fitted: 99, relativity: 99 }));
            }
            await route.fulfill({ response, json: data });
        };
        await page.route('**/api/results/Frequency/table?*', handler);
        return {
            release,
            held,
            remove: () => page.unroute('**/api/results/Frequency/table?*', handler),
        };
    }

    // The old table disappears immediately; a faster request for another factor wins.
    const first = await holdRegion();
    await variable.selectOption('Region');
    await first.held;
    await expect(page.getByRole('status').filter({ hasText: 'Loading rate table…' })).toBeVisible();
    await expect(chart).toHaveCount(0);
    await expect(method).toHaveCount(0);
    await variable.selectOption('DriverAge');
    await expect(chartHeading('DriverAge')).toBeVisible();
    const driverAge = await chart.innerHTML();
    first.release();
    await page.waitForTimeout(300);
    await expect(chartHeading('DriverAge')).toBeVisible();
    expect(await chart.innerHTML()).toBe(driverAge);
    await first.remove();

    // A -> B -> A still rejects the first A response, even though its context matches again.
    const sameFactor = await holdRegion({ staleMarker: true });
    await variable.selectOption('Region');
    await sameFactor.held;
    await variable.selectOption('AnnualMileage');
    await expect(chartHeading('AnnualMileage')).toBeVisible();
    await variable.selectOption('Region');
    await expect(chartHeading('Region')).toBeVisible();
    const region = await chart.innerHTML();
    sameFactor.release();
    await page.waitForTimeout(300);
    expect(await chart.innerHTML()).toBe(region);
    await sameFactor.remove();

    // Errors from obsolete requests must not leak into the newly selected factor.
    await variable.selectOption('DriverAge');
    await expect(chartHeading('DriverAge')).toBeVisible();
    const failure = await holdRegion({ failure: true });
    await variable.selectOption('Region');
    await failure.held;
    await variable.selectOption('AnnualMileage');
    await expect(chartHeading('AnnualMileage')).toBeVisible();
    failure.release();
    await page.waitForTimeout(300);
    await expect(page.getByText('Stale table error', { exact: true })).toHaveCount(0);
    await failure.remove();
    expect(await get('project')).toEqual(original);
    expect((await get('jobs')).Frequency.id).toBe(jobs.Frequency.id);
    expect(commits).toEqual([]);
    expect(errors).toEqual([]);
});
