import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('dropdown previews are transient, latest-only and committed only by explicit Apply', async ({
    page,
}) => {
    const button = (n) => page.getByRole('button', { name: n, exact: true });
    const method = page.getByLabel('Adjustment method', { exact: true });
    const apply = button('Apply adjustment');
    const errors = [];
    const starts = [];
    const commits = [];
    const candidates = new Map();
    let firstRequestAt;
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (
            r.method() === 'POST' &&
            /\/api\/review\/[^/]+$/.test(r.url()) &&
            ['moving', 'isotonic', 'cap', 'edit'].includes(r.postDataJSON().action)
        ) {
            starts.push(r.postDataJSON());
            firstRequestAt ??= Date.now();
        }
        if (r.method() === 'POST' && /\/api\/reviews\/[^/]+\/apply$/.test(r.url()))
            commits.push(r.url().split('/').at(-2));
    });
    page.on('response', async (r) => {
        if (r.request().method() !== 'GET' || !/\/api\/reviews\/[^/]+$/.test(r.url())) return;
        const data = await r.json().catch(() => null);
        if (data?.status === 'complete' && data.data?.preview_table)
            candidates.set(data.id, data.data);
    });
    await page.goto('/');
    await button('Model').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    await button('Rate tables').click();
    const ae = page.getByRole('img', {
        name: 'Actual fitted and adjusted by variable',
        exact: true,
    });
    await expect(ae).toBeVisible();
    const { token } = await (await page.request.get('/api/session')).json();
    const get = async (p) =>
        (await page.request.get('/api/' + p, { headers: { 'x-easyglm-token': token } })).json();
    const initial = await get('project');
    const jobs = await get('jobs');
    const history = await get('review-info/Frequency');
    const chart = page.locator('.relativity-chart');
    const initialChart = await chart.innerHTML();
    const initialAe = await ae.innerHTML();
    await expect(method).toHaveValue('');
    await expect(method.locator('option[value="round"]')).toHaveCount(0);
    // Four-decimal display must not turn an unchanged input into a saved edit.
    await method.selectOption('manual');
    const untouched = page.getByLabel('Relativity row 1', { exact: true });
    const displayedValue = await untouched.inputValue();
    expect(displayedValue).toMatch(/^[-+]?\d+(?:\.\d{1,4})?$/);
    await untouched.focus();
    await untouched.press('Tab');
    await untouched.fill(displayedValue);
    await untouched.press('Tab');
    await page.waitForTimeout(250);
    expect(await get('project')).toEqual(initial);
    await expect(apply).toHaveCount(0);
    await method.selectOption('');
    await page.waitForTimeout(250);
    expect(starts).toHaveLength(0);
    await expect(button('Moving average')).toHaveCount(0);
    // Selection creates a candidate; neither preparation nor changing options saves it.
    const selectedAt = Date.now();
    await method.selectOption('moving');
    await expect(method).toBeEnabled();
    await expect(apply).toBeEnabled();
    console.log(
        JSON.stringify({
            firstPreviewMs: Date.now() - selectedAt,
            workerAndPollingMs: Date.now() - firstRequestAt,
        }),
    );
    await expect(page.getByText('Preview · not applied', { exact: true })).toBeVisible();
    expect(await chart.innerHTML()).not.toBe(initialChart);
    expect(await ae.innerHTML()).not.toBe(initialAe);
    expect(commits).toHaveLength(0);
    expect(await get('project')).toEqual(initial);
    expect(await get('review-info/Frequency')).toEqual(history);
    const first = [...candidates.values()].at(-1);
    const original = first.preview_table.rows.map((r) => r.fitted);
    await page.getByLabel('Smoothing window').fill('5');
    await expect(apply).toBeDisabled();
    await expect(apply).toBeEnabled();
    const second = [...candidates.values()].at(-1);
    expect(second.preview_table.rows.map((r) => r.fitted)).toEqual(original);
    expect(second.preview_table.rows.map((r) => r.relativity)).not.toEqual(
        first.preview_table.rows.map((r) => r.relativity),
    );
    await button('Discard preview').click();
    await expect(method).toHaveValue('');
    expect(await chart.innerHTML()).toBe(initialChart);
    expect(await ae.innerHTML()).toBe(initialAe);
    expect(await get('project')).toEqual(initial);
    // Invalid drafts clear the old candidate and do not dispatch work.
    await method.selectOption('moving');
    await expect(apply).toBeEnabled();
    const beforeInvalid = starts.length;
    await page.getByLabel('Smoothing window').fill('');
    await expect(
        page.getByText('Enter a whole-number window from 1 to 25.', { exact: true }),
    ).toBeVisible();
    await page.waitForTimeout(250);
    expect(starts.length).toBe(beforeInvalid);
    await expect(apply).toHaveCount(0);
    expect(await chart.innerHTML()).toBe(initialChart);
    await page.getByLabel('Smoothing window').fill('4');
    await expect(apply).toBeEnabled();
    // Keep a completed older response in flight; a newer tool must win without waiting.
    let release, held;
    const gate = new Promise((r) => (release = r)),
        waiting = new Promise((r) => (held = r));
    let delayedId = null;
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch();
        const data = await response.json();
        if (!delayedId && data.status === 'complete' && data.data?.preview_table) {
            delayedId = data.id;
            held();
            await gate;
        }
        await route.fulfill({ response, json: data });
    });
    await page.getByLabel('Smoothing window').fill('6');
    await waiting;
    await expect(method).toBeEnabled();
    await method.selectOption('cap');
    await page.getByLabel('Relativity cap').fill('0.8');
    await expect(apply).toBeEnabled();
    const latest = await chart.innerHTML();
    release();
    await page.waitForTimeout(300);
    expect(await chart.innerHTML()).toBe(latest);
    expect(commits).toHaveLength(0);
    await page.unrouteAll({ behavior: 'wait' });
    const readyId = [...candidates.keys()].findLast((id) => id !== delayedId);
    const requestsBeforeApply = starts.length;
    await apply.dblclick();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect(commits).toEqual([readyId]);
    expect(starts).toHaveLength(requestsBeforeApply);
    const applied = await get('project');
    expect(applied.models.Frequency.adjustments).not.toEqual(initial.models.Frequency.adjustments);
    await expect(method).toHaveValue('');
    await expect(page.getByText('Saved adjustments', { exact: true })).toBeVisible();
    const savedChart = await chart.innerHTML(),
        savedAe = await ae.innerHTML();
    // Failure keeps saved edits; selecting another option retries only a new draft.
    await page.route('**/api/review/Frequency', (route) =>
        route.fulfill({ status: 422, json: { detail: 'Calculation failed for test' } }),
    );
    await method.selectOption('isotonic');
    await expect(page.getByText('Calculation failed for test', { exact: true })).toBeVisible();
    expect(await get('project')).toEqual(applied);
    expect(await chart.innerHTML()).toBe(savedChart);
    expect(await ae.innerHTML()).toBe(savedAe);
    await page.unroute('**/api/review/Frequency');
    await page.getByLabel('Smoothing direction').selectOption('decreasing');
    await expect(apply).toBeEnabled();
    await button('Discard preview').click();
    expect(await chart.innerHTML()).toBe(savedChart);
    expect(await ae.innerHTML()).toBe(savedAe);
    expect(await get('project')).toEqual(applied);
    // Changing the selected factor cannot display/apply an old factor's candidate.
    let release2, held2;
    const gate2 = new Promise((r) => (release2 = r)),
        waiting2 = new Promise((r) => (held2 = r));
    let delayed2 = false;
    await page.route('**/api/reviews/*', async (route) => {
        const response = await route.fetch(),
            data = await response.json();
        if (!delayed2 && data.status === 'complete' && data.data?.preview_table) {
            delayed2 = true;
            held2();
            await gate2;
        }
        await route.fulfill({ response, json: data });
    });
    await method.selectOption('moving');
    await waiting2;
    await page.getByLabel('Rate table variable', { exact: true }).selectOption('Region');
    release2();
    await expect(method).toHaveValue('');
    await expect(
        page.getByRole('heading', { name: 'Region · Holdout', exact: true }),
    ).toBeVisible();
    await expect(apply).toHaveCount(0);
    await expect(page.locator('.rate-preview-state')).toHaveText('Original fit');
    expect(commits).toHaveLength(1);
    await page.unrouteAll({ behavior: 'wait' });
    // Manual edits stay usable while previews run and commit together in one undo step.
    await method.selectOption('manual');
    const beforeManual = await get('project');
    await page.getByLabel('Relativity row 1', { exact: true }).fill('');
    await expect(
        page.getByText('Enter positive values for each edited row.', { exact: true }),
    ).toBeVisible();
    await expect(apply).toHaveCount(0);
    await page.getByLabel('Relativity row 1', { exact: true }).fill('1.6');
    await page.getByLabel('Relativity row 2', { exact: true }).fill('2.3');
    await expect(page.getByLabel('Relativity row 1', { exact: true })).toBeEnabled();
    await expect(apply).toBeEnabled();
    expect(await get('project')).toEqual(beforeManual);
    await apply.click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect(commits).toHaveLength(2);
    await button('Preview undo').click();
    await expect(apply).toBeEnabled();
    await apply.click();
    await expect(page.getByText('Adjustments applied.', { exact: true })).toBeVisible();
    expect((await get('project')).models.Frequency.adjustments).toEqual(
        beforeManual.models.Frequency.adjustments,
    );
    // A successful Apply followed by display-refresh failure remains saved exactly once.
    await method.selectOption('manual');
    await page.getByLabel('Relativity row 1', { exact: true }).fill('1.83');
    await expect(apply).toBeEnabled();
    await page.route('**/api/workbench', (route) =>
        route.fulfill({ status: 500, json: { detail: 'Refresh failed for test' } }),
    );
    const beforeCount = commits.length;
    await apply.click();
    await expect(
        page.getByText('Adjustments saved. Refresh the page to reload the charts.', {
            exact: true,
        }),
    ).toBeVisible();
    expect(commits).toHaveLength(beforeCount + 1);
    expect((await get('project')).models.Frequency.adjustments).not.toEqual(
        beforeManual.models.Frequency.adjustments,
    );
    await page.unroute('**/api/workbench');
    expect((await get('jobs')).Frequency.id).toBe(jobs.Frequency.id);
    expect(errors).toEqual([]);
});
