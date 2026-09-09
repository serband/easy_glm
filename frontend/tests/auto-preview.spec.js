import { test, expect } from '@playwright/test';
test.use({ viewport: { width: 884, height: 773 }, actionTimeout: 15000 });
test('automatic previews debounce, reject stale results and preserve manual drafts', async ({
    page,
}) => {
    const button = (name) => page.getByRole('button', { name, exact: true });
    const starts = [];
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (r.method() === 'POST' && /\/api\/review\/Frequency$/.test(r.url())) {
            const d = r.postDataJSON();
            if (['moving', 'isotonic', 'cap', 'round'].includes(d.action)) starts.push(d);
        }
    });
    await page.goto('/');
    await button('Model').click();
    if (!(await page.locator('.split-settings').evaluate((el) => el.open)))
        await page.locator('.split-settings > summary').click();
    await page.getByLabel('Split method').selectOption('random');
    await button('Apply split').click();
    await button('Create model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 30000 });
    const { token } = await (await page.request.get('/api/session')).json();
    const headers = { 'x-easyglm-token': token };
    const get = async (p) => (await page.request.get('/api/' + p, { headers })).json();
    const initialJob = await get('jobs');
    await button('Rate tables').click();
    await expect(page.locator('.rate-ae svg').first()).toBeVisible();
    await page.waitForTimeout(600);
    expect(starts).toHaveLength(0);
    await expect(button('Preview adjustment')).toHaveCount(0);
    let release,
        held = false;
    const gate = new Promise((r) => (release = r));
    let delayed = false;
    await page.route('**/api/reviews/*', async (route) => {
        if (route.request().method() !== 'GET') return route.continue();
        const response = await route.fetch();
        const body = await response.json();
        if (!delayed && body.status === 'complete' && body.data?.tool_details) {
            delayed = true;
            held = true;
            await gate;
        }
        await route.fulfill({ response });
    });
    await button('Cap / floor').click();
    await expect.poll(() => held).toBeTruthy();
    const cap = page.getByLabel('Relativity cap', { exact: true });
    await cap.fill('.6');
    await cap.fill('.7');
    await cap.fill('.8');
    const scroll = await page.evaluate(() => scrollY);
    release();
    await expect(page.locator('.preview-impact')).toBeVisible();
    await expect(button('Apply adjustment')).toBeEnabled();
    expect(starts.at(-1).options.cap).toBe(0.8);
    expect(starts.length).toBeLessThanOrEqual(2);
    await expect(cap).toBeFocused();
    expect(Math.abs((await page.evaluate(() => scrollY)) - scroll)).toBeLessThan(3);
    const applied = page.waitForRequest(
        (r) => r.method() === 'POST' && /\/api\/reviews\/[^/]+\/apply$/.test(r.url()),
    );
    await button('Apply adjustment').click();
    const applyRequest = await applied;
    const id = applyRequest.url().split('/').at(-2);
    const result = await get('reviews/' + id);
    expect(
        result.data.preview_table.rows
            .filter((r) => !(r.from === null && r.to === null))
            .every((r) => r.relativity <= 0.8 + 1e-12),
    ).toBeTruthy();
    await expect(page.getByText(/Adjustments applied/)).toBeVisible();
    const count = starts.length;
    await page.waitForTimeout(700);
    expect(starts).toHaveLength(count);
    expect((await get('jobs')).Frequency.id).toBe(initialJob.Frequency.id);
    await cap.fill('0');
    await expect(button('Apply adjustment')).toBeDisabled();
    await expect(page.getByText(/Use positive bounds/)).toBeVisible();
    await cap.fill('.5');
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Discard preview').click();
    await page.waitForTimeout(700);
    await expect(button('Apply adjustment')).toHaveCount(0);
    const discarded = starts.length;
    await page.waitForTimeout(500);
    expect(starts).toHaveLength(discarded);
    await button('Edit individual or multiple rows').click();
    const cell = page.getByLabel('Relativity row 2', { exact: true });
    await cell.fill('1.37');
    await cell.press('Tab');
    await expect(button('Moving average')).toBeDisabled();
    await expect(cell).toHaveValue('1.37');
    await button('Preview row edits (1)').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await cell.fill('1.41');
    await cell.press('Tab');
    await expect(button('Apply adjustment')).toHaveCount(0);
    await button('Preview row edits (1)').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await button('Discard preview').click();
    await expect(cell).toHaveValue('1.41');
    await button('Discard row edits').click();
    await button('Moving average').click();
    await expect(button('Apply adjustment')).toBeEnabled();
    await page.getByLabel('Smoothing window').fill('0');
    await expect(button('Apply adjustment')).toBeDisabled();
    await expect(page.locator('.auto-preview-status')).toContainText(
        'Enter a whole-number window from 1 to 25.',
    );
    await button('Discard preview').click();
    await button('Diagnostics').click();
    await button('Rate tables').click();
    await expect(page.locator('.rate-ae svg').first()).toBeVisible();
    const navigated = starts.length;
    await page.waitForTimeout(600);
    expect(starts).toHaveLength(navigated);
    expect(errors).toEqual([]);
});
