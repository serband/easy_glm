import { test, expect } from '@playwright/test';

test('time role survives JSON and shows all-row chronological diagnostics and factor drill-down', async ({
    page,
}) => {
    const button = (name) => page.getByRole('button', { name, exact: true });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto('/');
    await page.getByRole('button', { name: /^Variables/ }).click();
    await expect(page.getByLabel('Role for Year', { exact: true })).toHaveValue('time');
    await button('Role JSON').click();
    const roles = JSON.parse(await page.getByLabel('Role JSON', { exact: true }).inputValue());
    expect(roles.time).toBe('Year');
    await button('Table').click();
    await button('Model').click();
    await button('Fit model').click();
    await expect(page.getByText('Fit complete', { exact: true })).toBeVisible({ timeout: 45000 });
    await button('Diagnostics').click();
    await page.getByRole('tab', { name: 'Time stability', exact: true }).click();
    await expect(
        page.getByRole('img', { name: 'Actual / expected over time', exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await expect(
        page.getByText('Year · 12,000 rows · 5 time bands', { exact: true }),
    ).toBeVisible();
    await page.getByText('Compare a factor across time', { exact: true }).click();
    await page.getByLabel('Time comparison factor').selectOption('DriverAge');
    await expect(
        page.getByRole('img', { name: 'DriverAge · A/E by period', exact: true }),
    ).toBeVisible({ timeout: 30000 });
    await page.getByLabel('Relative to the whole book in each period').check();
    await page.screenshot({ path: '/private/tmp/easyglm-time-stability.png', fullPage: true });
    await page.getByLabel('Time bands', { exact: true }).fill('3');
    await expect(page.getByText('Year · 12,000 rows · 3 time bands', { exact: true })).toBeVisible({
        timeout: 30000,
    });
    await expect(
        page.getByRole('img', { name: 'DriverAge · A/E by period', exact: true }),
    ).toBeVisible();
    await button('Project & data').click();
    await page.getByRole('radio', { name: 'Example dataset', exact: true }).check();
    for (const example of ['french_motor', 'swedish_motorcycle']) {
        await page
            .getByRole('combobox', { name: 'Example dataset', exact: true })
            .selectOption(example);
        await page
            .getByRole('checkbox', {
                name: 'Replace this session. I have saved any work I want to keep.',
                exact: true,
            })
            .check();
        await button('Load example').click();
        await expect(
            page.getByRole('heading', { name: 'Project & data', exact: true }),
        ).toBeVisible();
        await expect(button('Load example')).toBeVisible();
        const session = await (await page.request.get('/api/session')).json();
        const headers = { 'X-EasyGLM-Token': session.token };
        const project = await (await page.request.get('/api/project', { headers })).json();
        expect(project.models).toEqual({});
        expect(project.data.roles.SyntheticYear).toBe('time');
        expect(await (await page.request.get('/api/jobs', { headers })).json()).toEqual({});
    }
    expect(errors).toEqual([]);
});
