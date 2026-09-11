import { test, expect } from '@playwright/test';
import { readFile } from 'node:fs/promises';

async function client(page) {
    const session = await (await page.request.get('/api/session')).json();
    const headers = { 'X-EasyGLM-Token': session.token };
    const snapshot = await (await page.request.get('/api/variables', { headers })).json();
    return { headers, snapshot };
}
async function loadExample(page, secondModel = false) {
    let { headers, snapshot } = await client(page);
    const response = await page.request.post('/api/project/open', {
        headers,
        data: {
            session_id: snapshot.session_id,
            revision: snapshot.revision,
            kind: 'example',
            example: 'french_motor',
        },
    });
    expect(response.ok()).toBe(true);
    if (secondModel) {
        ({ headers, snapshot } = await client(page));
        const frequency = await page.request.post('/api/models/save', {
            headers,
            data: {
                session_id: snapshot.session_id,
                revision: snapshot.revision,
                name: 'Frequency',
                create: true,
                fields: {
                    target: 'ClaimNb',
                    weight: 'Exposure',
                    family: 'poisson',
                    predictors: ['DrivAge', 'Region'],
                    divide_target_by_weight: true,
                },
            },
        });
        expect(frequency.ok(), await frequency.text()).toBe(true);
        ({ headers, snapshot } = await client(page));
        const saved = await page.request.post('/api/models/save', {
            headers,
            data: {
                session_id: snapshot.session_id,
                revision: snapshot.revision,
                name: 'Weighted density',
                create: true,
                fields: {
                    target: 'Density',
                    weight: 'Exposure',
                    family: 'poisson',
                    predictors: ['DrivAge', 'Region'],
                    divide_target_by_weight: false,
                },
            },
        });
        expect(saved.ok(), await saved.text()).toBe(true);
    }
    await page.goto('/');
}
const explore = (page) => page.getByRole('button', { name: 'Explore', exact: true }).click();
const chart = (page, column) =>
    page.getByRole('img', { name: 'One-way effects of ' + column, exact: true });
const waitColumn = (page, column) =>
    page.waitForResponse(
        (r) =>
            r.url().includes('/api/explore?') &&
            new URL(r.url()).searchParams.get('column') === column,
    );

test('training observed rates share an aligned chart with exposure and update without fitting', async ({
    page,
}, info) => {
    await loadExample(page, true);
    const errors = [],
        requests = [];
    page.on('pageerror', (e) => errors.push(e.message));
    page.on('request', (r) => {
        if (r.url().includes('/api/explore?')) requests.push(r.url());
    });
    expect(requests).toHaveLength(0);
    const [numericResponse] = await Promise.all([
        page.waitForResponse((r) => r.url().includes('/api/explore?')),
        explore(page),
    ]);
    const numeric = await numericResponse.json();
    expect(numeric.column).toBe('DrivAge');
    await expect(page.getByLabel('Plot variable')).toHaveValue('DrivAge');
    await expect(chart(page, 'DrivAge')).toBeVisible();
    await expect(chart(page, 'DrivAge').locator('.observed-line')).toHaveCount(1);
    await expect(chart(page, 'DrivAge').locator('.exposure-bar')).toHaveCount(numeric.table.length);
    const alignment = await chart(page, 'DrivAge').evaluate((svg) => {
        const bars = [...svg.querySelectorAll('.exposure-bar')];
        return [...svg.querySelectorAll('.observed-point')].every(
            (point, i) =>
                Math.abs(
                    Number(point.getAttribute('cx')) -
                        Number(bars[i].getAttribute('x')) -
                        Number(bars[i].getAttribute('width')) / 2,
                ) < 1e-8,
        );
    });
    expect(alignment).toBe(true);
    await expect(page.locator('.response-basis')).toContainText('ClaimNb / Exposure');
    await expect(page.locator('.training-caption')).toContainText('Training');
    await expect(page.locator('.one-way-panel details')).not.toHaveAttribute('open', '');
    await page.locator('.one-way-panel summary').click();
    const [download] = await Promise.all([
        page.waitForEvent('download'),
        page.getByRole('button', { name: 'Download CSV', exact: true }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(/One.way.effects.*DrivAge.*csv/);
    const csv = await readFile(await download.path(), 'utf8');
    expect(csv.split('\n')).toHaveLength(numeric.table.length + 1);
    expect(csv).toContain('observed_rate');
    await page.locator('.one-way-panel summary').click();
    const [categoricalResponse] = await Promise.all([
        waitColumn(page, 'Region'),
        page.getByLabel('Plot variable').selectOption('Region'),
    ]);
    const categorical = await categoricalResponse.json();
    await expect(page.getByLabel('Bands', { exact: true })).toBeDisabled();
    await expect(chart(page, 'Region').locator('.observed-line')).toHaveCount(0);
    await expect(chart(page, 'Region').locator('.observed-point')).toHaveCount(
        categorical.table.length,
    );
    const cachedRequests = requests.length;
    await page.getByLabel('Plot variable').selectOption('DrivAge');
    await expect(chart(page, 'DrivAge')).toBeVisible();
    await page.getByLabel('Plot variable').selectOption('Region');
    await expect(chart(page, 'Region')).toBeVisible();
    expect(requests).toHaveLength(cachedRequests);
    await page.getByLabel('Plot variable').selectOption('DrivAge');
    const bandsRead = page.waitForResponse(
        (r) =>
            r.url().includes('/api/explore?') &&
            new URL(r.url()).searchParams.get('n_bins') === '5',
    );
    await page.getByLabel('Bands', { exact: true }).fill('5');
    const binned = await (await bandsRead).json();
    await expect(chart(page, 'DrivAge').locator('.exposure-bar')).toHaveCount(binned.table.length);
    const modelRead = page.waitForResponse(
        (r) =>
            r.url().includes('/api/explore?') &&
            new URL(r.url()).searchParams.get('model') === 'Weighted density',
    );
    await page.getByLabel('Explore model').selectOption('Weighted density');
    const model = await (await modelRead).json();
    expect(model.target).toBe('Density');
    expect(model.divide_target_by_weight).toBe(false);
    await expect(page.locator('.response-basis')).toContainText('Density');
    const count = requests.length;
    await page
        .getByRole('navigation', { name: 'Workbench' })
        .getByRole('button', { name: /^Variables/ })
        .click();
    const freshRead = page.waitForResponse((r) => r.url().includes('/api/explore?'));
    await explore(page);
    await freshRead;
    expect(requests).toHaveLength(count + 1);
    const { headers } = await client(page);
    expect(await (await page.request.get('/api/jobs', { headers })).json()).toEqual({});
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    await page.screenshot({ path: info.outputPath('numeric-one-way-884.png'), fullPage: true });
    expect(errors).toEqual([]);
});

test('late variable responses cannot replace the current selection', async ({ page }) => {
    await loadExample(page);
    await explore(page);
    await expect(page.getByLabel('Plot variable')).toBeEnabled();
    let release,
        held = false;
    await page.route('**/api/explore?**', async (route) => {
        if (new URL(route.request().url()).searchParams.get('column') === 'BonusMalus') {
            const response = await route.fetch();
            held = true;
            await new Promise((resolve) => {
                release = resolve;
            });
            await route.fulfill({ response });
        } else await route.continue();
    });
    await page.getByLabel('Plot variable').selectOption('BonusMalus');
    await expect.poll(() => held).toBe(true);
    await expect(page.locator('.one-way-chart')).toHaveCount(0);
    await page.getByLabel('Plot variable').selectOption('Region');
    await expect(chart(page, 'Region')).toBeVisible();
    const oldRead = waitColumn(page, 'BonusMalus');
    release();
    await oldRead;
    await expect(chart(page, 'Region')).toBeVisible();
    await expect(page.getByLabel('Plot variable')).toHaveValue('Region');
    await expect(chart(page, 'BonusMalus')).toHaveCount(0);
});

test('unassigned target still shows row exposure and a direct setup action', async ({
    page,
}, info) => {
    const { headers, snapshot } = await client(page);
    const query = new URLSearchParams({
        kind: 'data',
        filename: 'unassigned.csv',
        source_type: 'csv',
        session_id: snapshot.session_id,
        revision: String(snapshot.revision),
    });
    const response = await page.request.post('/api/project/upload?' + query, {
        headers: { ...headers, 'Content-Type': 'application/octet-stream' },
        data: Buffer.from('Amount,Age\n0,20\n1,30\n0,40\n'),
    });
    expect(response.ok()).toBe(true);
    await page.goto('/');
    await explore(page);
    await expect(page.getByLabel('Plot variable')).toBeEnabled();
    await page.getByLabel('Plot variable').selectOption('Age');
    await expect(chart(page, 'Age')).toBeVisible();
    await expect(chart(page, 'Age').locator('.observed-point')).toHaveCount(0);
    await expect(chart(page, 'Age').locator('.rate-tick')).toHaveCount(0);
    await expect(chart(page, 'Age').locator('.exposure-axis-label')).toHaveText('Rows');
    await expect(
        page.getByRole('button', { name: 'Assign target in Variables', exact: true }),
    ).toBeVisible();
    await page.screenshot({
        path: info.outputPath('exposure-without-target-884.png'),
        fullPage: true,
    });
    await page.getByRole('button', { name: 'Assign target in Variables', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
});

test('negative responses, missing-rate gaps, numeric null and single groups render honestly', async ({
    page,
}, info) => {
    await loadExample(page);
    let single = false;
    await page.route('**/api/explore?**', async (route) => {
        const response = await route.fetch();
        const fresh = await response.json();
        await route.fulfill({
            json: {
                ...fresh,
                column: 'Derived risk',
                columns: [{ name: 'Derived risk', kind: 'numeric' }],
                kind: 'numeric',
                target: 'Signed outcome',
                weight: 'Exposure',
                rate_label: 'Mean signed outcome',
                table: single
                    ? [{ label: 'Other / Unknown', exposure: 4, share: 1, rate: 2, order: null }]
                    : [
                          { label: '1', exposure: 1234567890.1234, share: 0.8, rate: -2, order: 1 },
                          { label: '2', exposure: 0, share: 0, rate: null, order: 2 },
                          { label: '3', exposure: 2, share: 0.1, rate: 3, order: 3 },
                          {
                              label: 'Other / Unknown',
                              exposure: 2,
                              share: 0.1,
                              rate: 4,
                              order: null,
                          },
                      ],
            },
        });
    });
    await explore(page);
    const svg = chart(page, 'Derived risk');
    await expect(svg).toBeVisible();
    await expect(page.getByLabel('Plot variable')).toHaveValue('Derived risk');
    await expect(svg.locator('.observed-point')).toHaveCount(3);
    await expect(svg.locator('.rate-tick').first()).toHaveText('-2');
    const path = await svg.locator('.observed-line').getAttribute('d');
    expect(path.match(/M/g)).toHaveLength(2);
    expect(path).not.toContain('L');
    const ticksFit = await svg.evaluate((node) =>
        [...node.querySelectorAll('.exposure-tick')].every(
            (tick) => tick.getBBox().x + tick.getBBox().width <= 800,
        ),
    );
    expect(ticksFit).toBe(true);
    await page.screenshot({ path: info.outputPath('negative-null-gaps-884.png'), fullPage: true });
    single = true;
    await page.getByLabel('Bands', { exact: true }).fill('5');
    await expect(svg.locator('.observed-point')).toHaveCount(1);
    await expect(svg.locator('.exposure-bar')).toHaveCount(1);
    await expect(svg.locator('.observed-line')).toHaveCount(0);
    const point = await svg.locator('.observed-point').getAttribute('cx');
    const bar = svg.locator('.exposure-bar');
    expect(Number(point)).toBeCloseTo(
        Number(await bar.getAttribute('x')) + Number(await bar.getAttribute('width')) / 2,
        8,
    );
});

test('long categorical labels stay separated through the final category', async ({
    page,
}, info) => {
    await loadExample(page);
    await page.setViewportSize({ width: 919, height: 773 });
    const names = [
        'Île-de-France',
        'Haute-Normandie',
        'Centre',
        'Basse-Normandie',
        'Bourgogne',
        'Nord-Pas-de-Calais',
        'Lorraine',
        'Alsace',
        'Pays de la Loire',
        'Bretagne',
        'Poitou-Charentes',
        'Aquitaine',
        'Midi-Pyrénées',
        'Limousin',
        'Rhône-Alpes',
        'Auvergne',
        'Languedoc-Roussillon',
        'Provence-Alpes-Côte d’Azur',
        'Corse',
        'Champagne-Ardenne',
        'Picardie',
        'Franche-Comté',
    ];
    await page.route('**/api/explore?**', async (route) => {
        const response = await route.fetch();
        const fresh = await response.json();
        await route.fulfill({
            json: {
                ...fresh,
                column: 'Region',
                kind: 'categorical',
                columns: [{ name: 'Region', kind: 'categorical' }],
                table: names.map((label, i) => ({
                    label,
                    exposure: 1000 - i * 10,
                    share: 1 / names.length,
                    rate: 0.08 + i * 0.005,
                    order: i,
                })),
            },
        });
    });
    await explore(page);
    const svg = chart(page, 'Region');
    await expect(svg).toBeVisible();
    await expect(svg.locator('.observed-point')).toHaveCount(22);
    await expect(svg.locator('.band-label').last().locator('title')).toHaveText('Franche-Comté');
    const separated = await svg.evaluate((node) => {
        const boxes = [...node.querySelectorAll('.band-label')].map((label) => label.getBBox());
        return boxes.every((box, i) => i === 0 || box.x >= boxes[i - 1].x + boxes[i - 1].width + 4);
    });
    expect(separated).toBe(true);
    await page.screenshot({
        path: info.outputPath('categorical-label-spacing-919.png'),
        fullPage: true,
    });
});

test('missing training split links directly to Variables', async ({ page }) => {
    await loadExample(page);
    await page.route('**/api/explore?**', (route) =>
        route.fulfill({ status: 422, json: { detail: "Split column 'train_test' is missing." } }),
    );
    await explore(page);
    await expect(page.getByRole('alert')).toContainText('Split column');
    await page.getByRole('button', { name: 'Set up train / holdout', exact: true }).click();
    await expect(page.getByRole('heading', { name: 'Variables', exact: true })).toBeVisible();
    await expect(page.getByRole('region', { name: 'Variable settings' })).toBeVisible();
});
