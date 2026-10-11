import { test, expect } from '@playwright/test';
import { mkdir, readFile } from 'node:fs/promises';

test.use({ viewport: { width: 1120, height: 900 } });

async function session(page) {
    const response = await page.request.get('/api/session');
    expect(response.ok()).toBeTruthy();
    return response.json();
}

async function revision(page, token) {
    const response = await page.request.get('/api/variables', {
        headers: { 'X-EasyGLM-Token': token },
    });
    expect(response.ok()).toBeTruthy();
    const snapshot = await response.json();
    return { session_id: snapshot.session_id, revision: snapshot.revision };
}

async function createAndFitStack(page) {
    const { token } = await session(page);
    const headers = { 'X-EasyGLM-Token': token };
    const save = await page.request.post('/api/models/save', {
        headers,
        data: {
            ...(await revision(page, token)),
            name: 'Stack export proof',
            create: true,
            fields: {
                family: 'poisson',
                link: 'log',
                target: 'Claims',
                weight: 'Exposure',
                components: [
                    {
                        component_id: 'main',
                        name: 'Main GLM',
                        kind: 'glm',
                        factors: ['DriverAge'],
                        settings: { alpha: 0.02 },
                    },
                    {
                        component_id: 'mileage-vehicle-pair',
                        name: 'Mileage × vehicle age',
                        kind: 'catboost',
                        factors: ['AnnualMileage', 'VehicleAge'],
                        settings: {
                            tune: false,
                            iterations: 8,
                            depth: 2,
                            learning_rate: 0.1,
                            l2_leaf_reg: 3,
                            seed: 42,
                            thread_count: 1,
                        },
                    },
                    {
                        component_id: 'vehicle',
                        name: 'Vehicle correction',
                        kind: 'catboost',
                        factors: ['Region'],
                        settings: {
                            tune: false,
                            iterations: 8,
                            depth: 2,
                            learning_rate: 0.1,
                            l2_leaf_reg: 3,
                            seed: 42,
                            thread_count: 1,
                        },
                    },
                ],
            },
        },
    });
    expect(save.ok(), await save.text()).toBeTruthy();
    const started = await page.request.post('/api/models/Stack%20export%20proof/fit', {
        headers,
        data: await revision(page, token),
    });
    expect(started.status(), await started.text()).toBe(202);
    await expect
        .poll(
            async () => {
                const response = await page.request.get('/api/jobs', { headers });
                const jobs = await response.json();
                return jobs['Stack export proof']?.status;
            },
            { timeout: 90000 },
        )
        .toBe('complete');
    return { token, headers };
}

async function download(page, testInfo, name, suffix) {
    const [file] = await Promise.all([
        page.waitForEvent('download'),
        page.getByRole('button', { name, exact: true }).click(),
    ]);
    expect(file.suggestedFilename()).toMatch(suffix);
    const path = testInfo.outputPath(file.suggestedFilename());
    await file.saveAs(path);
    expect(await file.failure()).toBeNull();
    return { filename: file.suggestedFilename(), source: (await readFile(path)).toString() };
}

test('exports separate frozen scorer and replayable stacked Python workflow without changing the fit', async ({
    page,
}, testInfo) => {
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    const { headers } = await createAndFitStack(page);
    const beforeProject = await (await page.request.get('/api/project', { headers })).json();
    const beforeJobs = await (await page.request.get('/api/jobs', { headers })).json();

    await page.goto('/?view=export');
    await expect(
        page.getByRole('heading', { name: 'Export stacked model', exact: true }),
    ).toBeVisible();
    await expect(page.getByText('Main GLM · GLM', { exact: true })).toBeVisible();
    await expect(page.getByText('Mileage × vehicle age · CatBoost', { exact: true })).toBeVisible();
    await expect(page.getByText('Vehicle correction · CatBoost', { exact: true })).toBeVisible();
    await expect(
        page.getByText('Download the fitted scorer or a workflow script for Stack export proof.', {
            exact: true,
        }),
    ).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Score new data', exact: true })).toBeVisible();
    await expect(
        page.getByRole('heading', { name: 'Rebuild this model', exact: true }),
    ).toBeVisible();
    await expect(
        page.getByRole('heading', { name: 'Stage rating files', exact: true }),
    ).toBeVisible();
    const exportCards = page.locator('.stack-export-card');
    await expect(exportCards).toHaveCount(3);
    const firstCard = await exportCards.nth(0).boundingBox();
    const secondCard = await exportCards.nth(1).boundingBox();
    const thirdCard = await exportCards.nth(2).boundingBox();
    expect(secondCard.y - (firstCard.y + firstCard.height)).toBeGreaterThanOrEqual(20);
    expect(thirdCard.y - (secondCard.y + secondCard.height)).toBeGreaterThanOrEqual(20);

    const scorer = await download(page, testInfo, 'Download Python scorer', /_scorer\.py$/);
    expect(scorer.source).toContain('StackScorer.from_dict');
    expect(scorer.source).toContain('def predict(frame: pl.DataFrame)');
    expect(scorer.source).not.toContain('run_model(');

    const workflow = await download(page, testInfo, 'Download Python workflow', /_workflow\.py$/);
    expect(workflow.source).toContain('Project.from_dict');
    expect(workflow.source).toContain('prepare(project');
    expect(workflow.source).toContain("run_model(project, prepared, 'Stack export proof')");
    expect(workflow.source).toContain('write_stack_bundle');
    expect(workflow.source).toContain('export_stack_excel');
    expect(workflow.source).toContain('SOURCE_PATH');
    expect(workflow.source).toContain('Main GLM');
    expect(workflow.source).toContain('Mileage × vehicle age');
    expect(workflow.source).toContain('Vehicle correction');
    expect(workflow.source).not.toContain('StackScorer.from_dict');

    expect(await (await page.request.get('/api/project', { headers })).json()).toEqual(
        beforeProject,
    );
    expect(await (await page.request.get('/api/jobs', { headers })).json()).toEqual(beforeJobs);
    expect(errors).toEqual([]);

    await mkdir('/private/tmp/easyglm-workflow-export-review', { recursive: true });
    await page.screenshot({
        path: '/private/tmp/easyglm-workflow-export-review/stack-workflow-export-desktop.png',
        fullPage: true,
    });
    await page.setViewportSize({ width: 390, height: 780 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(
        true,
    );
    await page.screenshot({
        path: '/private/tmp/easyglm-workflow-export-review/stack-workflow-export-narrow.png',
        fullPage: true,
    });
});
