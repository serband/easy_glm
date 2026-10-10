import { test, expect } from '@playwright/test';

const setupStep = (page, name) =>
    page
        .getByRole('navigation', { name: 'Project setup steps' })
        .getByRole('button', { name: new RegExp(name) });

async function startStages(page) {
    await page.goto('/');
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Model structure').click();
    page.once('dialog', (dialog) => dialog.accept());
    await page.getByRole('button', { name: 'Build in stages', exact: true }).click();
    await page.getByRole('button', { name: 'Continue to models', exact: true }).click();
}

async function chooseStageVariables(page, index) {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Variables').click();
    await page.getByLabel('Model stage').selectOption({ index });
}

async function openModels(page) {
    await page.getByRole('button', { name: 'Project & data', exact: true }).click();
    await setupStep(page, 'Models').click();
}

test('a CatBoost stage defaults to Optuna tuning on training folds', async ({ page }) => {
    await startStages(page);
    await page.getByLabel('Model 1 type').selectOption('catboost');
    await chooseStageVariables(page, 0);
    await page.getByLabel('Use VehicleAge').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Fit Model 1 — Main effects', exact: true }),
    ).toBeVisible();
    await expect(page.getByRole('radio', { name: 'Optuna tuning' })).toBeChecked();
    await expect(page.getByLabel('Optuna budget')).toHaveValue('trials');
    await expect(page.getByLabel('Optuna trials')).toHaveValue('8');
    await expect(
        page.getByText(
            'Optuna tuning uses training rows only. Holdout rows are not used. Every trial is evaluated across the selected training folds.',
        ),
    ).toBeVisible();

    await page.getByLabel('Optuna trials').fill('0');
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(page.getByRole('alert')).toContainText(
        'CatBoost trials must be a positive integer',
    );
    await page.getByLabel('Optuna trials').fill('8');
    await page.getByLabel('Optuna budget').selectOption('time');
    await expect(page.getByLabel('Tuning time minutes')).toHaveValue('15');
    await page.getByRole('radio', { name: 'Manual settings' }).check();
    await expect(page.getByLabel('Iterations')).toHaveValue('500');
    await expect(page.getByLabel('Depth')).toHaveValue('6');
});

test('each stage keeps its own variables and local numeric definitions', async ({ page }) => {
    await startStages(page);
    await chooseStageVariables(page, 0);
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: 'Use custom cuts', exact: true }).click();
    await page.getByLabel('Custom cuts for DriverAge').fill('25, 35, 45, 55');
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();

    await openModels(page);
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await chooseStageVariables(page, 1);
    await page.getByRole('tab', { name: 'JSON' }).click();
    const json = page.getByLabel('Model variables JSON');
    await json.fill(JSON.stringify({ variables: { Claims: { kind: 'step' } } }, null, 2));
    await expect(page.getByRole('alert')).toContainText('Claims is not available');
    await expect(
        page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }),
    ).toBeDisabled();

    await json.fill(
        JSON.stringify(
            {
                defaults: { n_bins: 6 },
                variables: { VehicleAge: { kind: 'step', knots: [1, 3, 6, 10] } },
            },
            null,
            2,
        ),
    );
    await page.getByRole('button', { name: 'Apply variable JSON', exact: true }).click();
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();
    await chooseStageVariables(page, 1);
    await page.getByRole('tab', { name: 'Table' }).click();
    await expect(page.getByLabel('Use VehicleAge')).toBeChecked();
    await expect(page.getByLabel('Use DriverAge')).not.toBeChecked();
});

test('a catalogue stage exposes uploaded attributes but not its selected key', async ({ page }) => {
    await startStages(page);
    const addedStage = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/model-plan') &&
            response.request().method() === 'POST' &&
            response.request().postDataJSON()?.plan?.stages?.length === 2,
    );
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await addedStage;
    const savedCatalogueStage = page.waitForResponse(
        (response) =>
            response.url().endsWith('/api/model-plan') &&
            response.request().method() === 'POST' &&
            response.request().postDataJSON()?.plan?.stages?.[1]?.kind === 'catalogue',
    );
    await page.getByLabel('Model 2 type').selectOption('catalogue');
    const catalogueResponse = await savedCatalogueStage;
    const catalogueBody = await catalogueResponse.json();
    expect(catalogueResponse.ok(), JSON.stringify(catalogueBody)).toBe(true);
    expect(catalogueBody.plan.stages[1].kind).toBe('catalogue');
    await chooseStageVariables(page, 1);
    await expect(page.getByRole('heading', { name: 'Full catalogue', exact: true })).toBeVisible();
    await page.getByLabel('Upload catalogue').setInputFiles({
        name: 'vehicle-file.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from('PolicyID,VehicleGroup,Power\n1,Family,90\n2,Sport,120\n'),
    });
    await page.getByLabel('Catalogue key').selectOption('PolicyID');
    await expect(page.getByLabel('Use PolicyID')).toHaveCount(0);
    await page.getByLabel('Use VehicleGroup').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();
    await chooseStageVariables(page, 1);
    await expect(page.getByText('1 selected', { exact: true })).toBeVisible();
});

test('a catalogue stage fits the uploaded file and remains available for table review', async ({
    page,
}) => {
    await startStages(page);
    await chooseStageVariables(page, 0);
    await page.getByLabel('Use DriverAge').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();

    await openModels(page);
    await page.getByRole('button', { name: 'Add model', exact: true }).click();
    await page.getByLabel('Model 2 type').selectOption('catalogue');
    await chooseStageVariables(page, 1);
    const csv = ['PolicyID,VehicleGroup,Power']
        .concat(
            Array.from(
                { length: 12000 },
                (_, index) => `${index},${index % 3 ? 'Family' : 'Sport'},${70 + (index % 90)}`,
            ),
        )
        .join('\n');
    await page.getByLabel('Upload catalogue').setInputFiles({
        name: 'vehicle-file.csv',
        mimeType: 'text/csv',
        buffer: Buffer.from(csv),
    });
    await page.getByLabel('Catalogue key').selectOption('PolicyID');
    await page.getByLabel('Use VehicleGroup').check();
    await page.getByRole('button', { name: /^Save variables for .+ and open Fit$/ }).click();

    await page.getByRole('button', { name: 'Model', exact: true }).click();
    await page.getByRole('button', { name: 'Fit Model 1 — Main effects', exact: true }).click();
    await expect(
        page.getByRole('heading', { name: 'Fit Model 2 — Model 2', exact: true }),
    ).toBeVisible({ timeout: 60000 });
    await page.getByRole('radio', { name: 'Manual settings' }).check();
    await page.getByLabel('Iterations').fill('20');
    await page.getByLabel('Threads').fill('1');
    await page.getByRole('button', { name: 'Fit Model 2 — Model 2', exact: true }).click();
    await expect(page.getByText('All models are fitted.', { exact: false })).toBeVisible({
        timeout: 120000,
    });

    await page.getByRole('button', { name: 'Rate tables', exact: true }).click();
    await expect(page.getByLabel('Inspect model component')).toBeVisible({ timeout: 60000 });
    await page.getByLabel('Inspect model component').selectOption({ label: 'Model 2' });
    await expect(
        page.getByRole('columnheader', { name: 'VehicleGroup', exact: true }),
    ).toBeVisible();
});
