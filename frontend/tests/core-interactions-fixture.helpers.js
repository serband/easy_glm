import { expect } from '@playwright/test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

// A fresh, file-backed portfolio with two omitted supported pair effects. It lets
// the UI exercise discovery without API-seeding a plan or an accepted scorer.
export function useFreshCoreInteractionsProject(test) {
    let folder;
    test.beforeEach(async ({ request }) => {
        folder = await mkdtemp(path.join(tmpdir(), 'easyglm-core-interactions-browser-'));
        const dataPath = path.join(folder, 'data.csv');
        const rows = [
            'PolicyID,Claims,Exposure,DriverAge,VehicleAge,Region,AnnualMileage,InternalCode',
        ];
        let policyId = 0;
        for (const driverAge of [25, 55]) {
            for (const region of ['North', 'South']) {
                for (const mileage of ['low', 'high']) {
                    for (const vehicleAge of [1, 12]) {
                        const driverRate = driverAge === 55 ? 1.3 : 0.82;
                        const regionRate = region === 'South' ? 1.18 : 0.92;
                        const firstPair = mileage === 'high' && driverAge === 55 ? 2.6 : 0.72;
                        const secondPair = vehicleAge === 12 && region === 'South' ? 1.8 : 0.8;
                        const expected = driverRate * regionRate * firstPair * secondPair * 0.35;
                        for (let replicate = 0; replicate < 180; replicate += 1) {
                            // Deterministic balanced counts retain a clear residual signal on the
                            // training split while the browser still fits the model normally.
                            const claims = replicate < Math.round(expected * 10) ? 1 : 0;
                            rows.push(
                                `${policyId},${claims},0.1,${driverAge},${vehicleAge},${region},${mileage},${policyId % 20}`,
                            );
                            policyId += 1;
                        }
                    }
                }
            }
        }
        await writeFile(dataPath, rows.join('\n'));
        const projectPath = path.join(folder, 'project.json');
        await writeFile(
            projectPath,
            JSON.stringify({
                name: 'Core interactions browser fixture',
                data: {
                    source: { type: 'csv', path: dataPath },
                    split: { mode: 'random', fraction: 0.7, seed: 42 },
                    roles: {
                        PolicyID: 'id',
                        Claims: 'target',
                        Exposure: 'weight',
                        DriverAge: 'predictor',
                        VehicleAge: 'predictor',
                        Region: 'predictor',
                        AnnualMileage: 'predictor',
                        InternalCode: 'ignore',
                    },
                },
            }),
        );
        const session = await (await request.get('/api/session')).json();
        const headers = { 'X-EasyGLM-Token': session.token };
        const state = await (await request.get('/api/variables', { headers })).json();
        const opened = await request.post('/api/project/open', {
            headers,
            data: {
                session_id: state.session_id,
                revision: state.revision,
                kind: 'project',
                path: projectPath,
            },
        });
        expect(opened.ok(), await opened.text()).toBe(true);
    });
    test.afterEach(async () => {
        if (folder) await rm(folder, { recursive: true, force: true });
    });
}
