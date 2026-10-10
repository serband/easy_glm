import { expect } from '@playwright/test';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

// Each scenario starts with the same data and assignments. The server is shared
// across files, so replacing only the working plan would leak role edits/fits.
export function useFreshModelPlanProject(test) {
    let folder;
    test.beforeEach(async ({ request }) => {
        folder = await mkdtemp(path.join(tmpdir(), 'easyglm-plan-browser-'));
        const dataPath = path.join(folder, 'data.csv');
        const rows = ['PolicyID,Claims,Exposure,DriverAge,VehicleAge,Region,AnnualMileage,InternalCode'];
        for (let i = 0; i < 360; i += 1) {
            rows.push(`${i},${i % 7 === 0 ? 1 : 0},0.8,${18 + i % 72},${i % 25},${['North', 'South', 'East', 'West'][i % 4]},${5000 + i * 17},${i % 50}`);
        }
        await writeFile(dataPath, rows.join('\n'));
        const projectPath = path.join(folder, 'project.json');
        await writeFile(projectPath, JSON.stringify({
            name: 'Motor portfolio · synthetic sample',
            data: {
                source: { type: 'csv', path: dataPath },
                split: { mode: 'random', fraction: 0.7, seed: 42 },
                roles: {
                    PolicyID: 'id', Claims: 'target', Exposure: 'weight',
                    DriverAge: 'predictor', VehicleAge: 'predictor',
                    Region: 'predictor', AnnualMileage: 'predictor', InternalCode: 'ignore',
                },
            },
        }));
        const session = await (await request.get('/api/session')).json();
        const headers = { 'X-EasyGLM-Token': session.token };
        const state = await (await request.get('/api/variables', { headers })).json();
        const opened = await request.post('/api/project/open', {
            headers,
            data: {
                session_id: state.session_id, revision: state.revision,
                kind: 'project', path: projectPath,
            },
        });
        expect(opened.ok(), await opened.text()).toBe(true);
    });
    test.afterEach(async () => {
        if (folder) await rm(folder, { recursive: true, force: true });
    });
}
