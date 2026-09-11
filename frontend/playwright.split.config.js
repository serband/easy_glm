import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';
export default defineConfig({
    ...base,
    testMatch: 'variable-split.spec.js',
    timeout: 60000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8811' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python ../scripts/desktop_split_fixture.py --port 8811',
        url: 'http://127.0.0.1:8811/health',
    },
});
