import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';
export default defineConfig({
    ...base,
    testMatch: 'binning.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8822' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python ../scripts/desktop_binning_fixture.py --port 8822',
        url: 'http://127.0.0.1:8822/health',
    },
});
