import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'time-diagnostics.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8814' },
    webServer: {
        ...base.webServer,
        command:
            '../.venv/bin/python ../scripts/desktop_time_fixture.py --port 8814 --mock-examples',
        url: 'http://127.0.0.1:8814/health',
    },
});
