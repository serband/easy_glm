import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'screening.spec.js',
    timeout: 30000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8787', viewport: { width: 919, height: 773 } },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python tests/screening_server.py',
        url: 'http://127.0.0.1:8787/health',
    },
});
