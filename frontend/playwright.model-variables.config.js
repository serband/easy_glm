import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8788' },
    testMatch: [
        'model-variables.spec.js',
        'model-flow.spec.js',
    ],
    timeout: 120000,
    webServer: {
        command: '../.venv/bin/python -m easy_glm.desktop --demo --port 8788',
        url: 'http://127.0.0.1:8788/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
