import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    testMatch: 'guided-setup.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8789' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python -m easy_glm.desktop --demo --port 8789',
        url: 'http://127.0.0.1:8789/health',
        reuseExistingServer: false,
    },
});
