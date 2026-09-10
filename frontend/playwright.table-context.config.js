import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'table-context.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8784' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python -m easy_glm.desktop --demo --port 8784',
        url: 'http://127.0.0.1:8784/health',
    },
});
