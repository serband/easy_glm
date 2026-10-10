import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    testMatch: ['model-plan.spec.js', 'model-plan-edge.spec.js', 'variables.spec.js', 'single-plan-diagnostics.spec.js'],
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8786' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python -m easy_glm.desktop --demo --port 8786',
        url: 'http://127.0.0.1:8786/health',
        reuseExistingServer: false,
    },
});
