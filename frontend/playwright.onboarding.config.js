import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'onboarding.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8775', viewport: { width: 919, height: 773 } },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python -m easy_glm.desktop --port 8775',
        url: 'http://127.0.0.1:8775/health',
    },
});
