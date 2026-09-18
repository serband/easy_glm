import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    testMatch: 'feature-selection.spec.js',
    timeout: 180000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8825' },
    webServer: {
        ...base.webServer,
        command: '../.venv/bin/python ../scripts/desktop_feature_selection_fixture.py --port 8825',
        url: 'http://127.0.0.1:8825/health',
    },
});
