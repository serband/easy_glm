import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    testMatch: 'core-interactions.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8790' },
    webServer: {
        ...base.webServer,
        command: 'EASYGLM_CORE_INTERACTIONS_PORT=8790 ../.venv/bin/python tests/core_interactions_server.py',
        url: 'http://127.0.0.1:8790/health',
        reuseExistingServer: false,
    },
});
