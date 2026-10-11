import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    testMatch: 'model-variables-wide.spec.js',
    workers: 1,
    timeout: 120000,
    use: { baseURL: 'http://127.0.0.1:8771', headless: true, viewport: { width: 1332, height: 1000 } },
    webServer: {
        command: '../.venv/bin/python tests/model_variables_server.py',
        url: 'http://127.0.0.1:8771/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
