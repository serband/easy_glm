import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    testMatch: [
        'model-recovery.spec.js',
        'model-retry.spec.js',
        'stack-diagnostics.spec.js',
        'legacy-model-settings.spec.js',
    ],
    workers: 1,
    timeout: 120000,
    use: { baseURL: 'http://127.0.0.1:8787', headless: true, viewport: { width: 1332, height: 1000 } },
    webServer: {
        command: '../.venv/bin/python tests/legacy_model_server.py',
        url: 'http://127.0.0.1:8787/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
