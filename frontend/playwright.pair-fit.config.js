import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    testMatch: 'pair-fit.spec.js',
    workers: 1,
    timeout: 240000,
    use: {
        baseURL: 'http://127.0.0.1:8831',
        headless: true,
        channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
        viewport: { width: 1440, height: 1000 },
    },
    webServer: {
        command: '../.venv/bin/python tests/pair_server.py',
        env: { PYTHONPATH: '../src' },
        url: 'http://127.0.0.1:8831/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
