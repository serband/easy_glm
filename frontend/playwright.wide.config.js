import { defineConfig } from '@playwright/test';
export default defineConfig({
    testDir: './tests',
    testMatch: 'wide.spec.js',
    workers: 1,
    use: {
        baseURL: 'http://127.0.0.1:8771',
        headless: true,
        channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
        viewport: { width: 1440, height: 1000 },
    },
    webServer: {
        command: '../.venv/bin/python ../scripts/desktop_wide_fixture.py',
        url: 'http://127.0.0.1:8771/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
