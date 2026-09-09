import { defineConfig } from '@playwright/test';
export default defineConfig({
    testDir: './tests',
    testMatch: 'variables.spec.js',
    workers: 1,
    use: {
        baseURL: 'http://127.0.0.1:8770',
        headless: true,
        channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
        viewport: { width: 1440, height: 1000 },
    },
    webServer: {
        command: '../.venv/bin/python -m easy_glm.desktop --port 8770',
        url: 'http://127.0.0.1:8770/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
