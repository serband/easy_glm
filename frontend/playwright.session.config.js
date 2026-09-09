import { defineConfig } from '@playwright/test';
export default defineConfig({
    testDir: './tests',
    testMatch: 'session.spec.js',
    workers: 1,
    timeout: 30000,
    use: {
        baseURL: 'http://127.0.0.1:8773',
        headless: true,
        channel: process.env.PLAYWRIGHT_CHANNEL || undefined,
    },
});
