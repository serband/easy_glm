import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'exports.spec.js',
    timeout: 120000,
    webServer: { ...base.webServer, command: '../.venv/bin/python tests/export_server.py' },
});
