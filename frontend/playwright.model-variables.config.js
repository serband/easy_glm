import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({
    ...base,
    testMatch: [
        'model-variables.spec.js',
        'model-flow.spec.js',
        'model-recovery.spec.js',
        'model-retry.spec.js',
        'stack-diagnostics.spec.js',
    ],
    timeout: 120000,
});
