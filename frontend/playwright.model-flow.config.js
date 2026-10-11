import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';
export default defineConfig({ ...base, testMatch: 'model-flow.spec.js', timeout: 120000 });
