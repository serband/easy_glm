import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({ ...base, testMatch: 'model-controls.spec.js', timeout: 30000 });
