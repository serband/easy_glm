import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({ ...base, testMatch: 'ae-cache.spec.js', timeout: 120000 });
