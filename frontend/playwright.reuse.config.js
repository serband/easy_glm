import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({ ...base, testMatch: 'main-effects-reuse.spec.js', timeout: 120000 });
