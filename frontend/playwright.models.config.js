import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';
export default defineConfig({ ...base, testMatch: 'models.spec.js', timeout: 60000 });
