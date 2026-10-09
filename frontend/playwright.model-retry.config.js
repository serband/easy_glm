import base from './playwright.config.js';
import { defineConfig } from '@playwright/test';

export default defineConfig({ ...base, testMatch: 'model-retry.spec.js', timeout: 120000 });
