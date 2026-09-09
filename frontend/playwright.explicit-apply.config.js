import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({ ...base, testMatch: 'explicit-apply.spec.js', timeout: 120000 });
