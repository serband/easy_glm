import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({...base, testMatch:'reviews.spec.js', timeout:90000});
