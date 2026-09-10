import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';
export default defineConfig({
    ...base,
    testMatch: 'explore.spec.js',
    timeout: 30000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8776', viewport: { width: 884, height: 773 } },
    webServer: {
        ...base.webServer,
        env: { PYTHONPATH: '../tests:' + (process.env.EASYGLM_TEST_SOURCE || '../src') },
        command: `../.venv/bin/python -c "from desktop_onboarding_fixtures import install_example_loaders; install_example_loaders(); from easy_glm.desktop.__main__ import main; main()" --port 8776`,
        url: 'http://127.0.0.1:8776/health',
    },
});
