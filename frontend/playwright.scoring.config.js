import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    testMatch: 'scoring.spec.js',
    workers: 1,
    use: {
        baseURL: 'http://127.0.0.1:8894',
        headless: true,
        viewport: { width: 1440, height: 1000 },
    },
    webServer: {
        env: {
            PYTHONPATH: process.env.EASYGLM_TEST_SOURCE || '../src',
            EASYGLM_SCORING_PORT: '8894',
            OMP_NUM_THREADS: '1',
            OPENBLAS_NUM_THREADS: '1',
        },
        command: '../.venv/bin/python tests/scoring_server.py',
        url: 'http://127.0.0.1:8894/health',
        reuseExistingServer: false,
        timeout: 60000,
    },
});
