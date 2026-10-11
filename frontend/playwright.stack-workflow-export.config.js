import { defineConfig } from '@playwright/test';
import base from './playwright.config.js';

export default defineConfig({
    ...base,
    testMatch: 'stack-workflow-export.spec.js',
    timeout: 120000,
    use: { ...base.use, baseURL: 'http://127.0.0.1:8790' },
    webServer: {
        ...base.webServer,
        command:
            'EASYGLM_STACK_WORKFLOW_EXPORT_PORT=8790 ../.venv/bin/python tests/stack_workflow_export_server.py',
        url: 'http://127.0.0.1:8790/health',
        reuseExistingServer: false,
    },
});
