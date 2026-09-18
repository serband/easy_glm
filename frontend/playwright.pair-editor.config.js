import base from './playwright.config.js';

export default {
    ...base,
    testMatch: ['pair-stages.spec.js', 'interactions.spec.js'],
};
