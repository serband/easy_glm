// Importance measures the immutable fit, so saved rate edits never invalidate it.
const results = new Map();
export function importanceCacheKey(session, model, fit, percentage = 30, seed = 42) {
    return JSON.stringify([session, model, fit, Number(percentage), Number(seed)]);
}
export function cachedImportance(key) {
    return results.get(key);
}
export function rememberImportance(key, result) {
    results.delete(key);
    results.set(key, result);
    while (results.size > 16) results.delete(results.keys().next().value);
}
