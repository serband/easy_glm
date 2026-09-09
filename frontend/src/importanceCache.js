// Importance measures the immutable fit, so saved rate edits never invalidate it.
const results = new Map();
export function importanceCacheKey(session, model, fit) {
    return JSON.stringify([session, model, fit]);
}
export function cachedImportance(key) {
    return results.get(key);
}
export function rememberImportance(key, result) {
    results.delete(key);
    results.set(key, result);
    while (results.size > 16) results.delete(results.keys().next().value);
}
