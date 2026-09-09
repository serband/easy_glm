// Old local servers know the worker bridge but reject a newly added action enum.
export function unsupportedImportanceAction(error) {
    try {
        const detail = JSON.parse(error.message);
        return (
            Array.isArray(detail) &&
            detail.some(
                (item) =>
                    item.type === 'literal_error' &&
                    item.input === 'importance' &&
                    JSON.stringify(item.loc) === JSON.stringify(['body', 'action']) &&
                    typeof item.ctx?.expected === 'string' &&
                    !/\bimportance\b/.test(item.ctx.expected),
            )
        );
    } catch {
        return false;
    }
}
