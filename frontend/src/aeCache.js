// Revision includes every model adjustment; fit identities cover refits.
const packets = new Map();
export function rememberAe(key, packet) {
    packets.delete(key);
    packets.set(key, packet);
    while (packets.size > 8) packets.delete(packets.keys().next().value);
}
export function cachedAe(key, variable, subset, both) {
    const packet = packets.get(key),
        item = packet?.variables[variable];
    if (!item?.subsets[subset]) return null;
    return {
        rows: item.subsets[subset],
        kind: item.kind,
        subset,
        book_impact: packet.book_impact,
        ae_sets: ['train', 'holdout']
            .filter((s) => both && s !== subset && item.subsets[s])
            .map((s) => ({ title: `${variable} · ${s}`, rows: item.subsets[s] })),
    };
}
