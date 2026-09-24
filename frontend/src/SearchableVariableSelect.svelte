<script>
    export let options = [];
    export let value = '';
    export let label = 'Variable';
    export let selectLabel = 'Variable';
    export let searchLabel = 'Search variables';
    export let disabled = false;
    export let contextKey = '';
    export let emptyLabel = 'No variables available';
    export let onChange = () => {};

    let query = '';
    let previousContext = contextKey;
    $: if (contextKey !== previousContext) {
        previousContext = contextKey;
        query = '';
    }
    $: sortedOptions = [...options].sort((a, b) => {
        const byName = (a.sortLabel || a.value).localeCompare(b.sortLabel || b.value, undefined, {
            sensitivity: 'base',
        });
        return byName || (a.value < b.value ? -1 : a.value > b.value ? 1 : 0);
    });
    $: needle = query.trim().toLocaleLowerCase();
    $: matches = sortedOptions.filter(
        (option) =>
            !needle ||
            option.value.toLocaleLowerCase().includes(needle) ||
            option.label.toLocaleLowerCase().includes(needle),
    );
    $: visibleOptions = sortedOptions.filter(
        (option) => option.value === value || matches.includes(option),
    );
</script>

<div class="searchable-variable">
    <div class="search-line">
        <label class="search-label"
            >{searchLabel}
            <input
                type="search"
                aria-label={searchLabel}
                bind:value={query}
                {disabled}
                onkeydown={(event) => {
                    if (event.key === 'Escape') query = '';
                }}
            />
        </label>
        {#if query}<button
                type="button"
                class="clear-search"
                aria-label={'Clear ' + searchLabel.toLowerCase()}
                {disabled}
                onclick={() => (query = '')}>Clear</button
            >{/if}
    </div>
    <label class="select-label"
        >{label}
        <select
            aria-label={selectLabel}
            bind:value
            {disabled}
            onchange={(event) => onChange(event.currentTarget.value)}
        >
            {#if !options.length}<option value={value}>{value || emptyLabel}</option>{/if}
            {#each visibleOptions as option (option.value)}<option value={option.value}
                    >{option.label}</option
                >{/each}
        </select>
    </label>
    {#if needle && !matches.length}<span class="no-matches" role="status"
            >No matching variables. Clear the search to see all options.</span
        >{/if}
</div>

<style>
    .searchable-variable {
        display: grid;
        gap: 7px;
        min-width: 0;
        width: 100%;
    }
    .search-line {
        display: flex;
        align-items: end;
        gap: 8px;
        min-width: 0;
    }
    .search-label,
    .select-label {
        display: grid;
        gap: 7px;
        min-width: 0;
        font-size: 11px;
        color: var(--muted);
    }
    .search-label {
        flex: 1;
    }
    input,
    select {
        box-sizing: border-box;
        width: 100%;
        min-width: 0;
    }
    .clear-search {
        flex: 0 0 auto;
    }
    .no-matches {
        font-size: 12px;
        color: var(--muted);
    }
    @media (min-width: 700px) {
        .searchable-variable {
            grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
            align-items: end;
        }
        .no-matches {
            grid-column: 1 / -1;
        }
    }
</style>
