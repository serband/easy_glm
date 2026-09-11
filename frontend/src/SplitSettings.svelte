<script>
    export let setup;
    export let state;
    export let api;
    export let onchange;
    export let disabled = false;
    let values = null,
        error = '',
        loading = false,
        generation = 0;
    $: context = JSON.stringify([
        state?.session_id,
        state?.revision,
        setup?.split?.mode,
        setup?.split?.column,
        setup?.assignments,
        setup?.renames,
        setup?.types,
    ]);
    $: loadValues(context);
    async function loadValues(key) {
        const request = ++generation;
        values = null;
        error = '';
        loading = false;
        if (!key || setup.split.mode !== 'column' || !setup.split.column) return;
        loading = true;
        try {
            const result = await api('variables/split-values', {
                session_id: state.session_id,
                revision: state.revision,
                setup,
            });
            if (request === generation) values = result;
        } catch (e) {
            if (request === generation) error = e.message;
        } finally {
            if (request === generation) loading = false;
        }
    }
    function chooseValue(encoded) {
        const train = encoded === '' ? null : JSON.parse(encoded);
        const holdout =
            values?.distinct === 2 && train != null
                ? values.values.find((item) => JSON.stringify(item.value) !== JSON.stringify(train))
                      ?.value
                : null;
        onchange({
            ...setup,
            split: { ...setup.split, train_value: train, holdout_value: holdout },
        });
    }
    $: valid = values?.distinct === 2 && !values?.missing;
    $: other =
        valid && setup.split.train_value != null
            ? values.values.find(
                  (item) => JSON.stringify(item.value) !== JSON.stringify(setup.split.train_value),
              )
            : null;
    function count(value) {
        return (
            values?.values.find((item) => JSON.stringify(item.value) === JSON.stringify(value))
                ?.rows || 0
        );
    }
</script>

<div class="split-row-values" aria-label={'Split values for ' + setup.assignments.split}>
    <fieldset {disabled}>
        <div class="split-fields">
            <label
                >Training value <span class="required">required</span>
                <select
                    aria-label="Variable training value"
                    disabled={!valid || loading}
                    value={setup.split.train_value == null
                        ? ''
                        : JSON.stringify(setup.split.train_value)}
                    onchange={(e) => chooseValue(e.currentTarget.value)}
                >
                    <option value="">Choose value…</option>
                    {#each values?.values || [] as item}
                        <option value={JSON.stringify(item.value)}
                            >{JSON.stringify(item.value)} · {item.rows.toLocaleString()} rows</option
                        >
                    {/each}
                </select>
            </label>
        </div>
    </fieldset>
    {#if loading}<p class="help-text">Reading split values…</p>
    {:else if error}<p role="alert">{error}</p>
    {:else if values}
        {#if values.distinct !== 2}<p role="alert">
                This column has {values.distinct.toLocaleString()} distinct values. A train/test column
                must have exactly two.
            </p>
        {:else if values.missing}<p role="alert">
                {values.missing.toLocaleString()} missing split values. Correct or filter these rows.
            </p>
        {:else if other && count(setup.split.train_value)}
            <p class="split-counts">
                {count(setup.split.train_value).toLocaleString()} training · {other.rows.toLocaleString()}
                holdout ({JSON.stringify(other.value)})
            </p>
        {:else}<p class="help-text">Choose training; the other value becomes holdout.</p>{/if}
    {/if}
</div>

<style>
    .split-row-values {
        height: 132px;
        box-sizing: border-box;
        padding: 10px 16px 8px;
        min-width: 730px;
        background: #f4f6f9;
        border-left: 3px solid #637d9a;
    }
    fieldset {
        border: 0;
        margin: 0;
        padding: 0;
        min-width: 0;
    }
    .split-fields {
        display: grid;
        grid-template-columns: minmax(0, 420px);
        gap: 16px;
    }
    label {
        font-size: 12px;
        min-width: 0;
    }
    .required {
        color: #69717d;
        font-size: 10px;
        margin-left: 6px;
    }
    select {
        display: block;
        width: 100%;
        margin-top: 5px;
        height: 30px;
    }
    p {
        font-size: 12px;
        margin: 6px 0 0;
    }
    .split-counts {
        font-variant-numeric: tabular-nums;
    }
    [role='alert'] {
        color: #aa392f;
    }
</style>
