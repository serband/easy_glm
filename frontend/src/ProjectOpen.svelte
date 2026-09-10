<script>
    export let api;
    export let state;
    export let onOpened;
    let kind = 'data',
        path = '',
        pending = false,
        error = '',
        replace = false;
    let openedProject = state.project_id;
    let expanded = state.columns.length === 0;
    $: hasData = state.columns.length > 0;
    $: if (state.project_id !== openedProject) {
        openedProject = state.project_id;
        expanded = !hasData;
    }
    async function open() {
        if (pending || !path.trim() || (hasData && !replace)) return;
        pending = true;
        error = '';
        try {
            await api('project/open', {
                session_id: state.session_id,
                revision: state.revision,
                kind,
                path: path.trim(),
            });
            await onOpened();
            path = '';
            replace = false;
        } catch (e) {
            error = e.message;
        } finally {
            pending = false;
        }
    }
</script>

<details class="model-card project-open" bind:open={expanded}>
    <summary
        >{hasData ? 'Open another project or data file' : 'Open data or a saved project'}</summary
    >
    <form
        onsubmit={(event) => {
            event.preventDefault();
            void open();
        }}
    >
        <label
            >Open<select aria-label="Open file type" bind:value={kind} disabled={pending}>
                <option value="data">Data file</option>
                <option value="project">Saved project</option>
            </select></label
        >
        <label class="file-path"
            >File path<input
                bind:value={path}
                disabled={pending}
                placeholder={kind === 'data'
                    ? '/path/to/portfolio.parquet'
                    : '/path/to/project.easyglm-project.json'}
                autocomplete="off"
                required
            /></label
        >
        <p class="help-text">
            {kind === 'data'
                ? 'CSV, Parquet or Excel. Use a file on this computer.'
                : 'Open an exported project JSON. Its source data must still be available.'}
        </p>
        {#if hasData}<label class="replace-choice"
                ><input type="checkbox" bind:checked={replace} disabled={pending} />Replace this
                session. I have exported any work I want to keep.</label
            >{/if}
        {#if error}<p class="open-error" role="alert">{error}</p>{/if}
        <button class="primary" disabled={pending || !path.trim() || (hasData && !replace)}
            >{pending ? 'Opening…' : kind === 'data' ? 'Open data' : 'Open project'}</button
        >
    </form>
</details>

<style>
    summary {
        cursor: pointer;
        font-weight: 650;
    }
    form {
        display: grid;
        grid-template-columns: 180px minmax(0, 1fr);
        gap: 16px;
        padding-top: 20px;
    }
    label {
        display: grid;
        gap: 7px;
    }
    input,
    select {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    p,
    .replace-choice {
        grid-column: 1 / -1;
        margin: 0;
    }
    .replace-choice {
        display: flex;
        align-items: center;
        font-size: 14px;
    }
    .replace-choice input {
        width: auto;
    }
    button {
        justify-self: start;
        grid-column: 1 / -1;
    }
    .open-error {
        color: #a33b30;
    }
    @media (max-width: 650px) {
        form {
            grid-template-columns: 1fr;
        }
    }
</style>
