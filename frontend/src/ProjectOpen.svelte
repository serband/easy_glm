<script>
    export let api;
    export let upload;
    export let state;
    export let onOpened;
    export let onReconnect;
    let kind = 'data',
        selection = 'upload',
        sourceType = 'auto',
        path = '',
        file = null,
        example = 'french_motor',
        pending = false,
        error = '',
        needsReconnect = false,
        replace = false;
    const choices = [
        ['data', 'Own data'],
        ['project', 'Saved project'],
        ['example', 'Example dataset'],
    ];
    const formats = [
        ['auto', 'Automatic — from file name'],
        ['csv', 'CSV'],
        ['parquet', 'Parquet'],
        ['excel', 'Excel'],
        ['ipc', 'Arrow / Feather'],
        ['sas7bdat', 'SAS'],
    ];
    let openedProject = state.project_id;
    $: hasData = state.columns.length > 0;
    $: ready =
        kind === 'example' || (selection === 'upload' ? Boolean(file) : Boolean(path.trim()));
    $: if (state.project_id !== openedProject) {
        openedProject = state.project_id;
        file = null;
        path = '';
        replace = false;
        error = '';
    }
    function changeSource() {
        file = null;
        path = '';
        sourceType = 'auto';
        replace = false;
        error = '';
        needsReconnect = false;
    }
    async function reconnect() {
        pending = true;
        replace = false;
        try {
            await onReconnect();
            error = '';
            needsReconnect = false;
        } catch (e) {
            error = e.message;
        } finally {
            pending = false;
        }
    }
    async function open() {
        if (pending || !ready || (hasData && !replace)) return;
        pending = true;
        error = '';
        needsReconnect = false;
        const openingKind = kind;
        try {
            const body = {
                session_id: state.session_id,
                revision: state.revision,
                kind,
                source_type: kind === 'data' ? sourceType : 'auto',
            };
            let opened;
            if (kind === 'example') {
                opened = await api('project/open', { ...body, example });
            } else if (selection === 'upload') {
                if (file.size > 512 * 1024 * 1024)
                    throw new Error('Files above 512 MB must be opened using their file path.');
                opened = await upload(file, body);
            } else {
                opened = await api('project/open', { ...body, path: path.trim() });
            }
            await onOpened(opened, openingKind);
            file = null;
            path = '';
            replace = false;
        } catch (e) {
            error = e.message;
            needsReconnect = e.status === 409;
            replace = false;
        } finally {
            pending = false;
        }
    }
</script>

<section class="model-card project-open" aria-label="Open data">
    <h2>Open data</h2>
    <form
        onsubmit={(event) => {
            event.preventDefault();
            void open();
        }}
    >
        <fieldset class="source-choices" disabled={pending}>
            <legend>Start with</legend>
            <div>
                {#each choices as [value, label]}
                    <label class:selected={kind === value}>
                        <input
                            type="radio"
                            name="project-source"
                            {value}
                            bind:group={kind}
                            onchange={changeSource}
                        />
                        <span>{label}</span>
                    </label>
                {/each}
            </div>
        </fieldset>
        {#if kind === 'example'}
            <label
                >Example dataset
                <select
                    bind:value={example}
                    disabled={pending}
                    onchange={() => {
                        replace = false;
                        error = '';
                    }}
                >
                    <option value="french_motor">Motor insurance — claim frequency</option>
                    <option value="swedish_motorcycle">Motorcycle insurance — claims cost</option>
                </select>
            </label>
            <p class="help-text">
                {example === 'french_motor'
                    ? 'Poisson · Claim count (ClaimNb), weighted by exposure.'
                    : 'Tweedie · Total claim cost (ClaimAmount), including zero claims.'}
                Roles are preset; create your model when ready. SyntheticYear contains made-up years for
                trying time diagnostics, not actual claims history.
            </p>
        {:else}
            <div class="file-options">
                <label
                    >File selection
                    <select bind:value={selection} disabled={pending} onchange={changeSource}>
                        <option value="upload">Browse files</option>
                        <option value="path">Use file path</option>
                    </select>
                </label>
                {#if kind === 'data'}
                    <label
                        >Data format
                        <select
                            bind:value={sourceType}
                            disabled={pending}
                            onchange={() => {
                                replace = false;
                                error = '';
                            }}
                        >
                            {#each formats as [value, label]}<option {value}>{label}</option>{/each}
                        </select>
                    </label>
                {/if}
            </div>
            {#if selection === 'upload'}
                {#key kind + ':' + openedProject}
                    <label
                        >{kind === 'data' ? 'Data file' : 'Project file'}
                        <input
                            type="file"
                            accept={kind === 'project' ? '.json' : undefined}
                            disabled={pending}
                            onchange={(event) => {
                                file = event.currentTarget.files?.[0] || null;
                                replace = false;
                                error = '';
                            }}
                        />
                    </label>
                {/key}
            {:else}
                <label
                    >File path
                    <input
                        bind:value={path}
                        disabled={pending}
                        placeholder={kind === 'data'
                            ? '/path/to/portfolio.csv'
                            : '/path/to/project.json'}
                        autocomplete="off"
                        oninput={() => {
                            replace = false;
                            error = '';
                        }}
                    />
                </label>
            {/if}
            {#if kind === 'project'}<p class="help-text">
                    Open a project JSON. Its source data must still be available.
                </p>{/if}
        {/if}
        {#if hasData}
            <label class="replace-choice"
                ><input type="checkbox" bind:checked={replace} disabled={pending} />Replace this
                session. I have saved any work I want to keep.</label
            >
        {/if}
        {#if error}<p class="open-error" role="alert">{error}</p>{/if}
        {#if needsReconnect}<button type="button" onclick={reconnect} disabled={pending}
                >Reconnect</button
            >{/if}
        <button class="primary" disabled={pending || !ready || (hasData && !replace)}>
            {pending
                ? 'Opening…'
                : kind === 'example'
                  ? 'Load example'
                  : kind === 'project'
                    ? 'Open project'
                    : 'Open data'}
        </button>
    </form>
</section>

<style>
    h2 {
        margin: 0 0 18px;
    }
    form {
        display: grid;
        gap: 16px;
    }
    label {
        display: grid;
        gap: 7px;
        min-width: 0;
    }
    input,
    select {
        width: 100%;
        min-width: 0;
        box-sizing: border-box;
    }
    .source-choices {
        margin: 0;
        padding: 0;
        border: 0;
        min-width: 0;
    }
    .source-choices legend {
        margin-bottom: 8px;
        color: var(--muted, #617d72);
        font-size: 14px;
    }
    .source-choices > div {
        display: flex;
        gap: 20px;
        border-bottom: 1px solid #d6e0dc;
    }
    .source-choices label {
        position: relative;
        padding: 9px 0 10px;
        cursor: pointer;
        border-bottom: 3px solid transparent;
    }
    .source-choices label.selected {
        color: #1f705c;
        border-bottom-color: #1f705c;
        font-weight: 650;
    }
    .source-choices label:focus-within {
        outline: 2px solid #1f705c;
        outline-offset: 3px;
    }
    .source-choices input {
        position: absolute;
        opacity: 0;
        width: 1px;
        height: 1px;
    }
    .file-options {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 16px;
    }
    input[type='file'] {
        padding: 9px;
        border: 1px solid #cbd8d2;
        border-radius: 5px;
        background: white;
    }
    input[type='file']::file-selector-button {
        padding: 7px 12px;
        margin-right: 12px;
        border: 1px solid #cbd8d2;
        border-radius: 4px;
        background: #edf3f0;
        color: #25473d;
        cursor: pointer;
    }
    p {
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
    }
    .open-error {
        color: #a33b30;
    }
    @media (max-width: 550px) {
        .file-options {
            grid-template-columns: 1fr;
        }
        .source-choices > div {
            gap: 12px;
        }
        .source-choices label {
            font-size: 14px;
        }
    }
</style>
