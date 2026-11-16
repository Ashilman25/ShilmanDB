// The page: it holds the state, follows the shown job with the long poll, loads each open table's rows,
// and redraws a panel in full from the state whenever something it shows changes.
import * as api from "./api.js";
import { ApiError, isAbort } from "./api.js";
import { NO_SUCH_JOB, SCHEMA_RETRY, startSql, UNREACHABLE } from "./copy.js";
import { byId, resetScrolls } from "./dom.js";
import { drawEvidence } from "./evidence.js";
import { forgetGrid, revealColumn, settleGrids } from "./grid.js";
import { makeInput } from "./input.js";
import { drawResult, failedSentence, forgetAnswers } from "./result.js";
import { drawSidebar, revealRow } from "./sidebar.js";
import { askBlock, canStart, isRunning, isSql, MAX_TABLE_TABS, QUERY_TAB, queryBlock, runBlock, START_TABLES, tableKey, } from "./state.js";
import { drawStatusBar } from "./statusbar.js";
import { drawTablePane, PREVIEW_PAGE, revealColumnEntry } from "./tabletab.js";
import { drawTabs } from "./tabs.js";
import { applyTheme, savedTheme, saveTheme } from "./theme.js";
import { drawTested, drawTop } from "./topbar.js";
const STATUS_POLL_MS = 1000;
const RETRY_MS = 2000;
const SCHEMA_MAX_MS = 30_000;
const START_DATASET = "tpch";
const state = {
    status: null,
    statusError: null,
    schema: null,
    schemaError: null,
    dataset: START_DATASET,
    mode: "ask",
    theme: savedTheme(),
    testedOpen: false,
    sideTab: "tables",
    filter: "",
    expanded: new Set([`d:${START_DATASET}`]),
    tabs: [],
    active: QUERY_TAB,
    previews: new Map(),
    pick: null,
    history: [],
    job: null,
    changedFrom: null,
    jobError: null,
    submitError: null,
    submitting: false,
    stopping: false,
    extra: { first: [], second: [] },
    loadingMore: null,
    moreError: null,
    planOpen: false,
    sqlOpen: false,
    resultView: "table",
    editFrom: null,
};
const actions = {
    setDataset(name) {
        state.dataset = name;
        state.submitError = null;
        paint("ask");
        if (!state.job)
            paint("result");
    },
    setMode(mode) {
        modeChosen = true;
        state.mode = mode;
        state.submitError = null;
        paint("ask");
        if (!state.job)
            paint("result");
    },
    setTheme(theme) {
        state.theme = theme;
        saveTheme(theme);
        applyTheme(theme);
        paint("top");
    },
    toggleTested() {
        setTested(!state.testedOpen);
    },
    setSideTab(tab) {
        state.sideTab = tab;
        paint("sidebar");
    },
    setFilter(text) {
        state.filter = text;
        paint("sidebar");
    },
    toggleNode(key) {
        if (!state.expanded.delete(key))
            state.expanded.add(key);
        paint("sidebar");
    },
    openTable(dataset, table, column) {
        openTable(dataset, table, column);
    },
    closeTab(key) {
        const back = state.active === key;
        closeTab(key);
        paint("tabs", "table", "sidebar", "status");
        if (back)
            afterShow(state.active);
    },
    activate(key) {
        activate(key);
    },
    queryTable(dataset, table) {
        if (queryBlock(state, dataset) !== null)
            return;
        state.dataset = dataset;
        state.mode = "sql";
        modeChosen = true;
        state.editFrom = null;
        state.submitError = null;
        input.setSql(startSql(table));
        activate(QUERY_TAB);
        paint("ask");
        void submit();
    },
    showMorePreview(key) {
        const preview = state.previews.get(key);
        if (!preview?.data)
            return;
        const focused = document.activeElement instanceof HTMLElement && document.activeElement.dataset["focus"] === `more-${key}`;
        preview.shown += PREVIEW_PAGE;
        paint("table");
        // When the last page is in, Show more is gone; focus moves to the grid itself.
        if (focused && !byId("table-pane").querySelector(`[data-focus="more-${key}"]`)) {
            byId("table-pane").querySelector(".grid-table")?.focus({ preventScroll: true });
        }
    },
    pickColumn(key, column) {
        const preview = previewOf(key);
        preview.column = preview.column === column ? null : column;
        paint("table");
        if (preview.column)
            revealColumnEntry(byId("table-pane"), preview.column);
    },
    pickCell(pick) {
        state.pick = pick;
        paint("status");
    },
    openJob(id) {
        activate(QUERY_TAB);
        void follow(id, null);
    },
    askExample(question) {
        state.mode = "ask";
        input.setQuestion(question);
        void submit();
    },
    loadSql(sql) {
        state.mode = "sql";
        modeChosen = true;
        state.editFrom = null;
        state.submitError = null;
        input.setSql(sql);
        paint("ask", "result");
        input.focusSql();
    },
    submit() {
        void submit();
    },
    stop() {
        void stop();
    },
    reask(term, reading) {
        const job = state.job;
        if (job)
            void start(() => api.reask(job.id, term, reading));
    },
    editSql(sql) {
        const job = state.job;
        if (!job)
            return;
        state.mode = "sql";
        state.dataset = job.dataset;
        state.editFrom = job.kind === "sql" || job.kind === "edit" ? null : job;
        state.submitError = null;
        input.setSql(sql);
        activate(QUERY_TAB);
        paint("ask");
        input.focusSql();
    },
    togglePlan() {
        state.planOpen = !state.planOpen;
        paint("evidence");
    },
    toggleSql() {
        state.sqlOpen = !state.sqlOpen;
        paint("result");
    },
    setResultView(view) {
        state.resultView = view;
        paint("result");
    },
    showMore(answer) {
        void showMore(answer);
    },
};
const input = makeInput(actions);
// Whether you've switched the mode yourself, so the page never switches it on its own after that.
let modeChosen = false;
// The job this page just started from the input; the input keeps your text, not the server's copy.
let started = null;
// Bumped whenever another job is shown, so a page of rows asked for before that is dropped.
let switches = 0;
const panels = {
    top: () => drawTop(byId("top"), state, actions),
    tested: () => drawTested(byId("tested"), state, actions),
    sidebar: () => drawSidebar(byId("sidebar"), state, actions),
    tabs: () => {
        drawTabs(byId("tabs"), state, actions);
        byId("query-pane").hidden = state.active !== QUERY_TAB;
    },
    ask: () => input.draw(state),
    result: () => drawResult(byId("result"), state, actions),
    evidence: () => drawEvidence(byId("evidence"), state, actions),
    table: () => drawTablePane(byId("table-pane"), state, actions),
    status: () => drawStatusBar(byId("statusbar"), state),
};
// Whether a run could start when the table tab was last drawn: Query this table follows it.
let startable = true;
function paint(...names) {
    for (const name of names)
        panels[name]();
    if (canStart(state) !== startable) {
        startable = canStart(state);
        if (!names.includes("table"))
            panels.table();
    }
}
function focusKey(key) {
    document.querySelector(`[data-focus="${key}"]`)?.focus();
}
// Opens or shuts the tested sheet: focus goes into it on open and back to its link on close.
function setTested(open) {
    state.testedOpen = open;
    paint("top", "tested");
    if (open)
        focusKey("tested-close");
    else
        focusKey("tested-link");
}
function message(error) {
    return error instanceof ApiError ? error.message : UNREACHABLE;
}
// A call that never reached the server: recheck the status, whose error path says Not connected
// and retries. Nothing starts when a status call is already in flight or waiting.
function noticeGone(error) {
    if (error instanceof ApiError && error.status === 0 && !statusBusy)
        void loadStatus();
}
function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}
function previewOf(key) {
    let preview = state.previews.get(key);
    if (!preview) {
        preview = { data: null, error: null, lost: false, loading: false, shown: PREVIEW_PAGE, column: null };
        state.previews.set(key, preview);
    }
    return preview;
}
// Loads a table tab's rows once its dataset is ready. A call from a status poll never retries a failed load;
// opening the tab again does.
async function loadPreview(key, retry = true) {
    const tab = state.tabs.find((open) => open.key === key);
    if (!tab)
        return;
    const preview = previewOf(key);
    if (preview.data || preview.loading || (preview.error && !retry))
        return;
    if (state.status?.datasets[tab.dataset]?.state !== "ready")
        return;
    preview.loading = true;
    preview.error = null;
    preview.lost = false;
    if (state.active === key)
        paint("table");
    try {
        const data = await api.getPreview(tab.dataset, tab.table);
        if (state.previews.get(key) === preview)
            preview.data = data;
    }
    catch (error) {
        noticeGone(error);
        preview.error = message(error);
        preview.lost = error instanceof ApiError && error.status === 0;
    }
    preview.loading = false;
    if (state.active !== key)
        return;
    paint("table", "status");
    // the real rows widen the columns, so a column picked while they loaded comes back into view
    if (preview.data && preview.column)
        revealColumn(`table:${key}`, preview.column);
}
// Where the Query tab's own scrollers were when it was hidden: the browser drops those places while a pane is hidden.
const queryPlace = new Map();
function queryScrollers() {
    const checks = byId("evidence").querySelector(".ins-body");
    return [byId("result"), byId("query-pane"), ...(checks ? [checks] : [])];
}
function keepQueryPlace() {
    queryPlace.clear();
    for (const node of queryScrollers())
        queryPlace.set(node.id || node.className, { top: node.scrollTop, left: node.scrollLeft });
}
function restoreQueryPlace() {
    for (const node of queryScrollers()) {
        const kept = queryPlace.get(node.id || node.className);
        if (!kept)
            continue;
        node.scrollTop = kept.top;
        node.scrollLeft = kept.left;
    }
    settleGrids(byId("result"));
}
// Makes a tab the shown one, keeping the Query tab's place while it's away and putting it back on return.
function setActive(key) {
    if (state.active === key)
        return;
    if (state.active === QUERY_TAB)
        keepQueryPlace();
    state.active = key;
}
// Brings a tab into view once it's drawn: the Query tab where you left it, a table with its tree row in sight.
function afterShow(key) {
    if (key === QUERY_TAB)
        restoreQueryPlace();
    else
        revealRow(byId("sidebar"), `t:${key}`);
}
function activate(key) {
    if (state.active === key)
        return;
    setActive(key);
    paint("tabs", "table", "sidebar", "status");
    afterShow(key);
    if (key !== QUERY_TAB)
        void loadPreview(key);
}
// Takes a table's tab out of the strip, with its rows; the tab beside it takes over when it was shown,
// unless the caller shows another tab itself (takeover false), so no preview loads for nothing.
function closeTab(key, takeover = true) {
    const at = state.tabs.findIndex((tab) => tab.key === key);
    if (at < 0)
        return;
    state.tabs.splice(at, 1);
    state.previews.delete(key);
    forgetGrid(`table:${key}`);
    if (state.pick?.grid === `table:${key}`)
        state.pick = null;
    if (state.active !== key || !takeover)
        return;
    setActive(state.tabs[at]?.key ?? state.tabs[at - 1]?.key ?? QUERY_TAB);
    if (state.active !== QUERY_TAB)
        void loadPreview(state.active);
}
// Opens a table's tab (or shows the one already open), closing the oldest past eight; with a column, picks it.
function openTable(dataset, table, column) {
    const key = tableKey(dataset, table);
    if (!state.tabs.some((tab) => tab.key === key)) {
        if (state.tabs.length >= MAX_TABLE_TABS && state.tabs[0])
            closeTab(state.tabs[0].key, false);
        state.tabs.push({ key, dataset, table });
    }
    state.expanded.add(`d:${dataset}`);
    if (column)
        previewOf(key).column = column;
    const was = state.active;
    setActive(key);
    paint("tabs", "table", "sidebar", "status");
    revealRow(byId("sidebar"), `t:${key}`);
    if (column) {
        revealColumn(`table:${key}`, column);
        revealColumnEntry(byId("table-pane"), column);
    }
    if (was !== key || !previewOf(key).data)
        void loadPreview(key);
}
// The status line for the live region, said once per stage.
function announcement(job) {
    switch (job.stage) {
        case "queued": return "Queued.";
        case "writing": return job.kind === "sql" || job.kind === "edit" ? "Running." : "Writing SQL.";
        case "first": return "First answer back. The second is running.";
        case "done": return job.verdict?.text ?? "Done.";
        case "stopped": return "Stopped.";
        case "error": return failedSentence(job);
    }
}
// Puts the input back to what a job asked: questions in Ask mode with the question as typed
// (a re-ask's original), SQL in SQL mode, on the job's dataset.
function restoreInput(job) {
    state.dataset = job.dataset;
    state.editFrom = null;
    if (isSql(job)) {
        state.mode = "sql";
        input.setSql(job.text);
    }
    else {
        state.mode = "ask";
        input.setQuestion(job.asked?.question ?? job.text);
    }
}
// Draws a snapshot. A newer version of the shown job keeps what's on screen, the input included;
// another job starts clean, but one you just started from the input keeps your text there.
function show(job) {
    const before = state.job?.id === job.id ? state.job : null;
    if (!before) {
        switches++;
        forgetAnswers();
        resetScrolls(byId("result"));
        resetScrolls(byId("evidence"));
        state.submitError = null;
        state.extra = { first: [], second: [] };
        state.loadingMore = null;
        state.moreError = null;
        state.planOpen = false;
        state.sqlOpen = false;
        state.resultView = "table";
        state.stopping = false;
        if (state.pick?.grid.startsWith("job:"))
            state.pick = null;
        if (job.id === started)
            state.editFrom = null;
        else
            restoreInput(job);
        started = null;
        byId("result").scrollTop = 0;
    }
    if (!isRunning(job))
        state.stopping = false;
    if (before?.stage !== job.stage)
        byId("announce").textContent = announcement(job);
    state.job = job;
    state.jobError = null;
    state.changedFrom = before;
    paint("ask", "sidebar", "result", "evidence", "status");
    state.changedFrom = null;
    if (job.id === focusJob)
        focusShownSql(job);
}
// The job named in the URL at load. When it turns out to be SQL, the SQL editor takes the focus
// the question line had, unless you've already moved it.
let focusJob = null;
function focusShownSql(job) {
    focusJob = null;
    const active = document.activeElement;
    if (isSql(job) && (active === null || active === document.body || active === byId("question")))
        input.focusSql();
}
let following = 0;
let controller = null;
// Shows a job and keeps it current: each long poll sends the last version drawn and waits for a newer one.
async function follow(id, first) {
    const token = ++following;
    if (id !== focusJob)
        focusJob = null;
    controller?.abort();
    const own = new AbortController();
    controller = own;
    window.history.replaceState(null, "", `#job=${encodeURIComponent(id)}`);
    let after = -1;
    if (first) {
        show(first);
        after = first.version;
    }
    while (token === following) {
        let job;
        try {
            job = await api.getJob(id, after, own.signal);
        }
        catch (error) {
            if (token !== following || isAbort(error))
                return;
            if (error instanceof ApiError && error.status === 404) {
                forgetAnswers();
                focusJob = null;
                state.job = null;
                state.jobError = NO_SUCH_JOB;
                window.history.replaceState(null, "", window.location.pathname);
                paint("ask", "sidebar", "result", "evidence", "status");
                return;
            }
            noticeGone(error);
            const problem = message(error);
            if (problem !== state.jobError) {
                state.jobError = problem;
                paint("result");
            }
            await sleep(RETRY_MS);
            continue;
        }
        if (token !== following)
            return;
        if (job.version !== after || state.jobError)
            show(job);
        after = job.version;
        if (!isRunning(job)) {
            void loadHistory();
            return;
        }
    }
}
// Sends a new job (a question, SQL or a re-ask) and follows it; a refusal shows under the input.
async function start(send) {
    if (!canStart(state))
        return;
    state.submitting = true;
    state.submitError = null;
    paint("ask", "result", "evidence");
    const ticket = switches;
    try {
        const job = await send();
        state.submitting = false;
        if (job.kind === "sql" || job.kind === "edit")
            state.editFrom = null;
        // a re-ask isn't typed text, and a job opened meanwhile has taken the input, so both restore it
        started = job.kind !== "reask" && ticket === switches ? job.id : null;
        void loadHistory();
        void follow(job.id, job);
    }
    catch (error) {
        noticeGone(error);
        state.submitting = false;
        state.submitError = message(error);
        paint("ask", "result", "evidence");
    }
}
async function submit() {
    if (state.mode === "ask") {
        if (askBlock(state) === null)
            await start(() => api.ask(state.dataset, input.question()));
    }
    else if (runBlock(state) === null) {
        const from = state.editFrom;
        await start(() => api.runSql(state.dataset, input.sql(), from ? from.id : null));
    }
}
async function stop() {
    const job = state.job;
    if (!job || !isRunning(job))
        return;
    state.stopping = true;
    paint("ask");
    try {
        await api.stopJob(job.id);
    }
    catch (error) {
        noticeGone(error);
        state.stopping = false;
        state.submitError = message(error);
        paint("ask");
    }
}
// Fetches the next page of an answer's rows, starting after the rows on screen.
async function showMore(answer) {
    const job = state.job;
    const view = job?.[answer];
    if (!job || !view || state.loadingMore)
        return;
    const onScreen = view.rows.length + state.extra[answer].length;
    if (onScreen >= view.shown)
        return;
    const ticket = switches;
    state.loadingMore = answer;
    state.moreError = null;
    paint("result");
    try {
        const page = await api.getRows(job.id, answer, onScreen);
        if (ticket !== switches || state.job?.id !== job.id)
            return;
        state.extra[answer] = [...state.extra[answer], ...page.rows];
    }
    catch (error) {
        noticeGone(error);
        if (ticket !== switches || state.job?.id !== job.id)
            return;
        state.moreError = { answer, message: message(error) };
    }
    const focused = document.activeElement instanceof HTMLElement && document.activeElement.dataset["focus"] === `more-${answer}`;
    state.loadingMore = null;
    paint("result");
    if (focused)
        keepFocusAfterMore(answer);
}
// When the last page is in, Show more is gone; focus moves to the same grid's CSV link.
function keepFocusAfterMore(answer) {
    const result = byId("result");
    if (result.querySelector(`[data-focus="more-${answer}"]`))
        return;
    const csv = result.querySelector(`[data-focus="csv-${answer}"]`);
    if (csv instanceof HTMLElement)
        csv.focus({ preventScroll: true });
}
async function loadHistory() {
    try {
        state.history = await api.getHistory();
        paint("sidebar");
    }
    catch (error) {
        noticeGone(error); // the status bar then says the app can't be reached
    }
}
// Failed schema loads in a row, for the back-off.
let schemaTries = 0;
// The tables and columns of both datasets; they never change while the app runs, so this loads once.
// With the server gone it retries every 2 s; any other failure says so in the tree and backs off to 30 s.
async function loadSchema() {
    try {
        state.schema = await api.getSchema();
        state.schemaError = null;
        schemaTries = 0;
    }
    catch (error) {
        noticeGone(error);
        const lost = error instanceof ApiError && error.status === 0;
        if (!lost) {
            schemaTries++;
            state.schemaError = SCHEMA_RETRY;
            paint("sidebar");
        }
        setTimeout(() => void loadSchema(), lost ? RETRY_MS : Math.min(SCHEMA_MAX_MS, RETRY_MS * 2 ** (schemaTries - 1)));
        return;
    }
    paint("sidebar", "ask", "table");
    if (!state.job)
        paint("result");
}
function stillLoading() {
    const status = state.status;
    if (!status)
        return true;
    return status.model.state === "loading" || Object.values(status.datasets).some((info) => info.state === "loading");
}
// The last status reply, error and dataset drawn, so a poll that brings nothing new paints nothing.
let statusDrawn = null;
let statusSeen = false;
let statusBusy = false; // a status call is in flight or waiting to retry
// Started with --no-model, Ask can't work: on the first reply, an untouched page with no job opens in SQL.
function sqlWhenModelOff() {
    if (statusSeen || !state.status)
        return;
    statusSeen = true;
    if (state.status.model.state !== "off" || modeChosen || state.mode !== "ask" || state.job || jobFromHash())
        return;
    const focused = document.activeElement === byId("question");
    state.mode = "sql";
    paint("ask", "result");
    if (focused)
        input.focusSql();
}
// Previews that failed because the server was gone are forgotten, so the next load tries them again.
function forgetLostPreviews() {
    for (const preview of state.previews.values()) {
        if (!preview.lost)
            continue;
        preview.error = null;
        preview.lost = false;
    }
}
// Reads what's loaded; polls about once a second only while something is still loading.
async function loadStatus() {
    statusBusy = true;
    const wasGone = state.statusError !== null;
    try {
        state.status = await api.getStatus();
        state.statusError = null;
        if (!(state.dataset in state.status.datasets))
            state.dataset = Object.keys(state.status.datasets)[0] ?? state.dataset;
    }
    catch (error) {
        state.statusError = message(error);
    }
    if (wasGone && !state.statusError)
        forgetLostPreviews();
    const seen = JSON.stringify([state.status, state.statusError, state.dataset]);
    if (seen !== statusDrawn) {
        statusDrawn = seen;
        paint("tested", "sidebar", "ask", "evidence", "table", "status");
        if (!state.job)
            paint("result"); // the empty state follows the model's state
    }
    sqlWhenModelOff();
    if (state.active !== QUERY_TAB)
        void loadPreview(state.active, false);
    if (state.statusError || stillLoading())
        setTimeout(() => void loadStatus(), state.statusError ? RETRY_MS : STATUS_POLL_MS);
    else
        statusBusy = false;
}
function jobFromHash() {
    const found = /^#job=([^&]+)$/.exec(window.location.hash);
    if (!found?.[1])
        return null;
    try {
        return decodeURIComponent(found[1]);
    }
    catch {
        return null;
    }
}
// The first view: the start table's rows in a tab, its tree row selected and its database open.
// A job named in the URL shows in the Query tab instead, with the start table's tab beside it.
applyTheme(state.theme);
const startTable = START_TABLES[state.dataset];
if (startTable) {
    const key = tableKey(state.dataset, startTable);
    state.tabs.push({ key, dataset: state.dataset, table: startTable });
    state.active = key;
}
const shown = jobFromHash();
focusJob = shown;
if (shown)
    state.active = QUERY_TAB;
paint("top", "tested", "sidebar", "tabs", "ask", "result", "evidence", "table", "status");
void loadStatus();
void loadSchema();
void loadHistory();
if (shown)
    void follow(shown, null);
// Whether focus is in the tested sheet or on its link: only there does Escape shut it.
function fromSheet() {
    const active = document.activeElement;
    return active instanceof HTMLElement && (byId("tested").contains(active) || active.dataset["focus"] === "tested-link");
}
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.testedOpen && fromSheet()) {
        event.preventDefault();
        setTested(false);
    }
});
window.addEventListener("resize", () => { if (state.testedOpen)
    paint("tested"); });
window.addEventListener("hashchange", () => {
    const id = jobFromHash();
    if (id && id !== state.job?.id) {
        activate(QUERY_TAB);
        void follow(id, null);
    }
});
