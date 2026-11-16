// The Query tab's results: a header (the question, the meta line and the timing; typed SQL is already in the
// editor and the inspector, so it isn't repeated), a banner when the verdict calls for one, then the grid, the
// big figure or the chart. Skeleton lines hold the place while the answer is on its way. With no job shown, it
// teaches the tab: example questions to ask, or a table to start from.
import { csvHref } from "./api.js";
import { chartFor } from "./chart.js";
import { banner, choice, JOB_FAILED_LEAD, skeleton, SQL_FAILED_LEAD, word } from "./controls.js";
import { aheadText, ASK_RECOVERY, count, EXAMPLES_HINT, examplesTitle, moreText, rowsRange, rowsText, SQL_RECOVERY, START_FROM_TABLE, START_HINT, startSql, STOPPED_QUEUED, } from "./copy.js";
import { fill, h, redraw, redrawFrom } from "./dom.js";
import { forgetGrids, grid, settleGrids } from "./grid.js";
import { askBlock, canStart, datasetLabel, isSql, showsSecond, } from "./state.js";
const PAGE = 500;
const VIEWS = [{ value: "table", text: "Table" }, { value: "chart", text: "Chart" }];
// A question heads its answer; typed SQL has no heading, since the editor above already shows it.
function heading(job) {
    if (isSql(job))
        return [];
    return [
        h("h1", { class: "rtitle" }, job.asked?.question ?? job.text),
        job.asked ? h("p", { class: "asked" }, `Asked again with the ${job.asked.reading} reading of ${job.asked.term}.`) : null,
    ];
}
// The times to the first answer and to the verdict, as the server words them, on a line of their own
// from the first answer on, so the meta line above never changes when the verdict lands.
// Typed SQL has none: its meta line already has the engine's time.
function timingLine(job) {
    const times = job.timing;
    if (isSql(job) || !times.first_text)
        return null;
    const judged = job.verdict?.kind === "clean" || job.verdict?.kind === "warned";
    const verdict = judged && times.verdict_text ? `, verdict in ${times.verdict_text}` : "";
    return `First answer in ${times.first_text}${verdict}.`;
}
// Whether the shown answer can switch between its table and its bars.
function charted(job) {
    return job.first?.chart.kind === "bars" && !(job.second && showsSecond(job));
}
// The header, or nothing when it would be empty (typed SQL before its result is in).
function header(job, state, actions) {
    const first = job.first;
    const timing = timingLine(job);
    const text = [...heading(job), first ? h("p", { class: "meta" }, first.meta) : null, first && timing ? h("p", { class: "meta" }, timing) : null];
    if (!text.some(Boolean))
        return null;
    return h("div", { class: "rhead" }, h("div", { class: "rhead-text" }, text), first && charted(job) ? choice("view", "Show the answer as", VIEWS, state.resultView, (view) => actions.setResultView(view)) : null);
}
function waiting(job) {
    const status = job.stage === "queued" ? [word("Queued"), " ", aheadText(job.ahead)] : [word(isSql(job) ? "Running" : "Writing SQL")];
    return [h("div", { class: "rbody" }, h("p", { class: "progress" }, status), h("div", { class: "placeholder" }, skeleton([40, 100, 92, 96, 85, 70])))];
}
// No numbers to show: the verdict's banner is the result, with any withheld SQL one click away.
function withheld(job, state, actions) {
    const verdict = job.verdict;
    if (!verdict)
        return [h("div", { class: "rbody" }, banner("error", STOPPED_QUEUED))];
    const sql = verdict.sql;
    const toggle = sql ? h("p", {}, h("button", {
        class: "link",
        attrs: { type: "button", "aria-expanded": String(state.sqlOpen), "aria-controls": "withheld-sql", "data-focus": "see-sql" },
        onClick: () => actions.toggleSql(),
    }, state.sqlOpen ? "Hide the SQL" : "See the SQL")) : null;
    return [h("div", { class: "rbody" }, banner(verdict.kind, verdict.text, toggle), sql && state.sqlOpen ? h("div", { class: "withheld", attrs: { id: "withheld-sql" } }, h("pre", { class: "sql", attrs: { "data-scroll": "sql-withheld" } }, sql), h("div", { class: "actions" }, h("button", { class: "btn2", attrs: { type: "button", "data-focus": "edit-withheld" }, onClick: () => actions.editSql(sql) }, "Edit and run"))) : null)];
}
// The pair box of the shown job, kept like its grids so they never leave the page between redraws.
let keptPair = null;
function pairBox(jobId) {
    if (keptPair?.job !== jobId)
        keptPair = { job: jobId, node: h("div") };
    return keptPair.node;
}
// Lets go of everything kept for the shown job's answers, for when another job is shown.
export function forgetAnswers() {
    keptPair = null;
    forgetGrids("job:");
}
function csvLink(jobId, answer, view) {
    return h("a", { class: "btn2", attrs: { href: csvHref(jobId, answer), download: `shilmandb-${jobId}.csv`, "data-focus": `csv-${answer}` } }, view.csv_label);
}
// The grid's footer: which rows are on screen, Show 500 more while the reply holds more, and the CSV.
function answerFoot(jobId, answer, view, onScreen, state, actions) {
    const more = Math.min(PAGE, view.shown - onScreen);
    const loading = state.loadingMore === answer;
    const waitingRows = state.loadingMore !== null;
    const failed = state.moreError?.answer === answer ? state.moreError.message : null;
    return [
        failed ? h("span", { class: "foot-note msg bad", attrs: { role: "alert" } }, failed) : h("span", { class: "foot-note" }, rowsRange(onScreen, view.row_count)),
        // Off while any rows load, but still focusable, so focus stays on it through the load.
        more > 0 ? h("button", {
            class: "btn2",
            attrs: { type: "button", "aria-disabled": waitingRows ? "true" : null, "data-focus": `more-${answer}` },
            onClick: () => { if (!waitingRows)
                actions.showMore(answer); },
        }, loading ? "Loading rows" : moreText(more)) : null,
        csvLink(jobId, answer, view),
    ];
}
function answerGrid(job, answer, view, state, actions) {
    const rows = [...view.rows, ...state.extra[answer]];
    const key = `job:${job.id}:${answer}`;
    return grid({
        key,
        label: `Rows of the ${answer} answer`,
        columns: view.columns,
        rows,
        onPick: (column, value) => actions.pickCell({ grid: key, column, value }),
        foot: answerFoot(job.id, answer, view, rows.length, state, actions),
    });
}
// A single value, shown once: the big figure with its column's name, the CSV in the footer.
function figureView(job, view) {
    return h("div", { class: "grid" }, h("div", { class: "figure-box" }, chartFor(view)), h("div", { class: "grid-foot" }, h("span", { class: "foot-note" }, rowsText(view.row_count)), csvLink(job.id, "first", view)));
}
// The answer's bars in place of its grid, the CSV still at hand.
function chartView(job, view) {
    return h("div", { class: "grid" }, h("div", { class: "chart-box" }, chartFor(view)), h("div", { class: "grid-foot" }, h("span", { class: "foot-note" }, rowsRange(view.rows.length, view.row_count)), csvLink(job.id, "first", view)));
}
// The first answer, or both side by side when the second differs. A banner (a warning, or the
// error a job hit after its first answer) sits under the header, above the numbers.
function answer(job, view, state, actions, notice = null) {
    const verdict = job.verdict;
    const arrived = state.changedFrom !== null && state.changedFrom.verdict?.kind !== verdict?.kind;
    const warned = verdict?.kind === "warned" ? banner("warned", verdict.text) : null;
    if (warned && arrived)
        warned.classList.add("arrive");
    const note = warned ?? notice;
    const parts = [note ? h("div", { class: "rnote" }, note) : null];
    const second = job.second;
    if (second && showsSecond(job)) {
        const pair = pairBox(job.id);
        pair.className = arrived ? "pair arrive" : "pair";
        fill(pair, h("h2", {}, "First answer"), h("p", { class: "asked" }, job.asked?.question ?? job.text), h("p", { class: "meta" }, view.meta), answerGrid(job, "first", view, state, actions), h("h2", {}, "Second answer"), h("p", { class: "asked" }, second.question ?? ""), h("p", { class: "meta" }, second.meta), answerGrid(job, "second", second, state, actions));
        parts.push(pair);
    }
    else if (view.chart.kind === "number" && view.columns.length === 1 && view.row_count === 1) {
        parts.push(figureView(job, view));
    }
    else if (view.chart.kind === "number") {
        parts.push(h("div", { class: "rfigure" }, chartFor(view)), answerGrid(job, "first", view, state, actions));
    }
    else if (charted(job) && state.resultView === "chart") {
        parts.push(chartView(job, view));
    }
    else {
        parts.push(answerGrid(job, "first", view, state, actions));
    }
    return parts;
}
// What went wrong and what to do, in the page's words: the banner's sentence and the live region's.
export function failedSentence(job) {
    return isSql(job) ? `${SQL_FAILED_LEAD} ${SQL_RECOVERY}` : `${JOB_FAILED_LEAD} ${ASK_RECOVERY}`;
}
// A job that ended in an error: the page's sentence, then the server's own message as the detail.
function failure(job) {
    const message = job.error?.message;
    return banner("error", failedSentence(job), message ? h("p", { class: "detail" }, `Details: ${message}`) : null);
}
// Example questions to ask, each one row you can click.
function examples(state, actions) {
    const questions = state.status?.examples[state.dataset] ?? [];
    const blocked = askBlock(state);
    const usable = blocked === null && canStart(state);
    // Why Ask is off is said once, under the input; here the rows just turn off.
    return h("div", { class: "starter" }, h("h2", {}, examplesTitle(datasetLabel(state))), h("ul", { class: "starter-list" }, questions.map((question, i) => h("li", {}, h("button", {
        class: "starter-row",
        attrs: { type: "button", disabled: !usable, "data-focus": `example-${i}` },
        onClick: () => actions.askExample(question),
    }, h("span", { class: "starter-text" }, question))))), h("p", { class: "hint" }, EXAMPLES_HINT));
}
// A first query for each of the database's tables; a click puts it in the editor.
function startTables(state, actions) {
    const tables = state.schema?.datasets[state.dataset]?.tables ?? [];
    return h("div", { class: "starter" }, h("h2", {}, START_FROM_TABLE), h("ul", { class: "starter-list" }, tables.map((table) => h("li", {}, h("button", {
        class: "starter-row",
        attrs: { type: "button", "data-focus": `start-${table.name}` },
        onClick: () => actions.loadSql(startSql(table.name)),
    }, h("span", { class: "starter-text code" }, startSql(table.name)), h("span", { class: "starter-note" }, `${count(table.rows)} rows`))))), tables.length ? h("p", { class: "hint" }, START_HINT) : null);
}
// With no job shown: Ask teaches with examples, SQL with the tables. With the model off, it's always SQL.
function emptyState(state, actions) {
    const mode = state.status?.model.state === "off" ? "sql" : state.mode;
    return mode === "ask" ? examples(state, actions) : startTables(state, actions);
}
export function drawResult(root, state, actions) {
    const job = state.job;
    root.setAttribute("aria-busy", String(job !== null && (job.stage === "queued" || job.stage === "writing")));
    if (!job) {
        redraw(root, state.jobError ? h("div", { class: "rbody" }, banner("error", state.jobError)) : emptyState(state, actions));
        return;
    }
    const first = job.first;
    redrawFrom(root, () => {
        let body;
        if (job.stage === "queued" || job.stage === "writing")
            body = waiting(job);
        // a first answer that came back before the error keeps its place, the banner over its numbers
        else if (job.stage === "error")
            body = first ? answer(job, first, state, actions, failure(job)) : [h("div", { class: "rbody" }, failure(job))];
        else if (first)
            body = answer(job, first, state, actions);
        else
            body = withheld(job, state, actions);
        return [header(job, state, actions), state.jobError ? h("p", { class: "msg bad rmsg", attrs: { role: "alert" } }, state.jobError) : null, body];
    });
    settleGrids(root);
}
