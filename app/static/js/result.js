// The result column: the question, the meta line, a banner when the verdict calls for one,
// then the chart and the table. Skeleton lines hold the place while the answer is on its way.
import { chartFor } from "./chart.js";
import { banner, JOB_FAILED_LEAD, skeleton, SQL_FAILED_LEAD, word } from "./controls.js";
import { aheadText, ASK_RECOVERY, EMPTY_RESULT, EMPTY_RESULT_SQL, SQL_RECOVERY, STOPPED_QUEUED } from "./copy.js";
import { fill, h, redraw, redrawFrom } from "./dom.js";
import { isSql, showsSecond } from "./state.js";
import { answerTable, forgetRows } from "./table.js";
function heading(job) {
    if (isSql(job))
        return [h("h1", { class: "question" }, job.kind === "edit" ? "Edited SQL" : "SQL")];
    return [
        h("h1", { class: "question" }, job.asked?.question ?? job.text),
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
function waiting(job) {
    const status = job.stage === "queued" ? [word("Queued"), " ", aheadText(job.ahead)] : [word(isSql(job) ? "Running" : "Writing SQL")];
    return [h("p", { class: "progress" }, status), h("div", { class: "placeholder" }, skeleton([40, 100, 92, 96, 85, 70]))];
}
// No numbers to show: the verdict's banner is the result, with any withheld SQL one click away.
function withheld(job, state, actions) {
    const verdict = job.verdict;
    if (!verdict)
        return [banner("error", STOPPED_QUEUED)];
    const sql = verdict.sql;
    const toggle = sql ? h("p", {}, h("button", {
        class: "link",
        attrs: { type: "button", "aria-expanded": String(state.sqlOpen), "aria-controls": "withheld-sql", "data-focus": "see-sql" },
        onClick: () => actions.toggleSql(),
    }, state.sqlOpen ? "Hide the SQL" : "See the SQL")) : null;
    return [
        banner(verdict.kind, verdict.text, toggle),
        sql && state.sqlOpen ? h("div", { class: "withheld", attrs: { id: "withheld-sql" } }, h("pre", { class: "sql", attrs: { "data-scroll": "sql-withheld" } }, sql), h("div", { class: "actions" }, h("button", { class: "btn2", attrs: { type: "button", "data-focus": "edit-withheld" }, onClick: () => actions.editSql(sql) }, "Edit and run"))) : null,
    ];
}
// The pair box of the shown job, kept like its tables so they never leave the page between redraws.
let keptPair = null;
function pairBox(jobId) {
    if (keptPair?.job !== jobId)
        keptPair = { job: jobId, node: h("div") };
    return keptPair.node;
}
// Lets go of everything kept for the shown job's answers, for when another job is shown.
export function forgetAnswers() {
    keptPair = null;
    forgetRows();
}
// The first answer, or both side by side when the second differs. A banner (a warning, or the
// error a job hit after its first answer) sits under the meta lines, above the numbers.
function answer(job, view, state, actions, notice = null) {
    const verdict = job.verdict;
    const arrived = state.changedFrom !== null && state.changedFrom.verdict?.kind !== verdict?.kind;
    const warned = verdict?.kind === "warned" ? banner("warned", verdict.text) : null;
    if (warned && arrived)
        warned.classList.add("arrive");
    const timing = timingLine(job);
    const parts = [
        h("div", { class: "about" }, h("p", { class: "meta" }, view.meta), timing ? h("p", { class: "meta" }, timing) : null),
        warned ?? notice,
    ];
    const second = job.second;
    if (second && showsSecond(job)) {
        const pair = pairBox(job.id);
        pair.className = arrived ? "pair arrive" : "pair";
        fill(pair, h("h2", {}, "First answer"), h("p", { class: "asked" }, job.asked?.question ?? job.text), h("p", { class: "meta" }, view.meta), answerTable(job.id, "first", view, state, actions), h("h2", {}, "Second answer"), h("p", { class: "asked" }, second.question ?? ""), h("p", { class: "meta" }, second.meta), answerTable(job.id, "second", second, state, actions));
        parts.push(pair);
    }
    else {
        parts.push(chartFor(view), answerTable(job.id, "first", view, state, actions));
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
export function drawResult(root, state, actions) {
    const job = state.job;
    root.setAttribute("aria-busy", String(job !== null && (job.stage === "queued" || job.stage === "writing")));
    if (!job) {
        const hint = state.status?.model.state === "off" ? EMPTY_RESULT_SQL : EMPTY_RESULT;
        redraw(root, state.jobError ? banner("error", state.jobError) : h("p", { class: "hint" }, hint));
        return;
    }
    const first = job.first;
    redrawFrom(root, () => {
        let body;
        if (job.stage === "queued" || job.stage === "writing")
            body = waiting(job);
        // a first answer that came back before the error keeps its place, the banner over its numbers
        else if (job.stage === "error")
            body = first ? answer(job, first, state, actions, failure(job)) : [failure(job)];
        else if (first)
            body = answer(job, first, state, actions);
        else
            body = withheld(job, state, actions);
        return [heading(job), state.jobError ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.jobError) : null, body];
    });
}
