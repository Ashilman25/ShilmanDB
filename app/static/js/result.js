// The result column: the question, the meta line, a banner when the verdict calls for one,
// then the chart and the table. Skeleton lines hold the place while the answer is on its way.
import { chartFor } from "./chart.js";
import { banner, JOB_FAILED_LEAD, skeleton, SQL_FAILED_LEAD, word } from "./controls.js";
import { aheadText, ASK_RECOVERY, EMPTY_RESULT, SQL_RECOVERY, STOPPED_QUEUED } from "./copy.js";
import { h, redraw } from "./dom.js";
import { isSql, showsSecond } from "./state.js";
import { answerTable } from "./table.js";
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
        sql && state.sqlOpen ? h("div", { class: "withheld", attrs: { id: "withheld-sql" } }, h("pre", { class: "sql" }, sql), h("div", { class: "actions" }, h("button", { class: "btn2", attrs: { type: "button", "data-focus": "edit-withheld" }, onClick: () => actions.editSql(sql) }, "Edit and run"))) : null,
    ];
}
// The first answer, or both side by side when the second differs. A warned verdict's banner
// and the second answer's rows arrive together, in the same redraw.
function answer(job, view, state, actions) {
    const verdict = job.verdict;
    const arrived = state.changedFrom !== null && state.changedFrom.verdict?.kind !== verdict?.kind;
    const warned = verdict?.kind === "warned" ? banner("warned", verdict.text) : null;
    if (warned && arrived)
        warned.classList.add("arrive");
    const timing = timingLine(job);
    const parts = [
        h("div", { class: "about" }, h("p", { class: "meta" }, view.meta), timing ? h("p", { class: "meta" }, timing) : null),
        warned,
    ];
    const second = job.second;
    if (second && showsSecond(job)) {
        parts.push(h("div", { class: arrived ? "pair arrive" : "pair" }, h("h2", {}, "First answer"), h("p", { class: "asked" }, job.asked?.question ?? job.text), h("p", { class: "meta" }, view.meta), answerTable(job.id, "first", view, state, actions), h("h2", {}, "Second answer"), h("p", { class: "asked" }, second.question ?? ""), h("p", { class: "meta" }, second.meta), answerTable(job.id, "second", second, state, actions)));
    }
    else {
        parts.push(chartFor(view), answerTable(job.id, "first", view, state, actions));
    }
    return parts;
}
// A job that ended in an error: what went wrong and what to do, in the page's words, then the
// server's own message as the detail.
function failure(job) {
    const sentence = isSql(job) ? `${SQL_FAILED_LEAD} ${SQL_RECOVERY}` : `${JOB_FAILED_LEAD} ${ASK_RECOVERY}`;
    const message = job.error?.message;
    return banner("error", sentence, message ? h("p", { class: "detail" }, `Details: ${message}`) : null);
}
export function drawResult(root, state, actions) {
    const job = state.job;
    root.setAttribute("aria-busy", String(job !== null && (job.stage === "queued" || job.stage === "writing")));
    if (!job) {
        redraw(root, state.jobError ? banner("error", state.jobError) : h("p", { class: "hint" }, EMPTY_RESULT));
        return;
    }
    let body;
    if (job.stage === "queued" || job.stage === "writing")
        body = waiting(job);
    else if (job.stage === "error")
        body = [failure(job)];
    else if (job.first)
        body = answer(job, job.first, state, actions);
    else
        body = withheld(job, state, actions);
    redraw(root, heading(job), state.jobError ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.jobError) : null, body);
}
