// The left rail: this session's history, newest first, then the current dataset's examples.
import { EMPTY_HISTORY } from "./copy.js";
import { h, redraw } from "./dom.js";
import { askBlock, canStart, datasetLabel } from "./state.js";
function firstLine(text) {
    return text.trim().split("\n", 1)[0] ?? "";
}
// How a history entry reads: questions as asked, SQL with what kind it was.
function entryText(entry) {
    if (entry.kind === "sql")
        return `SQL: ${firstLine(entry.text)}`;
    if (entry.kind === "edit")
        return `Edited SQL: ${firstLine(entry.text)}`;
    return entry.text;
}
function historyItem(entry, state, actions) {
    const current = state.job?.id === entry.id;
    const other = entry.dataset !== state.dataset ? datasetLabel(state, entry.dataset) : null;
    return h("li", {}, h("button", {
        class: current ? "item on" : "item",
        attrs: { type: "button", "aria-current": current ? "true" : null, title: entry.text, "data-focus": `job-${entry.id}` },
        onClick: () => actions.openJob(entry.id),
    }, h("span", { class: "clamp" }, entryText(entry)), other ? h("span", { class: "item-note" }, other) : null));
}
export function drawRail(root, state, actions) {
    const examples = state.status?.examples[state.dataset] ?? [];
    const blocked = askBlock(state);
    const usable = blocked === null && canStart(state);
    redraw(root, h("h2", {}, "History"), state.history.length
        ? h("ul", { class: "items" }, state.history.map((entry) => historyItem(entry, state, actions)))
        : h("p", { class: "hint" }, EMPTY_HISTORY), h("h2", {}, "Examples"), blocked && state.status && !state.statusError ? h("p", { class: "hint" }, blocked) : null, h("ul", { class: "items" }, examples.map((question, i) => h("li", {}, h("button", {
        class: "item",
        attrs: { type: "button", disabled: !usable, "data-focus": `example-${i}` },
        onClick: () => actions.askExample(question),
    }, question)))));
}
