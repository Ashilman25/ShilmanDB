// The left rail: this session's history, newest first, then the current dataset's examples.
import { EMPTY_HISTORY } from "./copy.js";
import { h, redraw } from "./dom.js";
import { askBlock, canStart, datasetLabel, type Actions, type State } from "./state.js";
import type { HistoryEntry } from "./types.js";

function firstLine(text: string): string {
  return text.trim().split("\n", 1)[0] ?? "";
}

// How a history entry reads: questions as asked, SQL with what kind it was.
function entryText(entry: HistoryEntry): string {
  if (entry.kind === "sql") return `SQL: ${firstLine(entry.text)}`;
  if (entry.kind === "edit") return `Edited SQL: ${firstLine(entry.text)}`;
  return entry.text;
}

function historyItem(entry: HistoryEntry, state: State, actions: Actions): HTMLElement {
  const current = state.job?.id === entry.id;
  const other = entry.dataset !== state.dataset ? datasetLabel(state, entry.dataset) : null;
  return h("li", {},
    h("button", {
      class: current ? "item on" : "item",
      attrs: { type: "button", "aria-current": current ? "true" : null, title: entry.text, "data-focus": `job-${entry.id}` },
      onClick: () => actions.openJob(entry.id),
    }, h("span", { class: "clamp" }, entryText(entry)), other ? h("span", { class: "item-note" }, other) : null));
}

// The job whose History entry was last brought into view.
let revealed: string | null = null;

// Scrolls the rail, and only the rail, just far enough to show the current entry.
function reveal(root: HTMLElement, item: Element): void {
  const box = root.getBoundingClientRect();
  const at = item.getBoundingClientRect();
  if (at.top < box.top) root.scrollTop -= box.top - at.top;
  else if (at.bottom > box.bottom) root.scrollTop += Math.min(at.bottom - box.bottom, at.top - box.top);
}

export function drawRail(root: HTMLElement, state: State, actions: Actions): void {
  const examples = state.status?.examples[state.dataset] ?? [];
  const blocked = askBlock(state);
  const usable = blocked === null && canStart(state);
  redraw(root,
    h("h2", {}, "History"),
    state.history.length
      ? h("ul", { class: "items" }, state.history.map((entry) => historyItem(entry, state, actions)))
      : h("p", { class: "hint" }, EMPTY_HISTORY),
    h("h2", {}, "Examples"),
    blocked && state.status && !state.statusError ? h("p", { class: "hint" }, blocked) : null,
    h("ul", { class: "items" }, examples.map((question, i) => h("li", {},
      h("button", {
        class: "item",
        attrs: { type: "button", disabled: !usable, "data-focus": `example-${i}` },
        onClick: () => actions.askExample(question),
      }, question)))));

  // When the shown job changes, its entry comes into view once it's in the list.
  const job = state.job;
  const item = job && job.id !== revealed ? root.querySelector(".item.on") : null;
  if (job && item) {
    revealed = job.id;
    reveal(root, item);
  }
}
