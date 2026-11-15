// An answer's table: the first 500 rows from the snapshot plus any pages fetched since,
// "Show 500 more" while the reply holds more, and the CSV link. Rows stay one line tall:
// a long text cell is cut with an ellipsis and keeps its full text in a tooltip.
import { csvHref } from "./api.js";
import { count, NO_ROWS } from "./copy.js";
import { fill, h } from "./dom.js";
import type { Actions, State } from "./state.js";
import type { AnswerName, AnswerView } from "./types.js";

const NUMBER_TYPES = ["INTEGER", "BIGINT", "DECIMAL"];
const PAGE = 500;

// One answer's table as drawn: each row is built once and the same nodes go back on screen
// in every later redraw, so a redraw only builds the rows it hasn't seen.
interface Kept {
  columns: string;
  box: HTMLElement;
  scroller: HTMLElement;
  body: HTMLTableSectionElement;
  rows: string[][];
  elements: HTMLTableRowElement[];
}

let keptJob: string | null = null;
const kept = new Map<AnswerName, Kept>();

// Lets go of every kept row, for when another job is shown.
export function forgetRows(): void {
  keptJob = null;
  kept.clear();
}

function sameRow(a: string[], b: string[]): boolean {
  return a === b || (a.length === b.length && a.every((cell, i) => cell === b[i]));
}

function rowElement(row: string[], numeric: boolean[]): HTMLTableRowElement {
  return h("tr", {}, row.map((cell, i) => numeric[i]
    ? h("td", { class: "num" }, cell)
    : h("td", { class: "text", attrs: { title: cell || null } }, h("span", { class: "cell" }, cell))));
}

// The kept table for this answer, built on first use, with its rows brought up to these.
function keptTable(jobId: string, answer: AnswerName, view: AnswerView, rows: string[][]): Kept {
  if (keptJob !== jobId) {
    forgetRows();
    keptJob = jobId;
  }
  const numeric = view.columns.map((column) => NUMBER_TYPES.includes(column.type));
  const columns = JSON.stringify(view.columns);
  let table = kept.get(answer);
  if (!table || table.columns !== columns) {
    const body = h("tbody");
    const scroller = h("div", { class: "scroller", attrs: { "data-scroll": `rows-${answer}`, tabindex: "0", role: "region", "aria-label": `Rows of the ${answer} answer` } },
      h("table", {},
        h("thead", {}, h("tr", {}, view.columns.map((column, i) =>
          h("th", { class: numeric[i] ? "num" : undefined, attrs: { scope: "col" } }, column.name)))),
        body));
    table = { columns, box: h("div", { class: "answer-table" }), scroller, body, rows: [], elements: [] };
    kept.set(answer, table);
  }

  let same = 0;
  const known = table.rows;
  while (same < rows.length && same < known.length && sameRow(rows[same] ?? [], known[same] ?? [])) same++;
  for (const stale of table.elements.splice(same)) stale.remove();
  const fresh = rows.slice(same).map((row) => rowElement(row, numeric));
  if (fresh.length > 0) {
    table.body.append(...fresh);
    table.elements.push(...fresh);
  }
  table.rows = rows;
  return table;
}

export function answerTable(jobId: string, answer: AnswerName, view: AnswerView, state: State, actions: Actions): HTMLElement {
  const rows = [...view.rows, ...state.extra[answer]];
  const more = Math.min(PAGE, view.shown - rows.length);
  const loading = state.loadingMore === answer;
  const waiting = state.loadingMore !== null;
  const table = keptTable(jobId, answer, view, rows);

  fill(table.box,
    table.scroller,
    rows.length === 0 ? h("p", { class: "hint" }, NO_ROWS) : null,
    h("div", { class: "table-foot" },
      rows.length < view.row_count ? h("span", { class: "muted" }, `Showing ${count(rows.length)} of ${count(view.row_count)} rows.`) : null,
      // Off while any rows load, but still focusable, so focus stays on it through the load.
      more > 0 ? h("button", {
        class: "btn2",
        attrs: { type: "button", "aria-disabled": waiting ? "true" : null, "data-focus": `more-${answer}` },
        onClick: () => { if (!waiting) actions.showMore(answer); },
      }, loading ? "Loading rows" : `Show ${count(more)} more`) : null,
      h("a", { class: "btn2", attrs: { href: csvHref(jobId, answer), download: `shilmandb-${jobId}.csv`, "data-focus": `csv-${answer}` } },
        view.csv_label)),
    state.moreError?.answer === answer ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.moreError.message) : null);
  return table.box;
}
