// An answer's table: the first 500 rows from the snapshot plus any pages fetched since,
// "Show 500 more" while the reply holds more, and the CSV link. Rows stay one line tall:
// a long text cell is cut with an ellipsis and keeps its full text in a tooltip.
import { csvHref } from "./api.js";
import { count, NO_ROWS } from "./copy.js";
import { h } from "./dom.js";
import type { Actions, State } from "./state.js";
import type { AnswerName, AnswerView } from "./types.js";

const NUMBER_TYPES = ["INTEGER", "BIGINT", "DECIMAL"];
const PAGE = 500;

export function answerTable(jobId: string, answer: AnswerName, view: AnswerView, state: State, actions: Actions): HTMLElement {
  const numeric = view.columns.map((column) => NUMBER_TYPES.includes(column.type));
  const rows = [...view.rows, ...state.extra[answer]];
  const more = Math.min(PAGE, view.shown - rows.length);
  const loading = state.loadingMore === answer;

  const table = h("table", {},
    h("thead", {}, h("tr", {}, view.columns.map((column, i) =>
      h("th", { class: numeric[i] ? "num" : undefined, attrs: { scope: "col" } }, column.name)))),
    h("tbody", {}, rows.map((row) => h("tr", {}, row.map((cell, i) => numeric[i]
      ? h("td", { class: "num" }, cell)
      : h("td", { class: "text", attrs: { title: cell || null } }, h("span", { class: "cell" }, cell)))))));

  return h("div", { class: "answer-table" },
    h("div", { class: "scroller", attrs: { "data-scroll": `rows-${answer}`, tabindex: "0", role: "region", "aria-label": `Rows of the ${answer} answer` } },
      table),
    rows.length === 0 ? h("p", { class: "hint" }, NO_ROWS) : null,
    h("div", { class: "table-foot" },
      rows.length < view.row_count ? h("span", { class: "muted" }, `Showing ${count(rows.length)} of ${count(view.row_count)} rows.`) : null,
      more > 0 ? h("button", {
        class: "btn2",
        attrs: { type: "button", disabled: loading, "data-focus": `more-${answer}` },
        onClick: () => actions.showMore(answer),
      }, loading ? "Loading rows" : `Show ${count(more)} more`) : null,
      h("a", { class: "btn2", attrs: { href: csvHref(jobId, answer), download: `shilmandb-${jobId}.csv`, "data-focus": `csv-${answer}` } },
        view.csv_label)),
    state.moreError?.answer === answer ? h("p", { class: "msg bad", attrs: { role: "alert" } }, state.moreError.message) : null);
}
