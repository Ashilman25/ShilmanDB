// A table's tab: a toolbar (its name, database, size and Query this table), its first rows in the grid,
// and the Columns inspector, which shows each column's notes as the analyst is given them.
import { banner } from "./controls.js";
import { count, moreText, previewCap, QUERY_TABLE, rowsRange, tableMeta } from "./copy.js";
import { fill, h, redraw, redrawFrom } from "./dom.js";
import { grid, settleGrids } from "./grid.js";
import { activeTable, datasetLabel, queryBlock, schemaTable, } from "./state.js";
import { tabId } from "./tabs.js";
export const PREVIEW_PAGE = 200;
const SKELETON_ROWS = 12;
const VALUES_LINES = 2;
const VALUES_FONT = "13px -apple-system, BlinkMacSystemFont, \"Helvetica Neue\", Helvetica, Arial, sans-serif";
// Measures text in the inspector's font, so a column's values can stop at about two lines.
let measure;
function textWidth(text) {
    if (measure === undefined) {
        measure = document.createElement("canvas").getContext("2d");
        if (measure)
            measure.font = VALUES_FONT;
    }
    return measure ? measure.measureText(text).width : text.length * 7;
}
function quoted(value) {
    return `'${value.replace(/'/g, "''")}'`;
}
// As many values as fit in about two lines of the given width, then how many were left out.
function valuesText(values, width) {
    const room = width * VALUES_LINES * 0.9;
    const shown = [];
    let used = textWidth("Values: ");
    for (const [i, value] of values.entries()) {
        const piece = quoted(value) + (i < values.length - 1 ? ", " : "");
        const rest = values.length - i - 1;
        const tail = rest > 0 ? textWidth(` and ${rest} more`) : 0;
        if (shown.length > 0 && used + textWidth(piece) + tail > room)
            break;
        shown.push(quoted(value));
        used += textWidth(piece);
    }
    const left = values.length - shown.length;
    return `Values: ${shown.join(", ")}${left > 0 ? ` and ${count(left)} more` : ""}`;
}
// The table's name, database and size, then Query this table: off, with the reason beside it, while a run
// is under way or the database isn't ready, as the Run button is.
function toolbar(tab, table, state, actions) {
    const blocked = queryBlock(state, tab.dataset);
    const why = "query-why";
    return h("div", { class: "toolbar" }, h("h1", { class: "tb-name" }, tab.table), h("span", { class: "tb-note" }, datasetLabel(state, tab.dataset)), table ? h("span", { class: "tb-note" }, tableMeta(table.rows, table.columns.length)) : null, h("span", { class: "tb-end" }, blocked ? h("span", { class: "tb-why", attrs: { id: why, title: blocked } }, blocked) : null, h("button", {
        class: "btn2",
        attrs: { type: "button", disabled: blocked !== null, "aria-describedby": blocked ? why : null, "data-focus": `query-${tab.key}` },
        onClick: () => actions.queryTable(tab.dataset, tab.table),
    }, QUERY_TABLE)));
}
// Why the rows aren't here: the dataset still loading, or not there at all.
function waitingWords(state, tab) {
    const info = state.status?.datasets[tab.dataset];
    if (!info || info.state === "loading")
        return { loading: true, text: null, detail: null };
    if (info.state !== "ready")
        return { loading: false, text: `${info.label} isn't available.`, detail: info.detail };
    return { loading: false, text: null, detail: null };
}
function rowsGrid(tab, table, preview, state, actions) {
    const waiting = waitingWords(state, tab);
    const data = preview?.data ?? null;
    if (!data && waiting.text) {
        return h("div", { class: "grid-note" }, banner("error", waiting.text, waiting.detail ? h("p", { class: "pre detail" }, waiting.detail) : null));
    }
    if (!data && preview?.error)
        return h("div", { class: "grid-note" }, banner("error", preview.error));
    const columns = data?.columns ?? table?.columns.map((column) => ({ name: column.name, type: column.type })) ?? [];
    const total = data?.total ?? table?.rows ?? 0;
    const shown = data ? Math.min(preview?.shown ?? PREVIEW_PAGE, data.rows.length) : 0;
    const more = data ? Math.min(PREVIEW_PAGE, data.rows.length - shown) : 0;
    const capped = data !== null && more === 0 && data.rows.length < total;
    const foot = data
        ? capped
            ? [h("span", { class: "foot-note" }, previewCap(data.rows.length, total))]
            : [h("span", { class: "foot-note" }, rowsRange(shown, total)),
                more > 0 ? h("button", {
                    class: "btn2",
                    attrs: { type: "button", "data-focus": `more-${tab.key}` },
                    onClick: () => actions.showMorePreview(tab.key),
                }, moreText(more)) : null]
        : [h("span", { class: "foot-note" }, waiting.loading ? `Loading ${datasetLabel(state, tab.dataset)}.` : "Loading rows.")];
    return grid({
        key: `table:${tab.key}`,
        label: `Rows of ${tab.table}`,
        columns,
        rows: data ? data.rows.slice(0, shown) : [],
        skeleton: data ? 0 : SKELETON_ROWS,
        column: preview?.column ?? null,
        onPick: (column, value) => actions.pickCell({ grid: `table:${tab.key}`, column, value }),
        onColumn: (name) => actions.pickColumn(tab.key, name),
        foot,
    });
}
function columnEntry(column, picked, width) {
    return h("li", { class: picked ? "col-entry picked" : "col-entry", attrs: { "data-col": column.name, "aria-current": picked ? "true" : null } }, h("div", { class: "col-line" }, h("span", { class: "col-entry-name" }, column.name), column.key ? h("span", { class: "col-entry-key" }, "key") : null, h("span", { class: "col-entry-type" }, column.type)), column.references ? h("p", { class: "col-entry-ref" }, `References ${column.references}`) : null, column.description ? h("p", { class: "col-entry-text" }, column.description) : null, column.values?.length ? h("p", { class: "col-entry-values" }, valuesText(column.values, width)) : null);
}
const INSPECTOR_TEXT = 308; // the inspector's text width at 340px, less its padding
function inspector(tab, table, preview, width) {
    const picked = preview?.column ?? null;
    return h("aside", { class: "inspector", attrs: { "aria-label": "Columns" } }, h("div", { class: "ins-head" }, h("h2", {}, "Columns")), h("div", { class: "ins-body", attrs: { "data-scroll": `cols-${tab.key}` } }, table
        ? [
            h("div", { class: "cols-lead" }, h("p", { class: "cols-table" }, tab.table), table.primary_key.length ? h("p", { class: "cols-key" }, `Key: ${table.primary_key.join(", ")}`) : null),
            h("ol", { class: "col-list" }, table.columns.map((column) => columnEntry(column, column.name === picked, width))),
        ]
        : h("p", { class: "hint" }, "The column notes show once the schema is in.")));
}
// Scrolls the inspector, and only it, just far enough to show the picked column's entry.
export function revealColumnEntry(root, column) {
    const body = root.querySelector(".ins-body");
    const entry = body?.querySelector(`[data-col="${CSS.escape(column)}"]`);
    if (!body || !entry || body.scrollHeight <= body.clientHeight)
        return;
    const box = body.getBoundingClientRect();
    const at = entry.getBoundingClientRect();
    if (at.top < box.top)
        body.scrollTop -= box.top - at.top + 8;
    else if (at.bottom > box.bottom)
        body.scrollTop += Math.min(at.bottom - box.bottom + 8, at.top - box.top);
}
export function drawTablePane(root, state, actions) {
    const tab = activeTable(state);
    root.hidden = !tab;
    if (!tab) {
        redraw(root);
        return;
    }
    root.setAttribute("aria-labelledby", tabId(tab.key));
    const table = schemaTable(state, tab.dataset, tab.table);
    const preview = state.previews.get(tab.key);
    root.setAttribute("aria-busy", String(!preview?.data && !preview?.error));
    // the text width the inspector had last time, so values fill about two lines wherever it sits
    const list = root.querySelector(".col-list");
    const width = list && list.clientWidth > 0 ? list.clientWidth : INSPECTOR_TEXT;
    // The grid stays in the same wrapper from redraw to redraw, so it never leaves the page and keeps its scroll.
    paneMain ??= h("div", { class: "pane-main" });
    const main = paneMain;
    redrawFrom(root, () => {
        fill(main, toolbar(tab, table, state, actions), rowsGrid(tab, table, preview, state, actions));
        return [main, inspector(tab, table, preview, width)];
    });
    settleGrids(root);
}
let paneMain = null;
