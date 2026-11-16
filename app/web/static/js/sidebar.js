// The sidebar: a Tables / History switch, then either the schema tree (both databases, their tables
// with row counts, each table's columns with their types) under a filter, or this session's runs.
import { chevron, word } from "./controls.js";
import { count, EMPTY_HISTORY, FILTER_LABEL, NO_MATCH, NO_TABLES } from "./copy.js";
import { fill, h, redraw } from "./dom.js";
import { activeTable, datasetLabel, QUERY_TAB, tableKey } from "./state.js";
// The words a database row shows when it isn't ready, and the tone behind each.
const DATASET_STATES = {
    loading: ["Loading", "quiet"], missing: ["Missing", "bad"], failed: ["Failed", "bad"],
};
const OUTCOMES = { warned: "Warning", refused: "Refused", not_run: "Not run" };
let filterInput = null;
let treeFocus = null; // the row that takes the tree's one tab stop
let visible = [];
let filtering = false; // a filter holds rows open, so Left steps out of them instead of closing them
// The filter field, made once so typing, the caret and focus survive every redraw.
function filterField(actions) {
    if (!filterInput) {
        filterInput = h("input", {
            class: "filter",
            attrs: { type: "search", placeholder: FILTER_LABEL, "aria-label": FILTER_LABEL, autocomplete: "off", spellcheck: "false" },
            onInput: (event) => actions.setFilter(event.target instanceof HTMLInputElement ? event.target.value : ""),
        });
    }
    return filterInput;
}
function matches(text, filter) {
    return text.toLowerCase().includes(filter);
}
function treeRow(level, open, parts, onTwisty) {
    return h("div", { class: `tree-row l${level}` }, h("span", {
        class: open === null ? "twisty none" : open ? "twisty open" : "twisty",
        onClick: onTwisty ? (event) => { event.stopPropagation(); onTwisty(); } : undefined,
    }, open === null ? null : chevron()), parts);
}
function treeItem(key, level, open, selected, row, group) {
    return h("li", {
        class: "tree-item",
        attrs: {
            role: "treeitem", "aria-level": level, "aria-expanded": open === null ? null : String(open),
            "aria-selected": selected === null ? null : String(selected), tabindex: "-1", "data-key": key, "data-focus": `tree-${key}`,
        },
    }, row, group);
}
// One table, its columns under it when open; with a filter, only the columns that match (or all, when the table does).
function tableItem(dataset, table, state, actions, filter, parent) {
    const key = `t:${tableKey(dataset, table.name)}`;
    const nameHit = filter !== "" && matches(table.name, filter);
    const columns = filter && !nameHit ? table.columns.filter((column) => matches(column.name, filter)) : table.columns;
    if (filter && !nameHit && columns.length === 0)
        return null;
    const open = state.expanded.has(key) || (filter !== "" && !nameHit);
    const current = activeTable(state);
    const selected = current?.dataset === dataset && current.table === table.name;
    const toggle = () => actions.toggleNode(key);
    const openTable = () => actions.openTable(dataset, table.name);
    visible.push({ key, parent, open, activate: openTable });
    const children = open ? columns.map((column) => {
        const columnKey = `c:${tableKey(dataset, table.name)}.${column.name}`;
        visible.push({ key: columnKey, parent: key, open: null, activate: () => actions.openTable(dataset, table.name, column.name) });
        const row = treeRow(3, null, [
            h("span", { class: "tree-name" }, column.name),
            column.key ? h("span", { class: "tree-key" }, "key") : null,
            h("span", { class: "tree-type" }, column.type),
        ], null);
        row.addEventListener("click", () => actions.openTable(dataset, table.name, column.name));
        return treeItem(columnKey, 3, null, null, row, null);
    }) : [];
    const row = treeRow(2, open, [h("span", { class: "tree-name" }, table.name), h("span", { class: "tree-count" }, count(table.rows))], toggle);
    row.addEventListener("click", openTable);
    return treeItem(key, 2, open, selected, row, children.length ? h("ul", { attrs: { role: "group" } }, children) : null);
}
function databaseItem(name, info, state, actions, filter) {
    const key = `d:${name}`;
    const node = { key, parent: null, open: null, activate: () => actions.toggleNode(key) };
    visible.push(node);
    const tables = info.tables.map((table) => tableItem(name, table, state, actions, filter, key)).filter((item) => item !== null);
    if (filter && tables.length === 0) {
        visible.pop();
        return null;
    }
    const open = info.tables.length === 0 ? null : filter !== "" || state.expanded.has(key);
    node.open = open;
    if (!open)
        visible.splice(visible.indexOf(node) + 1); // a closed database's tables aren't on screen
    const words = DATASET_STATES[(state.status?.datasets[name] ?? info).state]; // the live state, not the one read with the schema
    const row = treeRow(1, open, [
        h("span", { class: "tree-name db" }, info.label),
        words ? h("span", { class: `tree-state ${words[1]}` }, words[0]) : null,
    ], open === null ? null : () => actions.toggleNode(key));
    row.addEventListener("click", () => { if (open !== null)
        actions.toggleNode(key); });
    return treeItem(key, 1, open, null, row, open && tables.length ? h("ul", { attrs: { role: "group" } }, tables) : null);
}
// Moves the tree's tab stop to one row and focuses it.
function focusRow(tree, key) {
    treeFocus = key;
    for (const item of tree.querySelectorAll("[role=treeitem]")) {
        const here = item.dataset["key"] === key;
        item.tabIndex = here ? 0 : -1;
        if (here)
            item.focus();
    }
}
// Up and Down walk the rows on screen; Right opens a row or steps into it; Left closes it or steps out;
// Enter opens a table (or a column's table), and opens or closes a database.
function onTreeKey(tree, event, actions) {
    const item = event.target instanceof Element ? event.target.closest("[role=treeitem]") : null;
    const key = item?.dataset["key"];
    const at = visible.findIndex((node) => node.key === key);
    const node = visible[at];
    if (!node)
        return;
    let next;
    switch (event.key) {
        case "ArrowDown":
            next = visible[at + 1];
            break;
        case "ArrowUp":
            next = visible[at - 1];
            break;
        case "Home":
            next = visible[0];
            break;
        case "End":
            next = visible[visible.length - 1];
            break;
        case "ArrowRight":
            if (node.open === false)
                actions.toggleNode(node.key);
            else if (node.open)
                next = visible[at + 1];
            break;
        case "ArrowLeft":
            if (node.open && !filtering)
                actions.toggleNode(node.key);
            else if (node.parent)
                next = visible.find((other) => other.key === node.parent);
            break;
        case "Enter":
            node.activate();
            break;
        default:
            return;
    }
    event.preventDefault();
    if (next)
        focusRow(tree, next.key);
}
// The tree under the filter, or why there's nothing to show.
function tablesPanel(state, actions) {
    const filter = state.filter.trim().toLowerCase();
    filtering = filter !== "";
    visible = [];
    const datasets = Object.entries(state.schema?.datasets ?? {});
    const items = datasets.map(([name, info]) => databaseItem(name, info, state, actions, filter)).filter((item) => item !== null);
    if (!visible.some((node) => node.key === treeFocus)) {
        const current = activeTable(state);
        const selected = current ? `t:${current.key}` : null;
        treeFocus = visible.find((node) => node.key === selected)?.key ?? visible[0]?.key ?? null;
    }
    const tree = h("ul", { class: "tree", attrs: { role: "tree", "aria-label": "Tables" } }, items);
    tree.addEventListener("keydown", (event) => onTreeKey(tree, event, actions));
    tree.addEventListener("focusin", (event) => {
        const key = event.target instanceof HTMLElement ? event.target.dataset["key"] : undefined;
        if (key)
            treeFocus = key;
    });
    for (const item of tree.querySelectorAll("[role=treeitem]"))
        item.tabIndex = item.dataset["key"] === treeFocus ? 0 : -1;
    const empty = !state.schema ? state.schemaError : filter && items.length === 0 ? NO_MATCH : datasets.length === 0 ? NO_TABLES : null;
    return [empty ? h("p", { class: "hint side-hint" }, empty) : tree];
}
function kindWord(entry) {
    if (entry.kind === "edit")
        return "Edited SQL";
    return entry.kind === "sql" ? "SQL" : "Ask";
}
// One run: its kind, its outcome word, its text, and its database when it isn't the one the Query tab uses.
function historyItem(entry, state, actions) {
    const current = state.active === QUERY_TAB && state.job?.id === entry.id;
    const outcome = entry.verdict ? OUTCOMES[entry.verdict.kind] : undefined;
    const other = entry.dataset !== state.dataset ? datasetLabel(state, entry.dataset) : null;
    return h("li", {}, h("button", {
        class: current ? "hist on" : "hist",
        attrs: { type: "button", "aria-current": current ? "true" : null, title: entry.text, "data-focus": `job-${entry.id}` },
        onClick: () => actions.openJob(entry.id),
    }, h("span", { class: "hist-top" }, h("span", { class: "hist-kind" }, kindWord(entry)), outcome ? word(outcome) : null), h("span", { class: entry.kind === "sql" || entry.kind === "edit" ? "clamp code" : "clamp" }, entry.text.trim()), other ? h("span", { class: "hist-note" }, other) : null));
}
// The job whose History entry was last brought into view.
let revealed = null;
// Scrolls History, and only it, just far enough to show the shown run's entry, once per run.
function revealHistory(scroller, state) {
    const job = state.job;
    const item = job && job.id !== revealed ? scroller.querySelector(".hist.on") : null;
    if (!job || !item)
        return;
    revealed = job.id;
    const box = scroller.getBoundingClientRect();
    const at = item.getBoundingClientRect();
    if (at.top < box.top)
        scroller.scrollTop -= box.top - at.top;
    else if (at.bottom > box.bottom)
        scroller.scrollTop += Math.min(at.bottom - box.bottom, at.top - box.top);
}
function historyPanel(state, actions) {
    return [state.history.length
            ? h("ul", { class: "hist-list" }, state.history.map((entry) => historyItem(entry, state, actions)))
            : h("p", { class: "hint side-hint" }, EMPTY_HISTORY)];
}
const SIDE_TABS = [{ value: "tables", text: "Tables" }, { value: "history", text: "History" }];
// The Tables / History switch: two tabs, the arrow keys move between them.
function sideTabs(state, actions) {
    return h("div", {
        class: "side-tabs",
        attrs: { role: "tablist", "aria-label": "Sidebar" },
        onKeyDown: (event) => {
            if (event.key !== "ArrowLeft" && event.key !== "ArrowRight")
                return;
            event.preventDefault();
            const next = state.sideTab === "tables" ? "history" : "tables";
            actions.setSideTab(next);
            document.getElementById(`side-${next}`)?.focus();
        },
    }, SIDE_TABS.map((tab) => {
        const on = state.sideTab === tab.value;
        return h("button", {
            class: on ? "side-tab on" : "side-tab",
            attrs: {
                type: "button", role: "tab", id: `side-${tab.value}`, "aria-selected": String(on), "aria-controls": "side-panel",
                tabindex: on ? "0" : "-1", "data-focus": `side-${tab.value}`,
            },
            onClick: () => actions.setSideTab(tab.value),
        }, tab.text, tab.value === "history" && state.history.length ? h("span", { class: "side-count" }, count(state.history.length)) : null);
    }));
}
let frame = null;
function frameOf(actions) {
    if (!frame) {
        frame = {
            tabs: h("div", { class: "side-tabs-slot" }),
            panel: h("div", { class: "side-panel", attrs: { id: "side-panel", role: "tabpanel" } }),
            filterRow: h("div", { class: "filter-row" }, filterField(actions)),
            tree: h("div", { class: "side-scroll" }),
            history: h("div", { class: "side-scroll" }),
        };
    }
    return frame;
}
export function drawSidebar(root, state, actions) {
    const f = frameOf(actions);
    fill(root, f.tabs, f.panel);
    redraw(f.tabs, sideTabs(state, actions));
    f.panel.setAttribute("aria-labelledby", `side-${state.sideTab}`);
    if (state.sideTab === "tables") {
        fill(f.panel, f.filterRow, f.tree);
        redraw(f.tree, tablesPanel(state, actions));
    }
    else {
        fill(f.panel, f.history);
        redraw(f.history, historyPanel(state, actions));
        revealHistory(f.history, state);
    }
}
// Brings a tree row into view inside the sidebar only, as when its table opens from somewhere else.
export function revealRow(root, key) {
    const scroller = frame?.tree;
    const row = root.querySelector(`[data-key="${CSS.escape(key)}"] > .tree-row`);
    if (!scroller || !row)
        return;
    const box = scroller.getBoundingClientRect();
    const at = row.getBoundingClientRect();
    if (at.top < box.top)
        scroller.scrollTop -= box.top - at.top;
    else if (at.bottom > box.bottom)
        scroller.scrollTop += at.bottom - box.bottom;
}
