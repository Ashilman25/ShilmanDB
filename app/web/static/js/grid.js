// The data grid the table tabs and the results share: a real table in its own scroll box, with the
// row numbers and the header held in place and each column's type under its name. One cell can be
// picked by click or with the arrow keys; Cmd+C copies its value. Every value goes in as text.
import { fill, h } from "./dom.js";
const NUMBER_TYPES = ["INTEGER", "BIGINT", "DECIMAL"];
const LONG_CELL = 24; // past this many characters a cell may pass 340px and be cut, so it keeps its value in a tooltip
const SKELETON_WIDTHS = [70, 45, 85, 60, 40, 92, 55];
const kept = new Map();
let made = 0;
export function forgetGrid(key) {
    kept.delete(key);
}
export function forgetGrids(prefix) {
    for (const key of [...kept.keys()])
        if (key.startsWith(prefix))
            kept.delete(key);
}
export function isNumberType(type) {
    return NUMBER_TYPES.includes(type);
}
function sameRow(a, b) {
    return a === b || (a.length === b.length && a.every((cell, i) => cell === b[i]));
}
function rowElement(index, row, numeric) {
    return h("tr", {}, h("th", { class: "rn", attrs: { scope: "row" } }, String(index + 1)), row.map((cell, i) => numeric[i]
        ? h("td", { class: "num" }, cell)
        : h("td", { class: "text", attrs: { title: cell.length > LONG_CELL ? cell : null } }, h("span", { class: "cell" }, cell))));
}
function skeletonRow(index, columns) {
    return h("tr", { class: "skel-row", attrs: { "aria-hidden": "true" } }, h("th", { class: "rn" }), Array.from({ length: columns }, (_, i) => h("td", {}, h("span", { class: `skel w${SKELETON_WIDTHS[(index + i) % SKELETON_WIDTHS.length] ?? 70}` }))));
}
function headCell(column, numeric, picks) {
    const parts = [h("span", { class: "col-name" }, column.name), h("span", { class: "col-type" }, column.type)];
    const content = picks
        ? h("button", { class: "col-head", attrs: { type: "button", "data-col": column.name, "aria-pressed": "false" } }, parts)
        : h("span", { class: "col-head" }, parts);
    return h("th", { class: numeric ? "num" : undefined, attrs: { scope: "col" } }, content);
}
// The cell at a row and column, past the row number.
function cellAt(entry, row, col) {
    return entry.elements[row]?.cells[col + 1];
}
// Keeps the grid's place at once, not on the scroll event, so a redraw right after a scroll puts it back there.
function remember(entry) {
    entry.scroll = { top: entry.scroller.scrollTop, left: entry.scroller.scrollLeft };
}
// Scrolls the grid just enough to show a cell clear of the held header and row numbers.
function reveal(entry, cell) {
    const box = entry.scroller;
    const header = entry.table.tHead?.offsetHeight ?? 0;
    const gutter = entry.head.cells[0]?.offsetWidth ?? 0;
    const top = entry.table.offsetTop + cell.offsetTop;
    const left = entry.table.offsetLeft + cell.offsetLeft;
    if (top - header < box.scrollTop)
        box.scrollTop = top - header;
    else if (top + cell.offsetHeight > box.scrollTop + box.clientHeight)
        box.scrollTop = top + cell.offsetHeight - box.clientHeight;
    if (left - gutter < box.scrollLeft)
        box.scrollLeft = left - gutter;
    else if (left + cell.offsetWidth > box.scrollLeft + box.clientWidth)
        box.scrollLeft = left + cell.offsetWidth - box.clientWidth;
    remember(entry);
}
// Draws the pick on one cell and tells the page which value it holds.
function mark(entry, row, col, report) {
    const old = entry.at ? cellAt(entry, entry.at.row, entry.at.col) : undefined;
    old?.classList.remove("picked");
    old?.removeAttribute("aria-selected");
    const cell = cellAt(entry, row, col);
    const column = entry.spec.columns[col];
    if (!cell || !column) {
        entry.at = null;
        entry.table.removeAttribute("aria-activedescendant");
        return;
    }
    entry.at = { row, col };
    if (!cell.id)
        cell.id = `${entry.table.id}-${row}-${col}`;
    cell.classList.add("picked");
    cell.setAttribute("aria-selected", "true");
    entry.table.setAttribute("aria-activedescendant", cell.id);
    if (report)
        entry.spec.onPick(column.name, entry.rows[row]?.[col] ?? "");
}
function copyText(text) {
    navigator.clipboard?.writeText(text).catch(() => undefined);
}
// A number cell's value without its thousands separators, so it pastes into SQL; the grid still shows them.
function plainNumber(text) {
    return text.replace(/,/g, "");
}
const MOVES = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
function onKey(entry, event) {
    const rows = entry.elements.length;
    const columns = entry.spec.columns.length;
    if (rows === 0 || columns === 0)
        return;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "c" && entry.at) {
        const selection = window.getSelection();
        if (selection && !selection.isCollapsed)
            return; // copying text you selected yourself
        event.preventDefault();
        const cell = entry.rows[entry.at.row]?.[entry.at.col] ?? "";
        copyText(entry.numeric[entry.at.col] ? plainNumber(cell) : cell);
        return;
    }
    const at = entry.at ?? { row: 0, col: 0 };
    let next = null;
    const move = MOVES[event.key];
    if (move)
        next = entry.at ? { row: at.row + move[0], col: at.col + move[1] } : at;
    else if (event.key === "Home")
        next = { row: event.metaKey || event.ctrlKey ? 0 : at.row, col: 0 };
    else if (event.key === "End")
        next = { row: event.metaKey || event.ctrlKey ? rows - 1 : at.row, col: columns - 1 };
    if (!next || event.altKey)
        return;
    event.preventDefault();
    const row = Math.max(0, Math.min(rows - 1, next.row));
    const col = Math.max(0, Math.min(columns - 1, next.col));
    mark(entry, row, col, true);
    const cell = cellAt(entry, row, col);
    if (cell)
        reveal(entry, cell);
}
function onClick(entry, event) {
    const target = event.target instanceof Element ? event.target.closest("td") : null;
    const row = target?.parentElement;
    if (!(target instanceof HTMLTableCellElement) || !(row instanceof HTMLTableRowElement))
        return;
    const index = entry.elements.indexOf(row);
    if (index < 0)
        return;
    entry.table.focus({ preventScroll: true }); // the arrow keys and Cmd+C follow the click
    mark(entry, index, target.cellIndex - 1, true);
}
function onHeadClick(entry, event) {
    const button = event.target instanceof Element ? event.target.closest("button.col-head") : null;
    const name = button instanceof HTMLElement ? button.dataset["col"] : undefined;
    if (name !== undefined)
        entry.spec.onColumn?.(name);
}
function build(spec) {
    const head = h("tr");
    const body = h("tbody");
    const table = h("table", {
        class: "grid-table",
        attrs: { id: `grid-${++made}`, role: "grid", tabindex: "0", "aria-label": spec.label, "aria-readonly": "true" },
    }, h("thead", {}, head), body);
    const scroller = h("div", { class: "grid-scroll" }, table);
    const foot = h("div", { class: "grid-foot" });
    const entry = {
        spec, shape: "", box: h("div", { class: "grid" }, scroller, foot), scroller, table, head, body, foot,
        rows: [], elements: [], numeric: [], skeleton: false, at: null, scroll: { top: 0, left: 0 },
    };
    table.addEventListener("keydown", (event) => onKey(entry, event));
    // The grid's own ring, drawn over the held header and row numbers, only when focus came by keyboard.
    table.addEventListener("focus", () => entry.box.classList.toggle("focused", table.matches(":focus-visible")));
    table.addEventListener("blur", () => entry.box.classList.remove("focused"));
    body.addEventListener("click", (event) => onClick(entry, event));
    head.addEventListener("click", (event) => onHeadClick(entry, event));
    scroller.addEventListener("scroll", () => { if (scroller.isConnected && scroller.offsetParent !== null)
        remember(entry); }, { passive: true });
    return entry;
}
// Brings the kept body up to these rows: rows already drawn stay, from the first that differs on.
function updateRows(entry, rows, skeleton) {
    if (rows.length === 0 && skeleton > 0) {
        fill(entry.body, Array.from({ length: skeleton }, (_, i) => skeletonRow(i, entry.spec.columns.length)));
        entry.rows = [];
        entry.elements = [];
        entry.skeleton = true;
        return;
    }
    if (entry.skeleton) {
        fill(entry.body);
        entry.skeleton = false;
    }
    let same = 0;
    while (same < rows.length && same < entry.rows.length && sameRow(rows[same] ?? [], entry.rows[same] ?? []))
        same++;
    for (const stale of entry.elements.splice(same))
        stale.remove();
    const fresh = rows.slice(same).map((row, i) => rowElement(same + i, row, entry.numeric));
    if (fresh.length > 0) {
        entry.body.append(...fresh);
        entry.elements.push(...fresh);
    }
    entry.rows = rows;
    if (entry.at && !cellAt(entry, entry.at.row, entry.at.col))
        entry.at = null;
}
// The grid for this spec, reusing the nodes kept under its key.
export function grid(spec) {
    const shape = JSON.stringify([spec.columns, Boolean(spec.onColumn)]);
    let entry = kept.get(spec.key);
    if (!entry || entry.shape !== shape) {
        entry = build(spec);
        entry.shape = shape;
        entry.numeric = spec.columns.map((column) => isNumberType(column.type));
        fill(entry.head, h("th", { class: "rn", attrs: { scope: "col" } }, h("span", { class: "sr" }, "Row")), spec.columns.map((column, i) => headCell(column, entry?.numeric[i] ?? false, Boolean(spec.onColumn))));
        kept.set(spec.key, entry);
    }
    entry.spec = spec;
    entry.table.setAttribute("aria-label", spec.label);
    updateRows(entry, spec.rows, spec.skeleton ?? 0);
    const cells = [...entry.head.cells].slice(1);
    cells.forEach((cell, i) => {
        const picked = spec.columns[i]?.name === spec.column;
        cell.classList.toggle("picked", picked);
        cell.querySelector("button")?.setAttribute("aria-pressed", String(picked));
    });
    if (entry.at)
        mark(entry, entry.at.row, entry.at.col, false);
    fill(entry.foot, spec.foot);
    return entry.box;
}
// Puts each grid on screen back where you left it, after its tab or answer comes back into view.
export function settleGrids(root) {
    for (const entry of kept.values()) {
        if (!root.contains(entry.box) || entry.scroller.offsetParent === null)
            continue;
        if (entry.scroller.scrollTop !== entry.scroll.top)
            entry.scroller.scrollTop = entry.scroll.top;
        if (entry.scroller.scrollLeft !== entry.scroll.left)
            entry.scroller.scrollLeft = entry.scroll.left;
    }
}
// Scrolls a grid sideways just enough to show one column, as when it's picked from the inspector.
export function revealColumn(key, name) {
    const entry = kept.get(key);
    const index = entry?.spec.columns.findIndex((column) => column.name === name) ?? -1;
    const cell = entry && index >= 0 ? entry.head.cells[index + 1] : undefined;
    if (!entry || !cell)
        return;
    const gutter = entry.head.cells[0]?.offsetWidth ?? 0;
    const box = entry.scroller;
    if (cell.offsetLeft - gutter < box.scrollLeft)
        box.scrollLeft = cell.offsetLeft - gutter;
    else if (cell.offsetLeft + cell.offsetWidth > box.scrollLeft + box.clientWidth)
        box.scrollLeft = cell.offsetLeft + cell.offsetWidth - box.clientWidth;
    remember(entry);
}
