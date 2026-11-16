// The tab strip: Query first and always there, then one tab per open table. The arrow keys move
// between tabs and open each; Delete, the close mark or a middle-click closes a table's tab.
import { closeMark } from "./controls.js";
import { h, redraw } from "./dom.js";
import { QUERY_TAB } from "./state.js";
// The id of a tab's button, which its panel is labelled by.
export function tabId(key) {
    return key === QUERY_TAB ? "tab-query" : `tab-${key.replace(/[^A-Za-z0-9_-]/g, "-")}`;
}
// Scrolls the strip, and only the strip, just far enough to show one tab.
function revealTab(root, cell) {
    if (!cell)
        return;
    const right = cell.offsetLeft + cell.offsetWidth;
    if (cell.offsetLeft < root.scrollLeft)
        root.scrollLeft = cell.offsetLeft;
    else if (right > root.scrollLeft + root.clientWidth)
        root.scrollLeft = right - root.clientWidth;
}
function focusTab(root, key) {
    const tab = root.querySelector(`#${tabId(key)}`);
    tab?.focus({ preventScroll: true });
    revealTab(root, tab?.parentElement ?? null);
}
export function drawTabs(root, state, actions) {
    const keys = [QUERY_TAB, ...state.tabs.map((tab) => tab.key)];
    const onKey = (event, key) => {
        const at = keys.indexOf(key);
        let next;
        if (event.key === "ArrowRight")
            next = keys[(at + 1) % keys.length];
        else if (event.key === "ArrowLeft")
            next = keys[(at - 1 + keys.length) % keys.length];
        else if (event.key === "Home")
            next = keys[0];
        else if (event.key === "End")
            next = keys[keys.length - 1];
        else if ((event.key === "Delete" || event.key === "Backspace") && key !== QUERY_TAB) {
            event.preventDefault();
            actions.closeTab(key);
            focusTab(root, state.active);
            return;
        }
        if (!next)
            return;
        event.preventDefault();
        actions.activate(next);
        focusTab(root, next);
    };
    // The close mark and the middle-click. Closing the shown tab (or losing focus with the mark) puts focus on the tab shown next.
    const close = (key) => {
        const wasShown = state.active === key;
        actions.closeTab(key);
        const lost = document.activeElement === null || document.activeElement === document.body;
        if (wasShown || lost)
            focusTab(root, state.active);
    };
    const cell = (key, label, closable) => {
        const on = state.active === key;
        const tab = h("button", {
            class: "tab",
            attrs: {
                type: "button", role: "tab", id: tabId(key), "aria-selected": String(on),
                "aria-controls": key === QUERY_TAB ? "query-pane" : "table-pane", tabindex: on ? "0" : "-1", "data-focus": `tab-${key}`,
            },
            onClick: () => actions.activate(key),
            onKeyDown: (event) => onKey(event, key),
        }, label);
        const classes = ["tab-cell", on ? "on" : "", closable ? "closable" : ""].filter(Boolean).join(" ");
        const wrap = h("div", { class: classes, attrs: { role: "presentation" } }, tab, closable ? h("button", {
            class: "tab-close",
            attrs: { type: "button", tabindex: "-1", "aria-label": `Close ${label}`, title: `Close ${label}` },
            onClick: () => close(key),
        }, closeMark()) : null);
        if (closable) {
            // A middle-click closes the tab; holding the middle button down never starts the browser's scroll mode.
            wrap.addEventListener("mousedown", (event) => { if (event.button === 1)
                event.preventDefault(); });
            wrap.addEventListener("auxclick", (event) => {
                if (event.button !== 1)
                    return;
                event.preventDefault();
                close(key);
            });
        }
        return wrap;
    };
    redraw(root, cell(QUERY_TAB, "Query", false), state.tabs.map((tab) => cell(tab.key, tab.table, true)), h("div", { class: "tab-fill", attrs: { role: "presentation" } }));
    revealTab(root, root.querySelector(".tab-cell.on"));
}
