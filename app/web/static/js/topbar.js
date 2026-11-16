// The top bar (the name, the tested link, the theme switch) and the sheet that link opens under it.
import { choice } from "./controls.js";
import { CONNECTING, RATES_HIDDEN, SETUP_DIFFERS, TESTED_LINK } from "./copy.js";
import { h, redraw } from "./dom.js";
const EDGE = 16; // the least room the sheet keeps from the window's sides
const GAP = 4; // between the top bar and the sheet
// A sentence's measure: its first number, with the "of N" or the unit that follows it.
const FIGURE = /\d+(?:[.,]\d+)*(?: of \d+(?:,\d+)*| s\b|%)?/;
export function drawTop(root, state, actions) {
    const themes = [{ value: "dark", text: "Dark" }, { value: "light", text: "Light" }];
    redraw(root, h("span", { class: "brand" }, "ShilmanDB"), h("div", { class: "top-end" }, h("button", {
        class: "link",
        attrs: { type: "button", "aria-expanded": String(state.testedOpen), "aria-controls": "tested", "data-focus": "tested-link" },
        onClick: () => actions.toggleTested(),
    }, TESTED_LINK), choice("theme", "Theme", themes, state.theme, (theme) => actions.setTheme(theme))));
}
// One measured sentence as a ledger row: its figure in the fixed left column, the sentence as written beside it.
// A sentence with no figure (where the details are) stays a plain line under the rows.
function fact(line) {
    const figure = FIGURE.exec(line)?.[0];
    return figure
        ? h("li", { class: "entry" }, h("span", { class: "fig" }, figure), h("span", { class: "entry-body" }, line))
        : h("li", { class: "fact-note" }, line);
}
// Sets the sheet under its link, right edges together, kept inside the window.
function place(root) {
    const link = document.querySelector('[data-focus="tested-link"]');
    const bar = document.getElementById("top");
    if (!link || !bar)
        return;
    const width = root.offsetWidth;
    const view = document.documentElement.clientWidth; // what a fixed box measures against: the window less its scrollbar
    const right = Math.max(EDGE, Math.min(view - link.getBoundingClientRect().right, view - EDGE - width));
    root.style.setProperty("--sheet-right", `${right}px`);
    root.style.setProperty("--sheet-top", `${bar.getBoundingClientRect().bottom + GAP}px`);
}
// The tested sheet: the measured numbers as rows, or why they're hidden, with a Close button.
export function drawTested(root, state, actions) {
    root.hidden = !state.testedOpen;
    if (!state.testedOpen) {
        redraw(root);
        return;
    }
    const status = state.status;
    const lines = status?.tested_panel ?? [];
    const hidden = `${RATES_HIDDEN} ${status?.lock.reason ?? SETUP_DIFFERS}.`;
    redraw(root, h("div", { class: "sheet-head" }, h("h2", { attrs: { id: "tested-title" } }, TESTED_LINK), h("button", { class: "btn2", attrs: { type: "button", "data-focus": "tested-close" }, onClick: () => actions.toggleTested() }, "Close")), h("div", { class: "sheet-body" }, lines.length ? h("ul", { class: "facts ledger" }, lines.map(fact)) : h("p", {}, status ? hidden : CONNECTING)));
    place(root);
}
