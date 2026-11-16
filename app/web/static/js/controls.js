// Small pieces more than one panel draws: the switch, a status word, skeleton lines, a banner,
// and the page's only two drawn marks (the tree's chevron and a tab's close mark).
import { h, svg } from "./dom.js";
// A switch made of real radio buttons, so arrow keys and screen readers work as usual.
// Each radio is named by its visible label, never by its value.
export function choice(name, label, options, current, pick, strong = false) {
    return h("div", { class: strong ? "seg strong" : "seg", attrs: { role: "radiogroup", "aria-label": label } }, options.map((option) => h("label", {}, h("input", {
        attrs: {
            type: "radio", name, value: option.value, checked: option.value === current,
            "aria-label": option.text, "data-focus": `${name}-${option.value}`,
        },
        onChange: () => pick(option.value),
    }), h("span", {}, option.text))));
}
// Status words and the color that backs each one up; the word always carries the meaning.
const WORD_TONES = {
    Pass: "pass", Same: "pass",
    Note: "quiet", Running: "quiet", Queued: "quiet", "Writing SQL": "quiet",
    Warning: "warn", Differs: "warn",
    Refused: "bad", "Not run": "bad",
};
export function word(text) {
    return h("span", { class: `word ${WORD_TONES[text] ?? "quiet"}` }, text);
}
// Grey lines that hold the place of text still on its way.
export function skeleton(widths) {
    return h("span", { class: "skeleton", attrs: { "aria-hidden": "true" } }, widths.map((width) => h("span", { class: `skel w${width}` })));
}
const BANNER_TONES = { warned: "warn", refused: "bad", not_run: "bad" };
export const JOB_FAILED_LEAD = "Couldn't finish.";
export const SQL_FAILED_LEAD = "Couldn't run.";
const LEADS = ["Warning.", "Can't answer this yet.", "Not run.", "No answer.", JOB_FAILED_LEAD, SQL_FAILED_LEAD];
function toneOf(kind) {
    return kind === "error" ? "neutral" : (BANNER_TONES[kind] ?? "neutral");
}
// A verdict's words with a known first sentence set in bold, in the tone the banner gives it.
export function leadText(kind, text) {
    const lead = LEADS.find((start) => text.startsWith(start));
    return lead ? [h("strong", { class: `lead ${toneOf(kind)}` }, lead), text.slice(lead.length)] : [text];
}
// A banner for a verdict or an error.
export function banner(kind, text, ...after) {
    return h("div", { class: `banner ${toneOf(kind)}` }, h("p", {}, leadText(kind, text)), after);
}
// A 16px mark in one 1.5px stroke, hidden from screen readers: its control carries the name.
function mark(className, path) {
    return svg("svg", { class: `icon ${className}`, viewBox: "0 0 16 16", width: 16, height: 16, "aria-hidden": "true", focusable: "false" }, svg("path", { d: path, fill: "none", stroke: "currentColor", "stroke-width": 1.5, "stroke-linecap": "round", "stroke-linejoin": "round" }));
}
// Points right when closed; the open row turns it down in CSS.
export function chevron() {
    return mark("chevron", "M6 4l4 4-4 4");
}
export function closeMark() {
    return mark("close-mark", "M5 5l6 6M11 5l-6 6");
}
