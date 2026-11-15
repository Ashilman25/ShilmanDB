// Small pieces more than one panel draws: the switch, a status word, skeleton lines, a banner.
import { h } from "./dom.js";
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
// A banner for a verdict or an error; a known first sentence is set in bold, in the banner's tone.
export function banner(kind, text, ...after) {
    const tone = kind === "error" ? "neutral" : (BANNER_TONES[kind] ?? "neutral");
    const lead = LEADS.find((start) => text.startsWith(start));
    const body = lead
        ? [h("strong", { class: `lead ${tone}` }, lead), text.slice(lead.length)]
        : [text];
    return h("div", { class: `banner ${tone}` }, h("p", {}, body), after);
}
