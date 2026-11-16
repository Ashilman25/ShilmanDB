// The chart in place of the table: bars from zero (negative ones on the other side of it), or one large number.
// A few short labels get upright bars with the labels under them; more bars, or longer labels,
// get bars across with the labels on the left. Each bar carries its value as the table writes it, and the
// value axis is labeled at 0 and at its top value.
import { h, svg } from "./dom.js";
const SLOT = 10; // width of one upright bar's slot in the drawing's own units
const BAR = 7; // the bar inside it
const HEIGHT = 100;
const MAX_UPRIGHT = 8; // more bars than this lie across
const LABEL_ROOM = 64; // characters of label the upright chart's width holds, shared by its bars
const ROW = 22; // pixels per bar lying across
const DENSE_ROW = 16; // the same from DENSE_FROM bars on, so a long chart stays short
const DENSE_FROM = 13;
// Where each value sits on the drawing's 0 to 100 scale, with zero included.
function scale(chart) {
    const values = chart.points.map(([, value]) => value);
    const top = Math.max(0, ...values);
    const bottom = Math.min(0, ...values);
    const span = top - bottom || 1;
    const at = (value) => ((value - bottom) / span) * 100;
    return { at, zero: at(0), negative: bottom < 0 };
}
// A bar's value as the table shows it, from the same row.
function valueText(view, chart, i) {
    return view.rows[i]?.[chart.value] ?? String(chart.points[i]?.[1] ?? "");
}
// The value axis in words: 0, the top value, and the lowest one when bars go under zero.
function axisMarks(chart, view, zero) {
    const values = chart.points.map(([, value]) => value);
    const high = values.indexOf(Math.max(...values));
    const low = values.indexOf(Math.min(...values));
    const zeroMark = h("span", { class: "axis-zero" }, "0");
    zeroMark.style.setProperty("--zero", String(zero));
    const top = (values[high] ?? 0) > 0 ? valueText(view, chart, high) : null;
    const bottom = (values[low] ?? 0) < 0 ? valueText(view, chart, low) : null;
    const widest = [top ?? "", bottom ?? "", "0"].reduce((a, b) => (b.length > a.length ? b : a));
    return [
        h("span", { class: "axis-size" }, widest), // gives the axis its width; never seen
        zeroMark,
        top !== null ? h("span", { class: "axis-top" }, top) : null,
        bottom !== null ? h("span", { class: "axis-low" }, bottom) : null,
    ].filter((mark) => mark !== null);
}
function describe(chart, view) {
    const label = view.columns[chart.label]?.name ?? "label";
    const value = view.columns[chart.value]?.name ?? "value";
    return `Bars of ${value} by ${label}.`;
}
// Upright bars, each value over its bar, the labels under them, the axis on the left.
function upright(chart, view) {
    const { at, zero, negative } = scale(chart);
    const y = (value) => HEIGHT - at(value);
    const base = HEIGHT - zero;
    const width = chart.points.length * SLOT;
    const drawing = svg("svg", { viewBox: `0 0 ${width} ${HEIGHT}`, preserveAspectRatio: "none", role: "img", "aria-label": describe(chart, view) }, chart.points.map(([name, amount], i) => svg("rect", {
        class: "bar", x: i * SLOT + (SLOT - BAR) / 2, width: BAR, y: Math.min(y(amount), base), height: Math.abs(y(amount) - base),
    }, svg("title", {}, name))), svg("line", { class: "zero", x1: 0, x2: width, y1: base, y2: base }));
    const values = chart.points.map(([, amount], i) => {
        const text = valueText(view, chart, i);
        const tag = h("span", { class: amount < 0 ? "bar-value below" : "bar-value", attrs: { title: text } }, text);
        tag.style.setProperty("--i", String(i));
        tag.style.setProperty("--at", String(at(amount)));
        return tag;
    });
    const plot = h("div", { class: negative ? "plot negative" : "plot", attrs: { "aria-hidden": "true" } }, values);
    plot.prepend(drawing);
    plot.style.setProperty("--count", String(chart.points.length));
    const labels = h("div", { class: "labels", attrs: { "aria-hidden": "true" } }, chart.points.map(([name]) => h("span", { attrs: { title: name } }, name)));
    labels.style.setProperty("--count", String(chart.points.length));
    const axis = h("div", { class: "axis", attrs: { "aria-hidden": "true" } }, axisMarks(chart, view, zero));
    return h("figure", { class: "chart upright" }, axis, plot, h("span"), labels);
}
// Bars lying across, each label in full on its left (cut only past a wide cap), each value on its right,
// the axis under them.
function across(chart, view) {
    const { at, zero, negative } = scale(chart);
    const dense = chart.points.length >= DENSE_FROM;
    const row = dense ? DENSE_ROW : ROW;
    const inset = dense ? 3 : 5;
    const height = chart.points.length * row;
    const drawing = svg("svg", {
        viewBox: `0 0 100 ${height}`, preserveAspectRatio: "none", height, role: "img", "aria-label": describe(chart, view),
    }, chart.points.map(([name, amount], i) => svg("rect", {
        class: "bar", x: Math.min(at(amount), zero), width: Math.abs(at(amount) - zero), y: i * row + inset, height: row - 2 * inset,
    }, svg("title", {}, name))), negative ? svg("line", { class: "zero", x1: zero, x2: zero, y1: 0, y2: height }) : null);
    const labels = h("div", { class: "row-labels", attrs: { "aria-hidden": "true" } }, chart.points.map(([name]) => h("span", { attrs: { title: name } }, name)));
    const values = h("div", { class: "row-values", attrs: { "aria-hidden": "true" } }, chart.points.map((_, i) => h("span", { class: "bar-value" }, valueText(view, chart, i))));
    const axis = h("div", { class: "axis", attrs: { "aria-hidden": "true" } }, axisMarks(chart, view, zero));
    return h("figure", { class: dense ? "chart across dense" : "chart across" }, labels, drawing, values, h("span"), axis, h("span"));
}
export function chartFor(view) {
    const chart = view.chart;
    if (chart.kind === "bars") {
        const count = chart.points.length;
        const longest = Math.max(...chart.points.map(([name]) => name.length));
        return count <= MAX_UPRIGHT && longest <= Math.floor(LABEL_ROOM / count) ? upright(chart, view) : across(chart, view);
    }
    if (chart.kind === "number") {
        return h("figure", { class: "big" }, h("span", { class: "big-number" }, chart.text), h("figcaption", {}, view.columns[chart.value]?.name ?? ""));
    }
    return null;
}
