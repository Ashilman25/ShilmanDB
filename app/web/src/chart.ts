// The chart above the table: bars from zero (negative ones on the other side of it), or one large number.
// A few short labels get upright bars with the labels under them; more bars, or longer labels,
// get bars across with the labels on the left. Bars print no values; the table has the numbers.
import { h, svg } from "./dom.js";
import type { AnswerView, ChartBars } from "./types.js";

const SLOT = 10;        // width of one upright bar's slot in the drawing's own units
const BAR = 7;          // the bar inside it
const HEIGHT = 100;
const MAX_UPRIGHT = 8;  // more bars than this lie across
const LABEL_ROOM = 64;  // characters of label the upright chart's width holds, shared by its bars
const ROW = 22;         // pixels per bar lying across

// Where each value sits on the drawing's 0 to 100 scale, with zero included.
function scale(chart: ChartBars): { at: (value: number) => number; zero: number; negative: boolean } {
  const values = chart.points.map(([, value]) => value);
  const top = Math.max(0, ...values);
  const bottom = Math.min(0, ...values);
  const span = top - bottom || 1;
  const at = (value: number) => ((value - bottom) / span) * 100;
  return { at, zero: at(0), negative: bottom < 0 };
}

function describe(chart: ChartBars, view: AnswerView): string {
  const label = view.columns[chart.label]?.name ?? "label";
  const value = view.columns[chart.value]?.name ?? "value";
  return `Bars of ${value} by ${label}.`;
}

// Upright bars, the labels under them.
function upright(chart: ChartBars, view: AnswerView): HTMLElement {
  const { at, zero } = scale(chart);
  const y = (value: number) => HEIGHT - at(value);
  const base = HEIGHT - zero;
  const width = chart.points.length * SLOT;
  const drawing = svg("svg", { viewBox: `0 0 ${width} ${HEIGHT}`, preserveAspectRatio: "none", role: "img", "aria-label": describe(chart, view) },
    chart.points.map(([name, amount], i) => svg("rect", {
      class: "bar", x: i * SLOT + (SLOT - BAR) / 2, width: BAR, y: Math.min(y(amount), base), height: Math.abs(y(amount) - base),
    }, svg("title", {}, name))),
    svg("line", { class: "zero", x1: 0, x2: width, y1: base, y2: base }));

  const labels = h("div", { class: "labels", attrs: { "aria-hidden": "true" } },
    chart.points.map(([name]) => h("span", { attrs: { title: name } }, name)));
  labels.style.setProperty("--count", String(chart.points.length));
  return h("figure", { class: "chart upright" }, drawing, labels);
}

// Bars lying across, each label in full on its left (cut only past a wide cap).
function across(chart: ChartBars, view: AnswerView): HTMLElement {
  const { at, zero, negative } = scale(chart);
  const height = chart.points.length * ROW;
  const drawing = svg("svg", {
    viewBox: `0 0 100 ${height}`, preserveAspectRatio: "none", height, role: "img", "aria-label": describe(chart, view),
  },
    chart.points.map(([name, amount], i) => svg("rect", {
      class: "bar", x: Math.min(at(amount), zero), width: Math.abs(at(amount) - zero), y: i * ROW + 5, height: ROW - 10,
    }, svg("title", {}, name))),
    negative ? svg("line", { class: "zero", x1: zero, x2: zero, y1: 0, y2: height }) : null);

  const labels = h("div", { class: "row-labels", attrs: { "aria-hidden": "true" } },
    chart.points.map(([name]) => h("span", { attrs: { title: name } }, name)));
  return h("figure", { class: "chart across" }, labels, drawing);
}

export function chartFor(view: AnswerView): HTMLElement | null {
  const chart = view.chart;
  if (chart.kind === "bars") {
    const count = chart.points.length;
    const longest = Math.max(...chart.points.map(([name]) => name.length));
    return count <= MAX_UPRIGHT && longest <= Math.floor(LABEL_ROOM / count) ? upright(chart, view) : across(chart, view);
  }
  if (chart.kind === "number") {
    return h("figure", { class: "big" },
      h("span", { class: "big-number" }, chart.text),
      h("figcaption", {}, view.columns[chart.value]?.name ?? ""));
  }
  return null;
}
