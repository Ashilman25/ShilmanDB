// The top bar (name, dataset switch, what's loaded, the tested link, the theme switch)
// and the inline panel that link opens.
import { choice } from "./controls.js";
import { CONNECTING, RATES_HIDDEN, SETUP_DIFFERS, TESTED_LINK } from "./copy.js";
import { h, redraw } from "./dom.js";
import type { Actions, State } from "./state.js";
import type { Theme } from "./theme.js";
import type { DatasetStatus, Status } from "./types.js";

const QUIET_STATES = ["loading", "ready", "off"];

// The current dataset's state in words. Missing files or a failed load are the data's problem;
// a missing dataset with only the build hint means the engine bridge isn't built.
function datasetWords(info: DatasetStatus): string {
  if (info.state === "missing") return info.detail.startsWith("Missing:") ? "Data missing" : "Engine missing";
  if (info.state === "failed") return "Data failed";
  return `Engine ${info.state}`;
}

// The prompt cache, for the live model only: on, or off with the reason in plain sight.
function cacheWords(status: Status): string | null {
  if (status.model.mode !== "mlx") return null;
  return status.cache.on ? "Prompt cache on" : `Prompt cache off: ${status.cache.reason ?? "no reason given"}`;
}

// What's loaded, in words: the current dataset's engine, then the model and its prompt cache.
function loaded(state: State): HTMLElement {
  if (state.statusError) {
    return h("span", { class: "loaded" }, h("span", { class: "state bad", attrs: { title: state.statusError } }, "Not connected"));
  }
  const status = state.status;
  if (!status) return h("span", { class: "loaded" }, h("span", { class: "state" }, "Connecting"));

  const engine = status.datasets[state.dataset];
  const model = status.model;
  const recorded = model.mode === "recorded" && model.state === "ready" ? " (recorded)" : "";
  const cache = cacheWords(status);
  return h("span", { class: "loaded" },
    engine ? h("span", { class: `state${QUIET_STATES.includes(engine.state) ? "" : " bad"}`, attrs: { title: engine.detail || null } },
      datasetWords(engine)) : null,
    h("span", { class: `state${QUIET_STATES.includes(model.state) ? "" : " bad"}`, attrs: { title: model.detail || null } },
      `Model ${model.state}${recorded}`),
    cache ? h("span", { class: "state" }, cache) : null);
}

export function drawTop(root: HTMLElement, state: State, actions: Actions): void {
  const datasets = Object.entries(state.status?.datasets ?? {}).map(([value, info]) => ({ value, text: info.label }));
  const themes: { value: Theme; text: string }[] = [{ value: "dark", text: "Dark" }, { value: "light", text: "Light" }];
  redraw(root,
    h("div", { class: "top-start" },
      h("span", { class: "brand" }, "ShilmanDB"),
      datasets.length ? choice("dataset", "Dataset", datasets, state.dataset, (name) => actions.setDataset(name)) : null,
      loaded(state)),
    h("div", { class: "top-end" },
      h("button", {
        class: "link",
        attrs: { type: "button", "aria-expanded": String(state.testedOpen), "aria-controls": "tested", "data-focus": "tested-link" },
        onClick: () => actions.toggleTested(),
      }, TESTED_LINK),
      choice("theme", "Theme", themes, state.theme, (theme) => actions.setTheme(theme))));
}

// The tested panel: the measured numbers, one sentence a line, or why they're hidden.
export function drawTested(root: HTMLElement, state: State): void {
  root.hidden = !state.testedOpen;
  if (!state.testedOpen) {
    redraw(root);
    return;
  }

  const status = state.status;
  const lines = status?.tested_panel ?? [];
  const hidden = `${RATES_HIDDEN} ${status?.lock.reason ?? SETUP_DIFFERS}.`;
  redraw(root,
    h("div", { class: "tested-inner" },
      h("h2", {}, TESTED_LINK),
      lines.length ? h("ul", { class: "facts" }, lines.map((line) => h("li", {}, line))) : h("p", {}, status ? hidden : CONNECTING)));
}
