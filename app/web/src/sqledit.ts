// The SQL editor's extras around a plain textarea: the colors and line numbers drawn behind its text
// (built from text nodes, never markup), table and column names offered as you type, and the splitter
// that sets its height. The textarea stays the one thing you type into.
import { fill, h } from "./dom.js";
import { namesAt, pieces, type Name } from "./sqltext.js";
import type { State } from "./state.js";

const MIN_HEIGHT = 96;
const START_HEIGHT = 140;
const STEP = 16;
const MAX_SHARE = 0.6;  // of the tab's height
const STACKED = "(max-width: 760px)";  // where the page scrolls as a whole, as in app.css

// Keys that move the caret away from the word, which closes the list.
const MOVES_CARET = ["ArrowLeft", "ArrowRight", "Home", "End", "PageUp", "PageDown"];

export interface SqlEditor {
  render(): void;
  draw(state: State, shown: boolean): void;
  onKey(event: KeyboardEvent): boolean;  // true when the list used the key
  close(): void;
}

export function makeSqlEditor(sql: HTMLTextAreaElement, box: HTMLElement, lines: HTMLElement, splitter: HTMLElement,
                              pane: HTMLElement): SqlEditor {
  const list = h("ul", { class: "complete", attrs: { id: "complete", role: "listbox", "aria-label": "Tables and columns" } });
  list.hidden = true;
  document.body.append(list);
  sql.setAttribute("aria-autocomplete", "list");
  sql.setAttribute("aria-controls", "complete");

  let state: State | null = null;
  let rows: { node: HTMLElement; start: number }[] = [];
  // The open list: its names, the highlighted one, whether Up or Down moved it, and where the word starts.
  let open: { choices: Name[]; at: number; moved: boolean; from: number } | null = null;
  let taking = false;  // a picked name going in, which mustn't open the list again
  let height = START_HEIGHT;

  // The overlay: one row per line, its number in the gutter, its words in color.
  function render(): void {
    const text = sql.value;
    const made: HTMLElement[] = [];
    rows = [];
    let row = h("div", { class: "row" });
    let start = 0;
    let offset = 0;
    const newRow = () => {
      made.push(row);
      rows.push({ node: row, start });
      row.prepend(h("span", { class: "ln" }, String(rows.length)));
      row = h("div", { class: "row" });
    };
    for (const [kind, piece] of pieces(text)) {
      const parts = piece.split("\n");
      parts.forEach((part, i) => {
        if (i > 0) {
          newRow();
          start = offset + 1;
          offset += 1;
        }
        if (part) row.append(kind ? h("span", { class: kind }, part) : document.createTextNode(part));
        offset += part.length;
      });
    }
    newRow();
    box.style.setProperty("--digits", String(Math.max(2, String(rows.length).length)));
    fill(lines, h("div", { class: "code-inner" }, made));
    follow();
  }

  // Keeps the overlay under the text: the same scroll, and room for the textarea's scrollbar.
  function follow(): void {
    const style = getComputedStyle(sql);
    const bar = sql.offsetWidth - sql.clientWidth - parseFloat(style.borderLeftWidth) - parseFloat(style.borderRightWidth);
    lines.style.setProperty("--bar", `${Math.max(0, bar)}px`);
    lines.scrollTop = sql.scrollTop;
    lines.scrollLeft = sql.scrollLeft;
  }

  // The tables, and every table's columns, of the database the Query tab uses.
  function names(): Name[] {
    const tables = state?.schema?.datasets[state.dataset]?.tables ?? [];
    return [
      ...tables.map((table) => ({ name: table.name, note: "table", table: null })),
      ...tables.flatMap((table) => table.columns.map((column) => ({ name: column.name, note: `${table.name} ${column.type}`, table: table.name }))),
    ];
  }

  // Where the caret's word sits on screen, read from the overlay, which lines up with the text.
  function wordBox(from: number, caret: number): DOMRect | null {
    const at = from < caret ? from : caret - 1;
    let row = rows[0];
    for (const next of rows) if (next.start <= at) row = next;
    if (!row || at < 0) return null;
    let left = at - row.start;
    const walk = document.createTreeWalker(row.node, NodeFilter.SHOW_TEXT);
    walk.nextNode();  // the line number
    for (let node = walk.nextNode(); node; node = walk.nextNode()) {
      const length = node.textContent?.length ?? 0;
      if (left < length) {
        const range = document.createRange();
        range.setStart(node, left);
        range.setEnd(node, left + 1);
        const rect = range.getBoundingClientRect();
        return from < caret ? rect : new DOMRect(rect.right, rect.top, 0, rect.height);
      }
      left -= length;
    }
    return null;
  }

  function place(from: number): void {
    const rect = wordBox(from, sql.selectionStart);
    if (!rect) return;
    const width = list.offsetWidth;
    const view = document.documentElement;  // what a fixed box measures against: the window less its scrollbars
    const left = Math.max(8, Math.min(rect.left - 4, view.clientWidth - width - 8));
    const below = rect.bottom + 2;
    const top = below + list.offsetHeight > view.clientHeight - 8 ? Math.max(8, rect.top - 2 - list.offsetHeight) : below;
    list.style.setProperty("--at-left", `${left}px`);
    list.style.setProperty("--at-top", `${top}px`);
  }

  function drawList(): void {
    if (!open) return;
    const current = open;
    // Until Up or Down moves it, the first name is marked quietly: Tab takes it, Enter still starts a new line.
    const marked = current.moved ? "choice on" : "choice first";
    fill(list, current.choices.map((choice, i) => {
      const item = h("li", {
        class: i === current.at ? marked : "choice",
        attrs: { id: `complete-${i}`, role: "option", "aria-selected": String(i === current.at) },
      }, h("span", { class: "choice-name" }, choice.name), h("span", { class: "choice-note" }, choice.note));
      item.addEventListener("mousedown", (event) => event.preventDefault());  // the editor keeps focus
      item.addEventListener("click", () => take(i));
      return item;
    }));
    sql.setAttribute("aria-activedescendant", `complete-${current.at}`);
    list.hidden = false;
    place(current.from);
  }

  function close(): void {
    open = null;
    list.hidden = true;
    sql.removeAttribute("aria-activedescendant");
  }

  // Opens, narrows or closes the list for the word at the caret. Only typing opens it.
  function suggest(typed: boolean): void {
    const caret = sql.selectionStart;
    const offer = caret === sql.selectionEnd ? namesAt(sql.value, caret, names()) : null;
    if (!offer || (!open && !typed)) {
      close();
      return;
    }
    open = { choices: offer.choices, at: 0, moved: false, from: offer.from };
    drawList();
  }

  // Puts a picked name in place of the typed part of the word, keeping the editor's undo.
  function take(index: number): void {
    const choice = open?.choices[index];
    if (!open || !choice) return;
    const caret = sql.selectionStart;
    sql.setSelectionRange(open.from, caret);
    close();
    taking = true;
    if (!document.execCommand("insertText", false, choice.name)) {
      sql.setRangeText(choice.name, sql.selectionStart, caret, "end");
      sql.dispatchEvent(new Event("input"));
    }
    taking = false;
  }

  function onKey(event: KeyboardEvent): boolean {
    if (!open || event.isComposing) return false;
    if ((event.key === "Enter" && (event.metaKey || event.ctrlKey)) || MOVES_CARET.includes(event.key)) {
      close();
      return false;
    }
    const current = open;
    const size = current.choices.length;
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      // The first press only arms the quietly marked first name; later presses move.
      if (current.moved) current.at = (current.at + (event.key === "ArrowDown" ? 1 : size - 1)) % size;
      else if (event.key === "ArrowUp") current.at = size - 1;
      current.moved = true;
      drawList();
    } else if ((event.key === "Tab" || (event.key === "Enter" && current.moved)) && !event.shiftKey && !event.altKey) {
      take(current.at);
    } else if (event.key === "Escape") {
      close();
      event.stopPropagation();
    } else {
      return false;
    }
    event.preventDefault();
    return true;
  }

  // The splitter: the editor's height between 96px and 60% of the tab, by drag or by the arrow keys.
  // The tab is the pane when it has a height of its own; hidden, or stacked on a narrow screen, it's the window.
  function maxHeight(): number {
    const view = document.documentElement.clientHeight;
    const tab = window.matchMedia(STACKED).matches || pane.clientHeight === 0 ? view : pane.clientHeight;
    return Math.max(MIN_HEIGHT, Math.floor(tab * MAX_SHARE));
  }

  function setHeight(next: number): void {
    height = Math.round(Math.max(MIN_HEIGHT, Math.min(maxHeight(), next)));
    box.style.setProperty("--code-h", `${height}px`);
    splitter.setAttribute("aria-valuenow", String(height));
    splitter.setAttribute("aria-valuemin", String(MIN_HEIGHT));
    splitter.setAttribute("aria-valuemax", String(maxHeight()));
    follow();
  }

  splitter.addEventListener("keydown", (event) => {
    const moves: Record<string, number> = { ArrowUp: -STEP, ArrowDown: STEP };
    if (event.key in moves) setHeight(height + (moves[event.key] ?? 0));
    else if (event.key === "Home") setHeight(MIN_HEIGHT);
    else if (event.key === "End") setHeight(maxHeight());
    else return;
    event.preventDefault();
  });
  splitter.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = height;
    splitter.setPointerCapture(event.pointerId);
    splitter.classList.add("dragging");
    const move = (next: PointerEvent) => setHeight(startHeight + next.clientY - startY);
    const done = () => {
      splitter.classList.remove("dragging");
      splitter.removeEventListener("pointermove", move);
      splitter.removeEventListener("pointerup", done);
      splitter.removeEventListener("pointercancel", done);
    };
    splitter.addEventListener("pointermove", move);
    splitter.addEventListener("pointerup", done);
    splitter.addEventListener("pointercancel", done);
  });

  sql.addEventListener("input", (event) => {
    render();
    if (taking) return;
    suggest(event instanceof InputEvent && event.inputType === "insertText");
  });
  sql.addEventListener("scroll", () => {
    follow();
    if (open) place(open.from);
  });
  sql.addEventListener("blur", close);
  sql.addEventListener("click", close);
  window.addEventListener("resize", () => {
    setHeight(height);
    close();
  });
  new ResizeObserver(() => follow()).observe(sql);
  setHeight(START_HEIGHT);
  render();

  return {
    render,
    draw(next, shown) {
      state = next;
      box.hidden = !shown;
      splitter.hidden = !shown;
      if (!shown) close();
      else if (open) suggest(false);
    },
    onKey,
    close,
  };
}
