// The one way the page makes nodes. Text always goes in as text nodes, so nothing from the
// server is ever read as markup.

export type Child = Node | string | null | undefined | false | Child[];
type AttrValue = string | number | boolean | null | undefined;

export interface Props {
  class?: string;
  attrs?: Record<string, AttrValue>;
  onClick?: (event: MouseEvent) => void;
  onKeyDown?: (event: KeyboardEvent) => void;
  onInput?: (event: Event) => void;
  onChange?: (event: Event) => void;
}

const SVG_NS = "http://www.w3.org/2000/svg";

// Sets attributes; false, null and undefined leave one out, true sets it empty.
// Inline handlers and inline styles are refused: the page's CSP blocks them anyway.
function setAttrs(node: Element, attrs: Record<string, AttrValue>): void {
  for (const [name, value] of Object.entries(attrs)) {
    if (/^on/i.test(name) || name.toLowerCase() === "style") throw new Error(`attribute ${name} isn't allowed`);
    if (value === null || value === undefined || value === false) continue;
    node.setAttribute(name, value === true ? "" : String(value));
  }
}

function append(node: Element, children: Child[]): void {
  for (const child of children) {
    if (Array.isArray(child)) append(node, child);
    else if (child !== null && child !== undefined && child !== false) node.append(child);
  }
}

// An HTML element with its class, attributes, listeners and children.
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, props: Props = {}, ...children: Child[]): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  const element: HTMLElement = node;
  if (props.class) element.className = props.class;
  if (props.attrs) setAttrs(element, props.attrs);
  if (props.onClick) element.addEventListener("click", props.onClick);
  if (props.onKeyDown) element.addEventListener("keydown", props.onKeyDown);
  if (props.onInput) element.addEventListener("input", props.onInput);
  if (props.onChange) element.addEventListener("change", props.onChange);
  append(element, children);
  return node;
}

// An SVG element with its attributes and children.
export function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, AttrValue> = {}, ...children: Child[]): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  setAttrs(node, attrs);
  append(node, children);
  return node;
}

interface Scroll {
  left: number;
  top: number;
}

// Replaces a panel's children. Focus and scroll positions (both ways) come back on the new nodes
// that carry the same data-focus or data-scroll key, and the panel keeps its own scroll, so a full
// redraw keeps your place: a table scrolled to row 500 stays there through "Show 500 more" or a verdict.
export function redraw(panel: Element, ...children: Child[]): void {
  const active = document.activeElement;
  const focusKey = active instanceof HTMLElement && panel.contains(active) ? active.dataset["focus"] : undefined;
  const own: Scroll = { left: panel.scrollLeft, top: panel.scrollTop };
  const scrolled = new Map<string, Scroll>();
  for (const node of panel.querySelectorAll("[data-scroll]")) {
    if (node instanceof HTMLElement && node.dataset["scroll"]) {
      scrolled.set(node.dataset["scroll"], { left: node.scrollLeft, top: node.scrollTop });
    }
  }

  panel.replaceChildren();
  append(panel, children);

  panel.scrollLeft = own.left;
  panel.scrollTop = own.top;
  for (const node of panel.querySelectorAll("[data-scroll]")) {
    const saved = node instanceof HTMLElement && node.dataset["scroll"] ? scrolled.get(node.dataset["scroll"]) : undefined;
    if (saved) {
      node.scrollLeft = saved.left;
      node.scrollTop = saved.top;
    }
  }
  if (focusKey === undefined) return;
  for (const node of panel.querySelectorAll("[data-focus]")) {
    if (node instanceof HTMLElement && node.dataset["focus"] === focusKey) {
      node.focus({ preventScroll: true });
      return;
    }
  }
}

// The element with this id from index.html; the page can't run without it.
export function byId(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`index.html has no #${id}`);
  return node;
}
