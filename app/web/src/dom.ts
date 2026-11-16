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

interface Place {
  focusKey: string | undefined;
  own: Scroll;
  scrolled: Map<string, Scroll>;
}

function flatten(children: Child[], out: Node[] = []): Node[] {
  for (const child of children) {
    if (Array.isArray(child)) flatten(child, out);
    else if (typeof child === "string") out.push(document.createTextNode(child));
    else if (child !== null && child !== undefined && child !== false) out.push(child);
  }
  return out;
}

// Gives a node exactly these children. A child that was already there stays in place unless the
// order changes, so a kept table keeps its layout and its scroll.
export function fill(node: Element, ...children: Child[]): void {
  const next = flatten(children);
  const keep = new Set(next);
  for (const old of [...node.childNodes]) {
    if (!keep.has(old)) old.remove();
  }
  let at = node.firstChild;
  for (const child of next) {
    if (child === at) at = at.nextSibling;
    else node.insertBefore(child, at);
  }
}

function savePlace(panel: Element): Place {
  const active = document.activeElement;
  const focusKey = active instanceof HTMLElement && panel.contains(active) ? active.dataset["focus"] : undefined;
  const scrolled = new Map<string, Scroll>();
  for (const node of panel.querySelectorAll("[data-scroll]")) {
    if (node instanceof HTMLElement && node.dataset["scroll"]) {
      scrolled.set(node.dataset["scroll"], { left: node.scrollLeft, top: node.scrollTop });
    }
  }
  return { focusKey, own: { left: panel.scrollLeft, top: panel.scrollTop }, scrolled };
}

function restorePlace(panel: Element, place: Place): void {
  panel.scrollLeft = place.own.left;
  panel.scrollTop = place.own.top;
  for (const node of panel.querySelectorAll("[data-scroll]")) {
    const saved = node instanceof HTMLElement && node.dataset["scroll"] ? place.scrolled.get(node.dataset["scroll"]) : undefined;
    if (saved) {
      node.scrollLeft = saved.left;
      node.scrollTop = saved.top;
    }
  }
  if (place.focusKey === undefined) return;
  for (const node of panel.querySelectorAll("[data-focus]")) {
    if (node instanceof HTMLElement && node.dataset["focus"] === place.focusKey) {
      node.focus({ preventScroll: true });
      return;
    }
  }
}

// Puts every scroller in a panel back at the top, so the next redraw doesn't carry its place over.
export function resetScrolls(panel: Element): void {
  for (const node of panel.querySelectorAll("[data-scroll]")) {
    node.scrollTop = 0;
    node.scrollLeft = 0;
  }
}

// Replaces a panel's children. Focus and scroll come back on new nodes with the same
// data-focus or data-scroll key, so a full redraw keeps your place.
export function redraw(panel: Element, ...children: Child[]): void {
  redrawFrom(panel, () => children);
}

// The same, for a panel that reuses nodes already on screen: the place is read before build runs,
// since build may move a kept node out of the panel.
export function redrawFrom(panel: Element, build: () => Child[]): void {
  const place = savePlace(panel);
  fill(panel, build());
  restorePlace(panel, place);
}

// The element with this id from index.html; the page can't run without it.
export function byId(id: string): HTMLElement {
  const node = document.getElementById(id);
  if (!node) throw new Error(`index.html has no #${id}`);
  return node;
}
