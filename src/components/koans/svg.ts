const SVGNS = "http://www.w3.org/2000/svg";

// Text between asterisks is set in italics, for the titles of texts.
function fill(e: Element, text: string, italic: () => Element) {
  text.split(/\*(.+?)\*/).forEach((part, j) => {
    if (!part) return;
    if (j % 2 === 0) { e.append(part); return; }
    const i = italic();
    i.textContent = part;
    e.append(i);
  });
}

/** Text without the asterisks that mark italics, for an aria-label. */
export const plain = (text: string) => text.replace(/\*/g, "");

/** A text's English name in italics, or its Chinese title. */
export const title = (text: [string, string, string | null]) => text[2] ? `*${text[2]}*` : text[1];

export function svgEl(tag: string, attrs: Record<string, string | number> = {}, text?: string | number) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  if (text != null) fill(e, String(text), () => { const t = document.createElementNS(SVGNS, "tspan"); t.setAttribute("font-style", "italic"); return t; });
  return e;
}


export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string | null, text?: string | number) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) fill(e, String(text), () => document.createElement("em"));
  return e;
}

/** Reads the JSON that a figure's component wrote into the page. */
export function dataOf<T>(root: Element): T {
  return JSON.parse(root.querySelector("script[type='application/json']")!.textContent!);
}

/** A name for a text: its English name and Chinese title, or the title alone. */
export function name(text: [string, string, string | null]) {
  const [, zh, en] = text;
  const span = el("span");
  if (en) span.append(en + " ");
  span.append(el("span", "zh", zh));
  return span;
}

/** A passage in Chinese, closed with an ellipsis if it stops inside a quotation,
 * and its English gloss if the figure has one. */
export function passage(zh: string, gloss: Record<string, string>, cls = "") {
  const open = (zh.match(/[「『]/g) ?? []).length, shut = (zh.match(/[」』]/g) ?? []).length;
  const out = document.createDocumentFragment();
  out.append(el("p", `passage zh ${cls}`.trim(), open > shut ? zh + "…" : zh));
  if (gloss[zh]) out.append(el("p", "gloss", gloss[zh]));
  return out;
}

/** Places dots at their true x. A dot stacks up only to clear the dots placed before it. */
export function swarm<T>(items: T[], x: (t: T) => number, gap: number) {
  const placed: { x: number; level: number }[] = [];
  return items.map(item => {
    const at = x(item);
    let level = 0;
    while (placed.some(p => p.level === level && Math.abs(p.x - at) < gap)) level++;
    placed.push({ x: at, level });
    return { item, x: at, level };
  });
}

/** Makes a chart's marks a list with one Tab stop: the arrow keys, Home and End move
 * the selection, and a click picks the nearest mark within reach of the pointer.
 * The keys step through the marks in their order, or left to right if byX. */
export function listbox(svg: SVGSVGElement, marks: SVGElement[], at: () => { x: number; y: number }[], go: (i: number) => void, byX = false) {
  svg.setAttribute("role", "listbox");
  marks.forEach((m, i) => {
    m.setAttribute("role", "option");
    m.addEventListener("keydown", e => {
      const p = at(), order = marks.map((_, j) => j);
      if (byX) order.sort((a, b) => p[a].x - p[b].x || p[b].y - p[a].y);
      const here = order.indexOf(i), k = (e as KeyboardEvent).key;
      const j = { ArrowRight: here + 1, ArrowDown: here + 1, ArrowLeft: here - 1, ArrowUp: here - 1, Home: 0, End: marks.length - 1 }[k];
      if (j == null) return;
      e.preventDefault();
      const to = order[Math.max(0, Math.min(marks.length - 1, j))];
      go(to);
      marks[to].focus();
    });
  });
  const nearest = (e: MouseEvent) => {
    const r = svg.getBoundingClientRect(), k = svg.viewBox.baseVal.width / r.width;
    const px = (e.clientX - r.left) * k, py = (e.clientY - r.top) * k;
    let best = -1, reach = 22;
    at().forEach((p, i) => { const d = Math.hypot(p.x - px, p.y - py); if (d < reach) { reach = d; best = i; } });
    return best;
  };
  svg.addEventListener("click", e => { const i = nearest(e); if (i >= 0) go(i); });
  svg.addEventListener("pointermove", e => { svg.style.cursor = nearest(e) >= 0 ? "pointer" : ""; });
}

/** Marks the selected option and gives it the Tab stop. */
export function selected(marks: SVGElement[], i: number) {
  marks.forEach((m, j) => {
    m.setAttribute("aria-selected", String(j === i));
    m.setAttribute("tabindex", j === i ? "0" : "-1");
  });
}

export type Verdict = { same: boolean | null; why: string };
export const MODELS = [["claude", "Claude Opus 5.5"], ["gpt", "GPT-5.5"]] as const;

/** What each model said about a match: a mark, the model and its reason. */
export function said(row: { claude: Verdict; gpt: Verdict }) {
  const out = el("div", "said");
  for (const [k, label] of MODELS) {
    const v = row[k], p = el("p", `verdict ${v.same == null ? "none" : v.same ? "yes" : "no"}`);
    p.append(el("span", "badge", `${v.same == null ? "–" : v.same ? "✓" : "✗"} ${label}`), v.same == null ? "No usable answer." : v.why);
    out.append(p);
  }
  return out;
}
