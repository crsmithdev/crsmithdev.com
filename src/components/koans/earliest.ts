import { el, title } from "./svg";

// What the three "earliest version" figures share: the data, the color of each
// source, and the hover text of one case.
export type Text = [string, string, string | null];
export type Case = { n: number; en: string; e: { i: number; y: number } | null };
export type Data = { texts: Text[]; bc: { year: number; cases: Case[] }; gg: { year: number; cases: Case[] } };
export type Key = "bc" | "gg";
export type Item = { k: Key; c: Case; y: number | null };

// The four big early sources keep the color Firsts gives them; every other source is grey.
export const GROUPS = [
  { label: "*Zutang ji*", color: "var(--kw-blue)", ids: ["B25n0144"] },
  { label: "*Jingde Record*", color: "var(--kw-violet)", ids: ["T51n2076"] },
  { label: "*Extended Record of Yunmen*", color: "var(--kw-sky)", ids: ["T47n1988"] },
  { label: "*Record of Baiyun Shouduan*", color: "var(--kw-green)", ids: ["X69n1351", "X69n1352"] },
  { label: "Other texts", color: "var(--kw-grey)", ids: [] as string[] },
];

const SHORT: Record<Key, string> = { bc: "Blue Cliff Record", gg: "Gateless Gate" };

export function items(data: Data): Item[] {
  return (["bc", "gg"] as Key[]).flatMap(k => data[k].cases.map(c => ({ k, c, y: c.e?.y ?? null })));
}

export function textName(data: Data, i: number) {
  return data.texts[i][2] ?? data.texts[i][1];
}

/** The color of the source a case is first told in. */
export function color(data: Data, i: number) {
  const id = data.texts[i][0];
  return (GROUPS.find(g => g.ids.includes(id)) ?? GROUPS[4]).color;
}

export function tip(data: Data, x: Item) {
  return `*${SHORT[x.k]}* ${x.c.n} · ${x.c.en}` + (x.c.e ? ` · ${title(data.texts[x.c.e.i])}, ${x.c.e.y}` : " · no earlier version");
}

/** The legend under a figure: the source colors, then any extra entries. */
export function legend(ul: Element, extra: [string, string][] = []) {
  for (const [label, style] of [...GROUPS.map(g => [g.label, `background: ${g.color}`]), ...extra]) {
    const li = document.createElement("li"), i = document.createElement("i");
    i.setAttribute("style", style);
    li.append(i, el("span", null, label));
    ul.append(li);
  }
}

/** Calls draw now and again whenever the element changes width. */
export function onWidth(e: Element, draw: (w: number) => void) {
  let width = 0;
  new ResizeObserver(() => { if (e.clientWidth !== width) { width = e.clientWidth; draw(width); } }).observe(e);
}
