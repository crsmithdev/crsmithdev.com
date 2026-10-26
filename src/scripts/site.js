const root = document.documentElement;
const themeButton = document.getElementById("theme");
const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
const showTheme = () => {
  const dark = isDark();
  themeButton.querySelector(".sun").toggleAttribute("hidden", !dark);
  themeButton.querySelector(".moon").toggleAttribute("hidden", dark);
  themeButton.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
};
showTheme();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", showTheme);
themeButton.addEventListener("click", () => {
  root.dataset.theme = isDark() ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
  showTheme();
});
const links = [...document.querySelectorAll("nav.sections a")];
const observer = new IntersectionObserver(es => es.forEach(e => {
  if (e.isIntersecting) links.forEach(l => l.classList.toggle("active", l.hash === "#" + e.target.id));
}), { rootMargin: "-30% 0px -60% 0px" });
document.querySelectorAll("main section").forEach(s => observer.observe(s));
addEventListener("scroll", () => {
  if (innerHeight + scrollY < document.documentElement.scrollHeight - 2) return;
  links.forEach((l, i) => l.classList.toggle("active", i === links.length - 1));
});

// Footnotes: hovering or focusing a reference shows its note beside it.
const tip = Object.assign(document.createElement("div"), { className: "fn-tip", hidden: true });
tip.setAttribute("role", "tooltip");
document.body.append(tip);
let hiding;
const hide = () => { hiding = setTimeout(() => { tip.hidden = true; }, 150); };
tip.addEventListener("mouseenter", () => clearTimeout(hiding));
tip.addEventListener("mouseleave", hide);
for (const ref of document.querySelectorAll(".post a[data-footnote-ref]")) {
  const note = document.getElementById(decodeURIComponent(ref.hash.slice(1)));
  if (!note) continue;
  const show = () => {
    clearTimeout(hiding);
    const copy = note.cloneNode(true);
    copy.querySelectorAll("[data-footnote-backref]").forEach(b => b.remove());
    tip.replaceChildren(...copy.childNodes);
    tip.hidden = false;
    const r = ref.getBoundingClientRect();
    tip.style.left = `${scrollX + Math.max(8, Math.min(r.left - 16, innerWidth - tip.offsetWidth - 8))}px`;
    tip.style.top = `${scrollY + r.bottom + 6}px`;
  };
  ref.addEventListener("mouseenter", show);
  ref.addEventListener("focus", show);
  ref.addEventListener("mouseleave", hide);
  ref.addEventListener("blur", hide);
}
