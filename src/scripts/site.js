const root = document.documentElement;
const themeButton = document.getElementById("theme");
const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
const showTheme = () => themeButton.textContent = isDark() ? "Light" : "Dark";
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
