// Development only: hugo server loads this; production builds leave it out.
{
const root = document.documentElement;

const FONTS = {
  Serif: ["Newsreader", "EB Garamond", "Source Serif 4", "Literata", "Lora", "Crimson Pro", "Cormorant Garamond",
    "Libre Baskerville", "Libre Caslon Text", "Spectral", "Fraunces", "Merriweather", "PT Serif", "Alegreya", "Cardo",
    "Gelasio", "Vollkorn", "Petrona", "Besley", "Brygada 1918", "Bitter", "Roboto Serif", "IBM Plex Serif", "Noto Serif",
    "Noto Serif Display", "Instrument Serif", "Young Serif", "Bodoni Moda", "Playfair Display", "Playfair", "Gloock",
    "DM Serif Display", "DM Serif Text", "Sorts Mill Goudy", "Old Standard TT", "Linden Hill", "Castoro",
    "Domine", "Lustria", "Cormorant", "Tinos"],
  Sans: ["Inter", "Inter Tight", "IBM Plex Sans", "Source Sans 3", "Work Sans", "DM Sans", "Manrope", "Plus Jakarta Sans",
    "Outfit", "Figtree", "Geist", "Public Sans", "Karla", "Nunito Sans", "Schibsted Grotesk", "Hanken Grotesk",
    "Instrument Sans", "Albert Sans", "Archivo", "Space Grotesk", "Sora", "Onest", "Bricolage Grotesque", "Libre Franklin",
    "Lato", "Mulish", "Rubik", "Red Hat Display", "Red Hat Text", "Familjen Grotesk", "Atkinson Hyperlegible Next",
    "Fira Sans", "Jost", "Josefin Sans", "Raleway"],
  Mono: ["IBM Plex Mono", "JetBrains Mono", "Fira Code", "Source Code Pro", "Space Mono", "DM Mono", "Geist Mono",
    "Roboto Mono", "Inconsolata", "Red Hat Mono", "Martian Mono", "Azeret Mono", "Courier Prime"],
  Display: ["Abril Fatface", "Cinzel", "Marcellus", "Italiana", "Prata", "Yeseva One", "Rozha One", "Syne", "Unbounded",
    "Bebas Neue", "Archivo Black", "Big Shoulders Display", "Della Respira", "Forum", "Bellefair", "Limelight", "Poiret One"],
};
const GENERIC = { Serif: "serif", Sans: "sans-serif", Mono: "monospace", Display: "serif" };
const ALL = Object.values(FONTS).flat();
const groupOf = f => Object.keys(FONTS).find(g => FONTS[g].includes(f));

// Google Fonts refuses a request that names a weight or style the family lacks, so each spec falls back to a smaller one.
const SPECS = [":ital,wght@0,300;0,400;0,500;0,600;0,700;1,400;1,500", ":wght@300;400;500;600;700", ":wght@400;700", ":ital@0;1", ""];
const loaded = new Set();
function load(family, i = 0) {
  if (i === 0 && loaded.has(family)) return;
  loaded.add(family);
  const link = document.createElement("link");
  link.rel = "stylesheet";
  link.href = `https://fonts.googleapis.com/css2?family=${family.replace(/ /g, "+")}${SPECS[i]}&display=swap`;
  link.onerror = () => { link.remove(); if (i + 1 < SPECS.length) load(family, i + 1); };
  document.head.append(link);
}

const ROLES = {
  name: "Name", heading: "Headings", body: "Body", label: "Labels",
};
const CASES = {
  uppercase: { case: "uppercase", caps: "normal", track: ".12em" },
  smallcaps: { case: "lowercase", caps: "small-caps", track: ".08em" },
  normal: { case: "none", caps: "normal", track: "0" },
};
const PRESETS = {
  chosen: { name: ["Merriweather", 1, 500], heading: ["Merriweather", 1.05, 500], body: ["Geist", .85, 400], label: ["Onest", 1.22, 500], case: "uppercase" },
  newsreader: { name: ["Newsreader", 1, 500], heading: ["Newsreader", 1, 500], body: ["Newsreader", 1, 400], label: ["Inter", 1, 500], case: "uppercase" },
  garamond: { name: ["EB Garamond", 1, 500], heading: ["EB Garamond", 1, 500], body: ["EB Garamond", 1.07, 400], label: ["EB Garamond", 1.45, 500], case: "smallcaps" },
  instrument: { name: ["Instrument Serif", 1.2, 400], heading: ["Instrument Serif", 1.2, 400], body: ["Instrument Sans", .86, 400], label: ["Geist Mono", 1, 500], case: "uppercase" },
  geist: { name: ["Newsreader", 1, 500], heading: ["Newsreader", 1.05, 500], body: ["Geist", .85, 400], label: ["Geist Mono", 1, 500], case: "uppercase" },
  source: { name: ["Source Serif 4", 1, 600], heading: ["Source Serif 4", 1, 500], body: ["Source Serif 4", .93, 400], label: ["IBM Plex Mono", 1.08, 500], case: "normal" },
};

let state = structuredClone(PRESETS.chosen);
try { const saved = JSON.parse(localStorage.getItem("type")); if (saved && saved.body) state = saved; } catch {}

const options = Object.entries(FONTS).map(([g, fs]) =>
  `<optgroup label="${g}">${fs.map(f => `<option>${f}</option>`).join("")}</optgroup>`).join("");
document.getElementById("roles").innerHTML = Object.entries(ROLES).map(([k, title]) => `
  <fieldset data-role="${k}">
    <legend>${title}</legend>
    <div class="step"><button type="button" data-step="-1" aria-label="Previous ${title} font">◀</button><select aria-label="${title} font">${options}</select><button type="button" data-step="1" aria-label="Next ${title} font">▶</button></div>
    <label>Size <input type="range" min="0.6" max="1.8" step="0.01" data-i="1"><output></output></label>
    <label>Weight <input type="range" min="200" max="900" step="100" data-i="2"><output></output></label>
    ${k === "label" ? `<label>Case <select id="case"><option value="uppercase">Uppercase</option><option value="smallcaps">Small caps</option><option value="normal">As written</option></select><span></span></label>` : ""}
  </fieldset>`).join("") + `<div style="height:.1rem"></div>`;

function apply() {
  for (const k of Object.keys(ROLES)) {
    const [font, scale, weight] = state[k];
    load(font);
    root.style.setProperty(`--${k}-font`, `"${font}", ${GENERIC[groupOf(font)] || "serif"}`);
    root.style.setProperty(`--${k}-scale`, scale);
    root.style.setProperty(`--${k}-weight`, weight);
    const fs = document.querySelector(`fieldset[data-role="${k}"]`);
    fs.querySelector("select").value = font;
    fs.querySelectorAll("input").forEach(inp => {
      inp.value = state[k][inp.dataset.i];
      inp.nextElementSibling.textContent = inp.dataset.i === "1" ? Math.round(state[k][1] * 100) + "%" : state[k][2];
    });
  }
  const c = CASES[state.case];
  root.style.setProperty("--label-case", c.case);
  root.style.setProperty("--label-caps", c.caps);
  root.style.setProperty("--label-track", c.track);
  document.getElementById("case").value = state.case;
  document.getElementById("settings").textContent = JSON.stringify(state);
  try { localStorage.setItem("type", JSON.stringify(state)); } catch {}
}

document.querySelectorAll("fieldset[data-role]").forEach(fs => {
  const k = fs.dataset.role;
  fs.querySelector("select").addEventListener("change", e => { state[k][0] = e.target.value; apply(); });
  fs.querySelectorAll("[data-step]").forEach(btn => btn.addEventListener("click", () => {
    const i = ALL.indexOf(state[k][0]);
    state[k][0] = ALL[(i + +btn.dataset.step + ALL.length) % ALL.length]; apply();
  }));
  fs.querySelectorAll("input").forEach(inp => inp.addEventListener("input", () => { state[k][inp.dataset.i] = +inp.value; apply(); }));
});
document.getElementById("case").addEventListener("change", e => { state.case = e.target.value; apply(); });
document.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", () => {
  state = structuredClone(PRESETS[b.dataset.preset]); apply();
}));
document.getElementById("copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(state)); document.getElementById("copied").textContent = "Copied"; }
  catch { document.getElementById("copied").textContent = "Select the text below"; }
});
apply();
}
