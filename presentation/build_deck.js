// Builds Final_Review_Updated.pptx: the S7 final-review deck in its original
// plain 4:3 style, updated with the current Phase 1 results.
// Usage: node build_deck.js <figures_dir> <out.pptx>
const path = require("path");
const pptxgen = require("pptxgenjs");
const { imageSize } = require("./imgsize");

const FIG = process.argv[2];
const OUT = process.argv[3];
const THEME = {
  name: "Final Review",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "000000", lt1: "FFFFFF", dk2: "1F2937", lt2: "F2F2F2",
    accent1: "2B6CB0", accent2: "C05621", accent3: "2F855A",
    accent4: "6B46C1", accent5: "C53030", accent6: "1C8C8C",
    hlink: "2B6CB0", folHlink: "6B46C1",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_4x3"; // 10 x 7.5 in, as the original deck
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Final Review - B.Tech Project (updated)";
const C = pres.SchemeColor;
const W = 10, H = 7.5, MX = 0.5;

pres.defineSlideMaster({
  title: "COVER",
  background: { color: "FFFFFF" },
  objects: [],
});
pres.defineSlideMaster({
  title: "TITLE_ONLY",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.3, w: W - 2 * MX, h: 0.75,
        fontSize: 28, bold: true, color: C.text1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 9.2, y: 7.0, w: 0.5, h: 0.3, fontSize: 12, color: C.text1, align: "right" },
});

let n = 0;
function slide(title, section) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY", sectionTitle: section });
  s.addText(title, { placeholder: "title" });
  n += 1;
  return s;
}
function fit(file, x, y, w, h, opts = {}) {
  const { width, height } = imageSize(file);
  const r = Math.min(w / width, h / height);
  const iw = width * r, ih = height * r;
  const ix = opts.align === "left" ? x : x + (w - iw) / 2;
  const iy = opts.valign === "top" ? y : y + (h - ih) / 2;
  return { path: file, x: ix, y: iy, w: iw, h: ih, altText: opts.alt || path.basename(file), objectName: opts.name || "figure" };
}
const fig = (f) => path.join(FIG, f);
const body = (t, o = {}) => Object.assign({ isTextBox: true, fontSize: 14, color: C.text1, valign: "top", margin: 0.05, paraSpaceAfter: 4 }, o);
function bullets(items, o = {}) {
  return items.map((t, i) => {
    const runs = Array.isArray(t) ? t : [{ text: t }];
    return runs.map((r, j) => ({ text: r.text, options: Object.assign({}, r.options || {},
      j === 0 ? { bullet: true } : {}, j === runs.length - 1 && i < items.length - 1 ? { breakLine: true } : {}) }));
  }).flat();
}
const B = (t) => ({ text: t, options: { bold: true } });
const T = (t) => ({ text: t });
function caption(s, text, x, y, w, h = 0.5) {
  s.addText(text, body(null, { x, y, w, h, fontSize: 11, italic: true, color: C.text2, align: "center", objectName: "caption" }));
}
function table(s, rows, x, y, w, colW, o = {}) {
  const hdr = rows[0].map((c) => ({ text: c, options: { bold: true, fill: { color: C.background2 } } }));
  const data = rows.slice(1).map((r) => r.map((c) => (typeof c === "string" ? { text: c } : c)));
  s.addTable([hdr, ...data], Object.assign({ x, y, w, colW, fontSize: 12, fontFace: "Calibri", color: C.text1,
    border: { type: "solid", pt: 0.75, color: "BFBFBF" }, valign: "middle", margin: 0.05, objectName: "table" }, o));
}
function stat(s, x, y, w, big, label, color) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.45, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2 }, objectName: "stat card" });
  s.addText(big, body(null, { x, y: y + 0.12, w, h: 0.75, fontSize: 32, bold: true, color, align: "center", valign: "middle" }));
  s.addText(label, body(null, { x: x + 0.1, y: y + 0.85, w: w - 0.2, h: 0.55, fontSize: 12, color: C.text1, align: "center" }));
}

// 1 -------------------------------------------------------------- Cover
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" }); n += 1;
  s.addText("Final Review – B.TECH PROJECT", body(null, { x: 0.5, y: 0.6, w: 9, h: 0.5, fontSize: 20, align: "center" }));
  s.addText("Graph-MoE: A Topology-Aware Mixture-of-Experts Framework for Drift Adaptive Weather Forecasting",
    body(null, { x: 0.8, y: 1.35, w: 8.4, h: 1.3, fontSize: 28, bold: true, align: "center", valign: "middle" }));
  s.addText([{ text: "Student Details:", options: { bold: true, breakLine: true } },
    { text: "1. Pranesh L - CB.AI.U4AID23133", options: { breakLine: true } },
    { text: "2. Kathirvelan M - CB.AI.U4AID23118", options: { breakLine: true } },
    { text: "3. Sasikumar P - CB.AI.U4AID23146", options: { breakLine: true } },
    { text: "4. Jeiesh J S - CB.AI.U4AID23116" }],
    body(null, { x: 2.5, y: 3.15, w: 5, h: 1.9, fontSize: 18, align: "center" }));
  s.addText([{ text: "Team Number: AB 13", options: { breakLine: true } }, { text: "Supervisor: Dr. Vinith" }],
    body(null, { x: 2.5, y: 5.35, w: 5, h: 0.8, fontSize: 18, align: "center" }));
  s.addNotes("Updated final-review deck. All results are taken from the project's current result files (after the September audit fix).");
}

// 2 -------------------------------------------------------------- Problem
{
  const s = slide("Problem statement and motivation", "Introduction");
  s.addText(bullets([
    "Weather and climate regimes shift over time (El Niño vs. normal years): a model trained once quietly goes stale as conditions change.",
    "Retraining the entire model, from scratch or on a fixed schedule, is wasteful: a shift is usually confined to particular regions or relationships.",
    "Most forecasters choose which regions to trust by correlation or geography, so they cannot tell \"this region's physics changed\" from \"a coincidental correlation broke\".",
    "As systems grow from regional to global scale, the number of regions that can drift grows too, and full retraining becomes even more costly.",
    [B("Our approach: "), T("validate real causal drivers first, then adapt only the regions that actually drifted.")],
  ]), body(null, { x: MX, y: 1.3, w: 9, h: 5.4, fontSize: 18, paraSpaceAfter: 14 }));
}

// 3-4 ------------------------------------------------------------ Objectives
function objectives(title, items) {
  const s = slide(title, "Introduction");
  const y0 = 1.35, step = 5.4 / items.length;
  items.forEach((t, i) => {
    const y = y0 + i * step;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "number" });
    s.addText(String(i + 1), body(null, { x: MX, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0 }));
    s.addText([B(t[0] + ": "), T(t[1])].map((r) => ({ text: r.text, options: r.options || {} })),
      body(null, { x: 1.2, y, w: 8.3, h: step - 0.1, fontSize: 14 }));
  });
}
objectives("Phase 1 objectives", [
  ["Causal graph", "discover which regions' past conditions carry information about another region's future OLR (PCMCI over 50 regions x 6 variables)."],
  ["Stress tests", "check every link against fake decoy variables, physical travel speeds, trimming to the strongest links, and redrawing the regions at double resolution."],
  ["Graph as guidance", "feed the validated graph to the causal splitter three ways: narrow its candidates, nudge its scores, or penalise selections outside the graph."],
  ["Unseen years", "compare the graph-guided splitter with an unguided one on held-out 2013–2022."],
  ["Drift-adaptive experts", "detect drift per region, then retrain, spawn, or reactivate only that region's expert."],
]);
objectives("Phase 2 objectives", [
  ["India forecasting backbone", "shared graph neural network on hourly surface data, with region-specific experts conditioned on Phase 1 validated causal drivers."],
  ["72-hour forecasts", "autoregressive roll-out to 72 h, with a station-level component checked against real observations."],
  ["Benchmark", "compare with GraphCast and WeatherNext on the same data, horizons and evaluation protocol."],
  ["Global scale", "extend the validated regional architecture to larger meshes, more regimes and a larger expert pool."],
]);

// 5 -------------------------------------------------------------- Literature
pres.addSection({ title: "Literature and data" });
{
  const s = slide("03 | Consolidated Literature Review & Research Gap", "Literature and data");
  table(s, [
    ["Paper", "Key idea / method", "Limitation"],
    ["Wu et al., \"Discovering invariant rationales for graph neural networks\", ICLR 2022",
      "Splits a graph into causal rationale and non-causal part; swaps the non-causal part across instances; stable prediction = invariant causal information.",
      "Built for graph classification; no spatiotemporal forecasting; no dynamic expert adaptation."],
    ["Ghaffari et al., \"Graph-conditioned mixture of GNN experts for traffic forecasting\", arXiv 2026",
      "A small router blends frozen, pre-trained experts per node using topology and recent input.",
      "Experts frozen forever: no retraining, no spawning, no answer to drift."],
    ["Cao et al., \"Geometric mixture-of-experts with curvature-guided adaptive routing\", arXiv 2026",
      "Routes each node to expert sub-networks by Ollivier–Ricci curvature across Riemannian spaces.",
      "Routes by topology/curvature alone: no notion of drift or causality."],
    ["Wu et al., \"GraphMETRO: mitigating complex graph distribution shifts via mixture of aligned experts\", NeurIPS 2024",
      "Each expert is invariant to one hand-constructed shift type; blended by a learned gate.",
      "Shift components hand-constructed and static; no shift structure learned from data."],
  ], MX, 1.2, 9, [2.9, 3.4, 2.7], { fontSize: 10.5 });
  s.addText("RESEARCH GAP", body(null, { x: MX, y: 4.55, w: 9, h: 0.35, fontSize: 16, bold: true, align: "center" }));
  s.addText("Existing studies address causal rationale discovery, graph distribution shifts, dynamic expert learning and graph-based weather forecasting separately. None combines causal structure discovery with drift-triggered expert adaptation for spatiotemporal weather forecasting.",
    body(null, { x: MX, y: 4.95, w: 9, h: 1.2, fontSize: 16, align: "center" }));
}

// 6-7 ------------------------------------------------------------ Dataset
{
  const s = slide("06 | Dataset", "Literature and data");
  s.addText(bullets([
    [B("Source: "), T("BSISO monitoring data; the Boreal Summer Intraseasonal Oscillation is a slow-moving pattern of tropical convection related to the MJO.")],
    [B("Grid: "), T("2.5° tropical belt, 30°S–30°N, all longitudes (25 x 144 = 3,600 cells, about 275 km each).")],
    [B("Time: "), T("daily, 1979-01-01 to 2022-12-31; 16,071 days, none missing.")],
    [B("Fields: "), T("SST, h850, u200, PW, u850, OLR, as anomalies normalised per grid cell.")],
    [B("Target: "), T("OLR, a proxy for convection (low OLR = active rain) and the least persistent field.")],
    [B("Regions: "), T("3,600 cells -> 50 regions by k-means on OLR -> 50 x 6 = 300 nodes; the adjacency mesh defines candidate pools.")],
    [B("Split: "), T("1979–2012 for graph discovery and training; 2013–2022 held out for testing only.")],
    [B("Season: "), T("Phase 1 scores BSISO-active target days only (208 days/yr on average); source history is kept in full.")],
  ]), body(null, { x: MX, y: 1.25, w: 9, h: 5.6, fontSize: 16, paraSpaceAfter: 9 }));
}
{
  const s = slide("06 | Dataset: active BSISO season", "Literature and data");
  s.addImage(fit(fig("ppt_01b_active_bsiso_season_per_year.png"), MX, 1.2, 6.0, 4.6, { alt: "Active BSISO season per year, 1979-2022" }));
  s.addText(bullets([
    "BSISO index rebuilt from our own OLR + u850 (combined EOF, May–Oct, 1979–2012 only).",
    [B("EOF variance 11.2% / 7.7%"), T(" vs. about 10% / 7% published by Lee et al. (2013).")],
    "Season = 31-day mean amplitude > 1 for 20+ days, padded by 21 days.",
    [B("208 days/yr"), T(" on average (82–286); onset April–July.")],
  ]), body(null, { x: 6.7, y: 1.4, w: 2.8, h: 4.6, fontSize: 13 }));
  caption(s, "Active season per year: blue = discovery/training years, orange = held-out test years.", MX, 6.05, 6.0);
  s.addNotes("A fixed May-October window would include inactive weeks in some years and miss active weeks in others, so the season is defined by the oscillation's own amplitude. Only target days are restricted; a lag-45 source for a 1 June target reads mid-April.");
}

// 8-12 ------------------------------------------------------------ Method
pres.addSection({ title: "Method" });
{
  const s = slide("System architecture", "Method");
  s.addImage(fit(fig("workflows/png/wf_phase1.png"), MX, 1.15, 9, 5.75, { alt: "Phase 1 workflow: build, validate, freeze and use the causal graph; detect drift and adapt experts" }));
  s.addNotes("Steps 1-8 build, validate, freeze and use the causal graph; steps 9-11 detect drift and adapt the regional experts. Bold numbers are measured results.");
}
{
  const s = slide("How PCMCI works?", "Method");
  const steps = [
    ["Build candidates", "For each OLR target, every node (region x field) at every lag 1–60 days: 18,000 candidates per target."],
    ["PC1 screening", "Test each candidate against the target while controlling for the strongest candidates found so far; weak candidates drop out early."],
    ["MCI final test", "Retest each survivor while controlling for the target's other parents AND the source's own past; a link survives only if it still adds information."],
    ["FDR correction", "Benjamini–Hochberg across all 18,000 tests per target keeps only reliable links (q ≤ 0.05)."],
  ];
  steps.forEach((t, i) => {
    const y = 1.3 + i * 1.2;
    s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.5, h: 0.5, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: "step" });
    s.addText(String(i + 1), body(null, { x: MX, y: y + 0.05, w: 0.5, h: 0.5, fontSize: 16, bold: true, color: C.background1, align: "center", valign: "middle", margin: 0 }));
    s.addText([{ text: t[0] + ": ", options: { bold: true } }, { text: t[1] }], body(null, { x: 1.2, y, w: 8.3, h: 1.1, fontSize: 14 }));
  });
  s.addText("Lag 0 is excluded, so direction comes from time order. Our vectorised implementation matches the Tigramite reference (identical PC1 parent sets, MCI values within 0.002) and runs 50 targets x 18,000 candidates in about 23 minutes.",
    body(null, { x: MX, y: 6.15, w: 9, h: 0.8, fontSize: 13, italic: true, color: C.text2 }));
}
{
  const s = slide("Graph validation: four independent checks", "Method");
  s.addImage(fit(fig("workflows/png/wf_validation.png"), MX, 1.15, 9, 3.4, { alt: "Validation checks B1-B4 and the three ways to use the graph" }));
  s.addText(bullets([
    [B("B1 Decoys: "), T("phase-randomised copies keep each series' memory but no real link; does PCMCI pick them?")],
    [B("B2 Lag mask: "), T("implied speed = distance / lag must be physically possible for the source field.")],
    [B("B3 Top-k: "), T("keep the k strongest outside links per target, k chosen on 2005–2012.")],
    [B("B4 Resolution: "), T("do links reappear when every region is split in two?")],
  ]), body(null, { x: MX, y: 4.75, w: 9, h: 2.2, fontSize: 16, paraSpaceAfter: 6 }));
}
{
  const s = slide("How the drift detectors and expert lifecycle work", "Method");
  s.addImage(fit(fig("workflows/png/wf_lifecycle.png"), MX, 1.15, 9, 4.4, { alt: "Dual-channel drift detection and expert lifecycle" }));
  s.addText(bullets([
    [B("Channel 1, ADWIN: "), T("raises an alarm when recent forecast errors differ significantly from older ones.")],
    [B("Channel 2, STARS: "), T("tests the causal-signature distance for a confirmed regime shift; it can react before errors rise.")],
    [B("Lifecycle: "), T("retrain, or archive and reactivate a matching expert (cosine similarity ≥ 0.9), else spawn a new one.")],
  ]), body(null, { x: MX, y: 5.65, w: 9, h: 1.3, fontSize: 13 }));
}
{
  const s = slide("Model, configuration and parameters", "Method");
  table(s, [
    ["Hyperparameter (splitter + expert)", "Value"],
    ["Epochs / training samples", "5 / 4,000 contiguous"],
    ["Optimiser, learning rate, batch", "Adam, 1e-3, 16"],
    ["Hidden width / generator window", "16 / 10 days"],
    ["Interventions; variance weight", "6 swaps; 0 -> 1 after 30% warm-up"],
  ], MX, 1.25, 4.3, [2.4, 1.9], { fontSize: 12 });
  table(s, [
    ["Causal graph (PCMCI)", "Value"],
    ["Nodes", "300 (50 regions x 6 fields)"],
    ["Candidates per target", "18,000"],
    ["Lags tested", "1–60 days"],
    ["PC alpha / FDR alpha", "0.05 / 0.05"],
    ["Discovery days", "7,153 season days, 1979–2012"],
  ], 5.2, 1.25, 4.3, [1.9, 2.4], { fontSize: 12 });
  s.addText(bullets([
    [B("50 independent splitter + expert pairs: "), T("one per region, trained from scratch, no weight sharing.")],
    [B("Splitter (DIR-style): "), T("rationale generator scores each candidate source from 10-day windows; top fraction = causal set; non-causal sources swapped with other days; loss = mean + variance of risk across swaps.")],
    [B("Expert: "), T("weighted mean of the selected sources' features (weights from the splitter) -> MLP -> OLR forecast.")],
  ]), body(null, { x: MX, y: 4.45, w: 9, h: 2.5, fontSize: 14 }));
}

// 13-? ------------------------------------------------------------ Results
pres.addSection({ title: "Phase 1 results" });
{
  const s = slide("Phase 1 results: the discovered causal graph", "Phase 1 results");
  stat(s, MX, 1.2, 2.1, "536", "significant links (FDR 0.05)", C.accent1);
  stat(s, 2.8, 1.2, 2.1, "160", "links from other regions", C.accent2);
  stat(s, 5.1, 1.2, 2.1, "1 day", "median lag of outside links", C.accent3);
  stat(s, 7.4, 1.2, 2.1, "2.06", "source regions per target", C.accent4);
  s.addImage(fit(fig("ppt_map_standard_180_convention.png"), MX, 2.85, 9, 2.1, { alt: "Map of validated outside drivers coloured by source field" }));
  s.addText(bullets([
    "Outside drivers by field: OLR 72, u850 37, u200 22, h850 18, PW 11, SST 0; mostly neighbouring regions.",
    "A few long links (wind and height) cross thousands of km in 1–2 days, at equatorial-wave speeds.",
    [B("Region 22: "), T("the graph picks regions 27 and 6; the unguided splitter independently picks 6, 12 and 27. Two different methods agree.")],
  ]), body(null, { x: MX, y: 5.1, w: 9, h: 1.85, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: B1 decoy test", "Phase 1 results");
  s.addImage(fit(fig("ppt_decoy_v1_big_count_bars.png"), MX, 1.2, 4.2, 4.4, { alt: "0 of 250 decoys selected" }));
  s.addImage(fit(fig("ppt_03b_links_through_checks_no_b3_b4.png"), 4.9, 1.2, 4.6, 4.4, { alt: "Links after PCMCI, B1 and B2" }));
  caption(s, "0 of 250 fake variables fooled the causal discovery", MX, 5.6, 4.2, 0.35);
  caption(s, "Own-region (blue) and outside (orange) links after each check", 4.9, 5.6, 4.6, 0.35);
  s.addText("Decoys share the real series' memory and 30–60-day rhythm but have no real link. None was selected (gate: below 30%), and the real graph barely changed (517 vs. 536 links), so PCMCI is not fooled by autocorrelation.",
    body(null, { x: MX, y: 6.05, w: 9, h: 0.9, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: B2 physical lag mask", "Phase 1 results");
  s.addImage(fit(fig("ppt_05b_b2_speed_no_title.png"), MX, 1.15, 9, 4.6, { alt: "Implied speed vs distance by field" }));
  s.addText(bullets([
    [B("Only 3 of 536 links "), T("need an impossible speed (64, 33 and 26 m/s at lag 1); they are dropped from the prior.")],
    "Limits: SST 1 m/s; OLR and PW 20 m/s (coupled Kelvin waves); winds and height 50 m/s.",
    "28 kept outside links span 500–5,500 km at up to 46 m/s, all carried by wind and height fields.",
  ]), body(null, { x: MX, y: 5.85, w: 9, h: 1.1, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: B3 top-k and B4 resolution", "Phase 1 results");
  s.addImage(fit(fig("07_b4_persistence.png"), MX, 1.15, 9, 3.2, { alt: "Persistence per target at 100 regions and by field" }));
  stat(s, MX, 4.5, 2.8, "0.62", "mean persistence at 100 regions", C.accent1);
  stat(s, 3.6, 4.5, 2.8, "33 / 48", "targets with persistence ≥ 0.5", C.accent3);
  stat(s, 6.7, 4.5, 2.8, "k = 10", "B3 keeps all 157 outside links", C.accent2);
  s.addText("OLR links persist best (56/70); moisture and low-level height links, being more local, least. PCMCI + FDR is already sparse, so B3 keeps every outside link.",
    body(null, { x: MX, y: 6.1, w: 9, h: 0.85, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: causal graph as the splitter's guide", "Phase 1 results");
  table(s, [
    ["Lead", "Prior", "Informative splitters", "Beat persistence", "Mean skill vs pers.", "Mean R² vs clim."],
    ["7 days", "none", "0/50", "50/50", "+0.415", "-0.015"],
    ["7 days", "PCMCI", { text: "24/50", options: { bold: true } }, "50/50", "+0.416", "-0.015"],
    ["1 day", "none", "0/50", "0/50", "-1.527", "+0.720"],
    ["1 day", "PCMCI", { text: "36/50", options: { bold: true } }, "3/50", { text: "-1.327", options: { bold: true } }, { text: "+0.745", options: { bold: true } }],
  ], MX, 1.25, 9, [0.9, 0.9, 1.9, 1.7, 1.8, 1.8], { fontSize: 13 });
  s.addText(bullets([
    [B("Full 50-region pool: "), T("without guidance the splitter's scores are uninformative everywhere; with the graph (restrict mode) 24 regions at 7 days and 36 at 1 day become informative.")],
    [B("1-day skill improves in 32 of 50 regions "), T("(median +0.40 skill vs persistence, about 17% lower MSE; Wilcoxon p = 0.049).")],
    [B("7-day skill unchanged: "), T("the validated links act within 1–2 days, so they help short leads; week-ahead guidance needs a graph built with lags of 7+ days.")],
    "Held-out 2013–2022 season days (1,989); skill = 1 − MSE / MSE(reference).",
  ]), body(null, { x: MX, y: 3.65, w: 9, h: 3.3, fontSize: 14 }));
}
{
  const s = slide("Phase 1 results: 1-day gain, region by region", "Phase 1 results");
  s.addImage(fit(fig("10_stage_c_lead1.png"), MX, 1.15, 9, 4.3, { alt: "Per-region 1-day skill with vs without the PCMCI prior" }));
  s.addText(bullets([
    "Each dot is a region; dots above the diagonal improve with the causal prior.",
    [B("Region 22: "), T("skill vs persistence -0.60 -> -0.05.")],
    "Regions with 2+ causal drivers improve on average; regions with a single driver are the hardest case (the splitter must choose between own history and one driver).",
  ]), body(null, { x: MX, y: 5.55, w: 9, h: 1.4, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: forecast accuracy", "Phase 1 results");
  s.addImage(fit(fig("ppt_c3_per_region_mse_rmse_7day.png"), MX, 1.15, 4.6, 3.0, { alt: "Per-region MSE and RMSE of the 7-day forecast" }));
  s.addImage(fit(fig("ppt_09_2_forecast_vs_actual_no_title.png"), 5.2, 1.15, 4.3, 3.0, { alt: "Region 22 one-day forecast vs observed OLR" }));
  caption(s, "7-day forecast error per region (mean MSE 0.317, RMSE 0.559 std)", MX, 4.15, 4.6, 0.5);
  caption(s, "Region 22: 1-day forecast vs observed OLR (illustrative 1979–80 window)", 5.2, 4.15, 4.3, 0.5);
  s.addText(bullets([
    [B("Beats persistence in all 50 regions at 7 days "), T("in every candidate pool (mean skill about +0.40, i.e. about 40% lower error than \"no change\").")],
    "The expert tracks OLR swings one day ahead, including sharp convective transitions.",
    "At 7 days OLR keeps little memory, so climatology becomes the harder reference (mean R² vs climatology about -0.015).",
  ]), body(null, { x: MX, y: 4.8, w: 9, h: 2.15, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: what makes source selection work", "Phase 1 results");
  s.addImage(fit(fig("ppt_03c_informative_scores_2_vs_5_epochs.png"), MX, 1.15, 3.6, 3.7, { alt: "Informative splitters after 2 vs 5 epochs" }));
  s.addImage(fit(fig("02_variant_comparison.png"), 4.3, 1.15, 5.2, 3.7, { alt: "Skill flat across pools, discrimination collapses as the pool grows" }));
  caption(s, "Training budget: 5/50 -> 27/50 informative", MX, 4.85, 3.6, 0.35);
  caption(s, "Pool size: accuracy flat, discrimination falls", 4.3, 4.85, 5.2, 0.35);
  s.addText(bullets([
    [B("Training: "), T("5 epochs instead of 2 raises informative splitters from 5 to 27 of 50 at equal accuracy.")],
    [B("Pool size: "), T("informative splitters fall from 27 (neighbours) to 8 (2-hop) to 0 (all 50 regions). This is exactly the gap the validated graph closes (0 -> 24 / 36).")],
  ]), body(null, { x: MX, y: 5.35, w: 9, h: 1.6, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: drift detection", "Phase 1 results");
  const bold = (t) => ({ text: t, options: { bold: true } });
  table(s, [
    ["Region", "Shift", "Ch.1 error (ADWIN)", "Ch.2 causal (STARS)", "Fused OR", "Fused AND"],
    ["22", "Abrupt El Niño 97–98", "159 d", bold("35 d"), "35 d", "missed"],
    ["22", "Gradual 2013–22", "966 d", bold("181 d"), "181 d", "966 d"],
    ["29", "Abrupt El Niño 97–98", "223 d", bold("92 d"), "92 d", "235 d"],
    ["29", "Gradual 2013–22", "1,158 d", bold("117 d"), "117 d", "1,195 d"],
    ["12", "Abrupt El Niño 97–98", bold("95 d"), "123 d", "95 d", "123 d"],
    ["12", "Gradual 2013–22", bold("70 d"), "100 d", "70 d", "100 d"],
    ["False alarms / yr", "22 / 29 / 12", "0.17 / 0.34 / 1.04", "1.62 / 2.25 / 2.19", "1.78 / 2.59 / 3.23", "0.17 / 0.34 / 0.94"],
  ], MX, 1.2, 9, [0.95, 1.75, 1.6, 1.6, 1.55, 1.55], { fontSize: 12 });
  s.addText(bullets([
    [B("The causal channel sees shifts first: "), T("at regions 22 and 29 it detects the El Niño 2.4–4.5x sooner and the gradual drift 5.3–9.9x sooner than the error channel.")],
    "Fused OR catches every shift at every region; AND is slower but raises far fewer false alarms: a real, measured trade-off.",
    "Latency = days from the start of the shift window to the first alarm in it.",
  ]), body(null, { x: MX, y: 4.6, w: 9, h: 2.35, fontSize: 15, paraSpaceAfter: 8 }));
}
{
  const s = slide("Phase 1 results: expert lifecycle", "Phase 1 results");
  s.addImage(fit(fig("11_lifecycle_timeline.png"), MX, 1.15, 9, 3.4, { alt: "Lifecycle events 1979-2022 for regions 22, 12, 29" }));
  table(s, [
    ["Reactivation strategy", "Region 22", "Region 12", "Region 29"],
    ["Top-3 blend beats single best", "30/49", "42/61", "41/61"],
    ["Mean MSE: single -> blend", "0.368 -> 0.352", "0.422 -> 0.402", "0.384 -> 0.376"],
    ["Forgetting: single -> blend", "-0.010 -> -0.013", "0.183 -> 0.170", "0.092 -> 0.086"],
  ], MX, 4.65, 9, [3.0, 2.0, 2.0, 2.0], { fontSize: 12 });
  s.addText([{ text: "104 of 171 regime changes ", options: { bold: true } }, { text: "reactivated an archived expert instead of training from scratch: regimes recur and the archive pays off. Blending the 3 closest archived experts lowers both error and forgetting at every region." }],
    body(null, { x: MX, y: 6.2, w: 9, h: 0.8, fontSize: 13 }));
}
{
  const s = slide("Phase 1 results: warm start vs fresh spawn", "Phase 1 results");
  s.addImage(fit(fig("ppt_forgetting_warm_vs_fresh_3regions_fixed.png"), MX, 1.2, 5.6, 4.6, { alt: "Average forgetting, warm start vs fresh spawn" }));
  s.addText(bullets([
    "Forgetting = increase in error on a regime seen before, after adapting (lower is better).",
    [B("Warm start "), T("(reuse archived weights) gives lower forecast error in 28/34 events at region 22 and 24/36 at region 29.")],
    "On forgetting it wins only at region 29, so the choice is region-dependent.",
    [B("Default: "), T("reactivate by pooled top-3 blend, the most consistent winner.")],
  ]), body(null, { x: 6.3, y: 1.4, w: 3.2, h: 5.5, fontSize: 13 }));
}
{
  const s = slide("Key findings", "Phase 1 results");
  const cards = [
    ["0 / 250", "decoys selected", C.accent1], ["3 / 536", "physically impossible links", C.accent2], ["62%", "links persist at 2x resolution", C.accent3],
    ["32 / 50", "regions: better 1-day skill", C.accent4], ["up to 9.9x", "faster drift detection (causal channel)", C.accent5], ["104 / 171", "regime changes reuse an archived expert", C.accent6],
  ];
  cards.forEach((c, i) => stat(s, MX + (i % 3) * 3.05, 1.35 + Math.floor(i / 3) * 1.75, 2.9, c[0], c[1], c[2]));
  s.addText(bullets([
    "The causal graph is trustworthy by every check that can reject it.",
    "It makes source selection meaningful and helps at the lead its links match (1 day).",
    "Watching causal structure reveals drift early; recurring regimes make archived experts valuable.",
  ]), body(null, { x: MX, y: 4.95, w: 9, h: 2.0, fontSize: 15 }));
}
{
  const s = slide("Current limitations", "Phase 1 results");
  s.addText(bullets([
    "No independently verified ground truth exists for real climate causal links: every claim rests on indirect checks (decoys, physical plausibility, resolution persistence).",
    "The validated graph is short-lag (mostly 1–2 days) while the main forecaster runs at 7 days: a time-scale mismatch that limits the prior there. A graph with lags of 7+ days is the fix.",
    "Forecast comparisons use one training seed and are made against the unguided splitter; random- and correlation-prior controls of equal size are the next runs.",
    "Drift and lifecycle results cover three regions so far.",
  ]), body(null, { x: MX, y: 1.3, w: 9, h: 5.5, fontSize: 18, paraSpaceAfter: 16 }));
}

// Phase 2 ------------------------------------------------------------
pres.addSection({ title: "Phase 2" });
{
  const s = slide("Phase 2 plan", "Phase 2");
  s.addImage(fit(fig("workflows/png/wf_phase2.png"), MX, 1.15, 9, 5.75, { alt: "Phase 2 India-focused forecasting workflow" }));
}
{
  const s = slide("Phase 2 plan", "Phase 2");
  s.addText(bullets([
    [B("India hourly system: "), T("shared GNN backbone over IMDAA (~12 km, hourly) with ERA5 boundary context; region-specific experts gated by Phase 1-validated causal drivers.")],
    [B("Lead-matched causal drivers: "), T("PCMCI tested only at lags ≥ forecast lead, with decoy, speed and persistence checks.")],
    [B("72-hour forecasts: "), T("autoregressive roll-out of 2 m temperature, humidity, 10 m wind and MSLP, validated against ISD station observations.")],
    [B("Selective adaptation: "), T("per-region drift detection; retrain, spawn or reactivate only the affected expert.")],
    [B("Benchmark: "), T("GraphCast and WeatherNext on the same data, horizons and protocol.")],
    [B("Global scaling: "), T("larger meshes, more regimes and long-range teleconnections once India is competitive.")],
  ]), body(null, { x: MX, y: 1.3, w: 9, h: 5.6, fontSize: 17, paraSpaceAfter: 14 }));
}
{
  const s = slide("14 | References", "Phase 2");
  s.addText([
    "[1] Y.-X. Wu et al., \"Discovering invariant rationales for graph neural networks,\" Proc. ICLR, 2022.",
    "[2] A. Ghaffari, S. Sheikhi and E. Gilman, \"Graph-conditioned mixture of graph neural network experts for traffic forecasting,\" arXiv:2605.30486, 2026.",
    "[3] H. Cao et al., \"Geometric mixture-of-experts with curvature-guided adaptive routing for graph representation learning,\" arXiv:2603.22317, 2026.",
    "[4] S. Wu et al., \"GraphMETRO: Mitigating complex graph distribution shifts via mixture of aligned experts,\" NeurIPS 37, pp. 9358–9387, 2024.",
    "[5] L. Kong et al., \"Dynamic mixture-of-experts for incremental graph learning,\" arXiv:2508.09974, 2025.",
    "[6] H. Li et al., \"Out-of-distribution generalization on graphs: A survey,\" arXiv:2202.07987, 2022.",
    "[7] R. Lam et al., \"Learning skillful medium-range global weather forecasting,\" Science 382, pp. 1416–1421, 2023.",
    "[8] J. Runge et al., \"Detecting and quantifying causal associations in large nonlinear time series datasets,\" Science Advances 5, eaau4996, 2019.",
    "[9] J.-Y. Lee et al., \"Real-time multivariate indices for the boreal summer intraseasonal oscillation,\" Climate Dynamics 40, pp. 493–509, 2013.",
    "[10] S. N. Rodionov, \"A sequential algorithm for testing climate regime shifts,\" GRL 31, L09204, 2004.",
    "[11] A. Bifet and R. Gavaldà, \"Learning from time-changing data with adaptive windowing,\" SIAM SDM, 2007.",
  ].map((t, i, a) => ({ text: t, options: i < a.length - 1 ? { breakLine: true } : {} })),
  body(null, { x: MX, y: 1.25, w: 9, h: 5.7, fontSize: 13, paraSpaceAfter: 8 }));
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  const { applyTheme } = require(process.env.APPLY_THEME);
  await applyTheme(OUT, THEME);
  console.log("slides:", n, "->", OUT);
})();
