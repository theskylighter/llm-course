// End-to-end smoke test for the study site, driven by headless Chrome (playwright-core).
//
//   npm run build && npm run e2e                    # builds, then starts its own preview server
//   BASE=https://your-site.vercel.app npm run e2e   # test a deployed site instead
//   CHROME_PATH="C:/Program Files/Google/Chrome/Application/chrome.exe" npm run e2e
//
// It checks behaviour a reader depends on: the sidebar keeps its scroll position while navigating,
// both side panels collapse and remember it, the phone drawer works, nothing scrolls sideways on a
// phone, and the console stays clean. Checks that cannot apply (for example scrolling a course list
// that is shorter than the screen) are reported as SKIP rather than silently passing.
import { spawn, execSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(readFileSync(resolve(here, "../src/data/lessons.json"), "utf-8"));
const PORT = 4173;
let BASE = process.env.BASE;
let server = null;

// ---------- locate Chrome ----------
function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    "C:/Program Files/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable", "/usr/bin/chromium", "/usr/bin/chromium-browser",
  ];
  const found = candidates.find((c) => c && existsSync(c));
  if (!found) throw new Error("Chrome not found. Install it or set CHROME_PATH to the executable.");
  return found;
}

// ---------- own preview server (only when BASE is not given) ----------
async function startServer() {
  server = spawn("npx", ["vite", "preview", "--port", String(PORT), "--strictPort"], { cwd: resolve(here, ".."), shell: true, stdio: "ignore" });
  BASE = `http://localhost:${PORT}`;
  for (let i = 0; i < 60; i++) {
    try { if ((await fetch(BASE + "/")).ok) return; } catch { /* not up yet */ }
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("preview server did not start (run `npm run build` first)");
}
function stopServer() {
  if (!server) return;
  try {
    if (process.platform === "win32") execSync(`taskkill /pid ${server.pid} /T /F`, { stdio: "ignore" });
    else server.kill("SIGTERM");
  } catch { /* already gone */ }
}

// ---------- tiny assertion helpers ----------
const results = [];
const check = (name, ok, detail = "") => results.push({ s: ok ? "PASS" : "FAIL", text: `${name}${detail ? "  - " + detail : ""}` });
const skip = (name, why) => results.push({ s: "SKIP", text: `${name}  - ${why}` });
const settle = (page, ms = 320) => page.waitForTimeout(ms); // wait out the 200ms panel/drawer transitions

// ---------- pick test lessons from the exported data ----------
const lessons = data.lessons;
const rich = lessons.find((l) => l.toc.length >= 3) ?? lessons[0];
const other = lessons.find((l) => l.slug !== rich.slug) ?? rich;
const chapterOf = (slug) => data.chapters.find((c) => c.lessons.includes(slug))?.title;

async function desktop(browser) {
  const page = await (await browser.newContext({ viewport: { width: 1440, height: 900 } })).newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  const metrics = () => page.evaluate(() => {
    const w = (s) => { const e = document.querySelector(s); return e ? Math.round(e.getBoundingClientRect().width) : null; };
    const m = document.querySelector(".content").getBoundingClientRect();
    return { left: w("aside.sidebar"), right: w("aside.toc-aside"), centre: Math.round(m.left + m.width / 2) };
  });

  await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  await page.evaluate(() => { localStorage.clear(); sessionStorage.clear(); });
  await page.reload({ waitUntil: "networkidle" });

  // --- home page ---
  await page.goto(BASE + "/", { waitUntil: "networkidle" });
  const cards = await page.locator(".roadmap .chapter-card").count();
  check("home: roadmap cards render", cards >= 1, `${cards} chapter cards`);
  const headings = await page.locator("main h2").allTextContents();
  const dupes = headings.filter((h, i) => headings.indexOf(h) !== i);
  check("home: no duplicated section headings", dupes.length === 0, dupes.join(", "));

  // --- lesson page + layout ---
  await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  check("lesson: title rendered", (await page.locator("article.lesson h1").textContent()) === rich.title);
  const geo = await page.evaluate(() => {
    const r = (s) => document.querySelector(s)?.getBoundingClientRect();
    const l = r("aside.sidebar"), c = r(".content"), t = r("aside.toc-aside");
    return { leftOfContent: !!l && l.right <= c.left + 1, rightOfContent: !!t && t.left >= c.right - 1 };
  });
  check("layout: course list left, 'On this page' right", geo.leftOfContent && geo.rightOfContent, JSON.stringify(geo));

  // --- sidebar keeps its scroll position across navigation ---
  // Expand every chapter, as a reader browsing the course would, so the list is as long as it gets.
  for (const t of await page.locator("#course-sidebar button.chapter-toggle").all()) {
    if ((await t.getAttribute("aria-expanded")) === "false") await t.click();
  }
  const scrollable = await page.evaluate(() => { const n = document.querySelector(".course-nav"); return n.scrollHeight - n.clientHeight > 80; });
  if (scrollable) {
    // Scroll the list down, then click a lesson link that is *visible there*, as a reader would.
    // (Playwright's own click() would scroll the link into view first and move the list.)
    const set = await page.evaluate(() => { const n = document.querySelector(".course-nav"); n.scrollTop = Math.min(n.scrollHeight - n.clientHeight, 700); window.__node = n; return n.scrollTop; });
    const targetPath = await page.evaluate((current) => {
      const nav = document.querySelector(".course-nav").getBoundingClientRect();
      const link = [...document.querySelectorAll("#course-sidebar .pages a")].find((a) => {
        const r = a.getBoundingClientRect();
        return r.top >= nav.top + 8 && r.bottom <= nav.bottom - 8 && a.getAttribute("href") !== "/" + current;
      });
      if (!link) return null;
      const href = link.getAttribute("href");
      link.click();
      return href;
    }, rich.slug);
    if (!targetPath) throw new Error("no visible lesson link found in the scrolled sidebar");
    await page.waitForURL(new RegExp(`${targetPath}$`));
    const after = await page.evaluate(() => ({ same: document.querySelector(".course-nav") === window.__node, top: document.querySelector(".course-nav").scrollTop }));
    check("sidebar: same element after navigating (never remounted)", after.same);
    check("sidebar: scroll position held while navigating", set > 50 && Math.abs(after.top - set) < 2, `set=${set} after=${after.top}`);
    await page.waitForTimeout(150);
    await page.reload({ waitUntil: "networkidle" });
    const restored = await page.evaluate(() => document.querySelector(".course-nav").scrollTop);
    check("sidebar: scroll position restored after a full reload", Math.abs(restored - after.top) < 2, `before=${after.top} after=${restored}`);
    await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  } else {
    skip("sidebar: scroll retention", "course list is shorter than the screen even with every chapter open (add more lessons to exercise this)");
  }

  // --- On this page + scroll-spy ---
  await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  const links = await page.locator(".toc-aside li a").count();
  check("toc: lists the page's headings", links >= 3, `${links} links`);
  const first = await page.locator(".toc-aside li.active a").first().textContent();
  // jump to the furthest heading the page can actually scroll to (short pages can't reach the last ones)
  const reachable = await page.evaluate((ids) => {
    const max = document.documentElement.scrollHeight - window.innerHeight;
    const ok = ids.filter((id) => { const el = document.getElementById(id); return el && el.getBoundingClientRect().top + window.scrollY - 120 <= max; });
    return ok[ok.length - 1];
  }, rich.toc.map((t) => t.id));
  if (reachable && reachable !== rich.toc[0].id) {
    await page.evaluate((id) => document.getElementById(id)?.scrollIntoView(), reachable);
    await settle(page, 450);
    const later = await page.locator(".toc-aside li.active a").first().textContent();
    check("toc: scroll-spy highlights the section being read", !!first && !!later && first !== later, `${first} -> ${later}`);
  } else {
    skip("toc: scroll-spy", "page is too short to scroll past its first heading");
  }

  // --- collapse / expand both panels, shortcut, persistence ---
  const base = await metrics();
  const leftBtn = page.locator("aside.sidebar .panel-head button"), leftRail = page.locator("aside.sidebar .rail-btn");
  const rightBtn = page.locator("aside.toc-aside .panel-head button"), rightRail = page.locator("aside.toc-aside .rail-btn");
  await leftBtn.click(); await settle(page);
  let m = await metrics();
  check("collapse: left panel becomes a 44px rail", m.left === 44 && m.right === base.right, JSON.stringify(m));
  check("collapse: rail shows an expand button", await leftRail.isVisible());
  await leftRail.click(); await settle(page);
  check("collapse: left panel expands again", (await metrics()).left === base.left);
  await page.keyboard.press("Control+b"); await settle(page);
  const viaKey = (await metrics()).left;
  await page.keyboard.press("Control+b"); await settle(page);
  check("collapse: Ctrl+B toggles the course sidebar", viaKey === 44 && (await metrics()).left === base.left);
  await rightBtn.click(); await settle(page);
  m = await metrics();
  check("collapse: right panel becomes a 44px rail, independently", m.right === 44 && m.left === base.left, JSON.stringify(m));
  await leftBtn.click(); await settle(page);
  m = await metrics();
  check("collapse: both collapsed -> lesson centred", m.left === 44 && m.right === 44 && Math.abs(m.centre - 720) < 30, `centre=${m.centre}`);
  const tabbable = await page.evaluate(() => [...document.querySelectorAll(".panel.collapsed .panel-body a, .panel.collapsed .panel-body button")].filter((e) => getComputedStyle(e).visibility !== "hidden").length);
  check("collapse: hidden panels are out of the tab order", tabbable === 0);
  await leftRail.click(); await settle(page);
  await page.reload({ waitUntil: "networkidle" });
  m = await metrics();
  check("collapse: choice survives a reload", m.right === 44 && m.left === base.left, JSON.stringify(m));
  await rightRail.click(); await settle(page);

  // --- answers toggle ---
  const answers = await page.locator("details.ans").count();
  if (answers > 0) {
    await page.locator("button.btn-ghost", { hasText: /answers/i }).click();
    const open = await page.evaluate(() => document.querySelectorAll("details.ans[open]").length);
    check("answers: 'Show answers' opens every answer", open === answers, `${open}/${answers}`);
  } else {
    const qLesson = lessons.find((l) => l.html.includes('class="ans"'));
    if (qLesson) {
      await page.goto(`${BASE}/${qLesson.slug}`, { waitUntil: "networkidle" });
      await page.locator("button.btn-ghost", { hasText: /answers/i }).click();
      const [o, t] = await page.evaluate(() => [document.querySelectorAll("details.ans[open]").length, document.querySelectorAll("details.ans").length]);
      check("answers: 'Show answers' opens every answer", o === t && t > 0, `${o}/${t}`);
    } else skip("answers toggle", "no lesson has answer blocks");
  }

  // --- theme, copy buttons, redirects, 404 ---
  await page.locator("button[aria-label*='theme']").click();
  check("theme: dark mode applies", await page.evaluate(() => document.documentElement.classList.contains("dark")));
  await page.locator("button[aria-label*='theme']").click();
  const codeLesson = lessons.find((l) => /codewrap (cpp|python)/.test(l.html));
  if (codeLesson) {
    await page.goto(`${BASE}/${codeLesson.slug}`, { waitUntil: "networkidle" });
    const [btns, blocks] = await page.evaluate(() => [document.querySelectorAll(".codewrap:not(.output):not(.text):not(.uml) .copybtn").length, document.querySelectorAll(".codewrap:not(.output):not(.text):not(.uml)").length]);
    if (blocks > 0) check("code: copy button on each code block", btns === blocks, `${btns}/${blocks}`);
  }
  await page.goto(`${BASE}/${rich.slug}.html`, { waitUntil: "networkidle" });
  check("routing: old .html URLs redirect to clean routes", page.url().endsWith(`/${rich.slug}`), page.url());
  await page.goto(`${BASE}/definitely-not-a-lesson`, { waitUntil: "networkidle" });
  check("routing: unknown URL shows a not-found page", ((await page.locator("h1").first().textContent()) ?? "").toLowerCase().includes("not found"));

  // --- design rule: no emoji in the UI chrome ---
  await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  const emoji = await page.evaluate(() => {
    const text = [...document.querySelectorAll(".topbar, .panel, .meta, .pager, .boxtitle")].map((e) => e.textContent).join(" ");
    return (text.match(/[\u{1F300}-\u{1FAFF}\u{2600}-\u{26FF}\u{2705}\u{274C}]/gu) || []).join("");
  });
  check("design: no emoji in the interface chrome", emoji === "", emoji);
  check("console: no errors on desktop", errors.length === 0, errors.slice(0, 2).join(" | "));
}

async function phone(browser) {
  const ctx = await browser.newContext({ ...devices["iPhone 13"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${BASE}/${rich.slug}`, { waitUntil: "networkidle" });
  const vw = page.viewportSize().width;
  const menu = page.locator("button.menu-btn");
  const box = await menu.boundingBox();
  check("phone: menu button visible, inside the screen, >= 40px", !!box && box.x >= 0 && box.x + box.width <= vw && box.width >= 40 && box.height >= 40, JSON.stringify(box));
  const hidden = () => page.evaluate(() => getComputedStyle(document.querySelector("#course-sidebar")).visibility === "hidden");
  check("phone: drawer is closed on load", await hidden());
  await menu.tap(); await settle(page, 400);
  const drawer = await page.evaluate(() => { const r = document.querySelector("#course-sidebar").getBoundingClientRect(); return { left: r.left, width: r.width }; });
  check("phone: menu opens the drawer from the left, backdrop visible", drawer.left === 0 && drawer.width < vw && (await menu.getAttribute("aria-expanded")) === "true", JSON.stringify(drawer));
  check("phone: page scroll is locked while the drawer is open", await page.evaluate(() => document.body.style.overflow === "hidden"));
  await page.keyboard.press("Escape"); await settle(page, 400);
  check("phone: Escape closes the drawer", await hidden());
  await menu.tap(); await settle(page, 400);
  await page.mouse.click(vw - 8, 300); await settle(page, 400);
  check("phone: tapping the backdrop closes the drawer", await hidden());
  if (other.slug !== rich.slug) {
    await menu.tap(); await settle(page, 400);
    const ch = chapterOf(other.slug);
    const toggle = page.locator("#course-sidebar button.chapter-toggle", { hasText: ch ?? "" }).first();
    if ((await toggle.getAttribute("aria-expanded")) === "false") await toggle.tap();
    await page.locator(`#course-sidebar a[href="/${other.slug}"]`).first().tap();
    await page.waitForURL(new RegExp(`/${other.slug}$`)); await settle(page, 400);
    check("phone: choosing a lesson navigates and closes the drawer", await hidden());
  }

  // every page must fit the screen width (no sideways scrolling)
  const bad = [];
  for (const p of ["/", ...lessons.map((l) => "/" + l.slug)]) {
    await page.goto(BASE + p, { waitUntil: "load" });
    const over = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
    if (over > 1) bad.push(`${p} (+${over}px)`);
  }
  check(`phone: no page scrolls sideways (${lessons.length + 1} pages)`, bad.length === 0, bad.slice(0, 5).join(", "));
  check("console: no errors on phone", errors.length === 0, errors.slice(0, 2).join(" | "));
  await ctx.close();
}

try {
  if (!BASE) await startServer();
  const browser = await chromium.launch({ executablePath: findChrome(), headless: true });
  await desktop(browser);
  await phone(browser);
  await browser.close();
} catch (e) {
  results.push({ s: "FAIL", text: `test run crashed: ${e.message}` });
} finally {
  stopServer();
}

for (const r of results) console.log(`${r.s}  ${r.text}`);
const failed = results.filter((r) => r.s === "FAIL").length;
console.log(`\n${results.filter((r) => r.s === "PASS").length} passed, ${failed} failed, ${results.filter((r) => r.s === "SKIP").length} skipped  (${BASE})`);
process.exit(failed ? 1 : 0);
