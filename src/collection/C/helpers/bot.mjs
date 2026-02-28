// bot.mjs
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import puppeteer from "puppeteer";
import axe from "axe-core";
import {
    getPresetConfig,
    safeNum,
    buildAxeSummary,
    extractContrastFromAxe,
    getShadowRootCount,
    round1
} from "./bot_helper.mjs";

function getArg(name, defVal = "") {
    const i = process.argv.indexOf(name);
    if (i === -1) return defVal;
    const v = process.argv[i + 1];
    return v === undefined ? defVal : v;
}

function mustArg(name) {
    const v = getArg(name, "");
    if (!v) throw new Error(`Missing required arg: ${name}`);
    return v;
}

function toInt(v, defVal) {
    const n = Number.parseInt(String(v), 10);
    return Number.isFinite(n) ? n : defVal;
}

function nowIso() {
    return new Date().toISOString();
}

async function ensureParent(filePath) {
    await fs.mkdir(path.dirname(filePath), { recursive: true });
}

async function writeJson(filePath, obj) {
    await ensureParent(filePath);
    await fs.writeFile(filePath, JSON.stringify(obj, null, 2), "utf-8");
}

async function run() {
    const url = mustArg("--url");
    const outHtml = mustArg("--out-html");
    const outSs = mustArg("--out-ss");
    const outMetrics = mustArg("--out-metrics");
    const timeoutMs = toInt(getArg("--timeout-ms", "60000"), 60000);
    const preset = getArg("--preset", "lighthouse-desktop");

    const startedAt = nowIso();
    const pconf = getPresetConfig(preset);

    let browser = null;

    try {
        await ensureParent(outHtml);
        await ensureParent(outSs);
        await ensureParent(outMetrics);

        browser = await puppeteer.launch({
            headless: "new",
            args: ["--no-sandbox", "--disable-setuid-sandbox"],
        });

        const page = await browser.newPage();
        await page.setBypassCSP(true);
        await page.setViewport(pconf.viewport);
        await page.setUserAgent(pconf.userAgent);
        page.setDefaultNavigationTimeout(timeoutMs);
        page.setDefaultTimeout(timeoutMs);

        // LCP + CLS (render-time)
        await page.evaluateOnNewDocument(() => {
            window.__renderm = { lcp_ms: null, cls: 0 };

            try {
                new PerformanceObserver((list) => {
                    for (const e of list.getEntries()) {
                        if (typeof e.startTime === "number") window.__renderm.lcp_ms = e.startTime;
                    }
                }).observe({ type: "largest-contentful-paint", buffered: true });

                new PerformanceObserver((list) => {
                    for (const e of list.getEntries()) {
                        if (!e.hadRecentInput && typeof e.value === "number") window.__renderm.cls += e.value;
                    }
                }).observe({ type: "layout-shift", buffered: true });
            } catch {
                // ignore
            }
        });

        const resp = await page.goto(url, { waitUntil: pconf.waitUntil, timeout: timeoutMs });
        const httpStatus = resp ? resp.status() : null;
        const finalUrl = page.url();

        if (pconf.settleMs > 0) {
            await new Promise((r) => setTimeout(r, pconf.settleMs));
        }

        // Save artifacts
        const html = await page.content();
        await fs.writeFile(outHtml, html, "utf-8");
        await page.screenshot({ path: outSs, fullPage: pconf.screenshotFullPage });

        // Navigation timings (replace your existing nav extraction + pickNavigationTimings usage)
        const timing = await page.evaluate(() => {
            // Prefer modern navigation entry, but return only primitives
            const nav = performance.getEntriesByType("navigation");
            if (nav && nav.length) {
                const n = nav[0];
                // These are relative to navigationStart already (startTime=0), but we return numbers directly
                return {
                    ttfb_ms: typeof n.responseStart === "number" ? n.responseStart : null,
                    dom_content_loaded_ms: typeof n.domContentLoadedEventEnd === "number" ? n.domContentLoadedEventEnd : null,
                    load_time_ms: typeof n.loadEventEnd === "number" ? n.loadEventEnd : null,
                };
            }

            // Fallback: legacy performance.timing
            const t = performance.timing;
            const ns = t.navigationStart || 0;
            const ttfb = (t.responseStart || 0) - ns;
            const dcl = (t.domContentLoadedEventEnd || 0) - ns;
            const load = (t.loadEventEnd || 0) - ns;

            return {
                ttfb_ms: Number.isFinite(ttfb) ? ttfb : null,
                dom_content_loaded_ms: Number.isFinite(dcl) ? dcl : null,
                load_time_ms: Number.isFinite(load) ? load : null,
            };
        });

        // LCP/CLS read
        const lcpc = await page.evaluate(() => window.__renderm || { lcp_ms: null, cls: 0 });

        // Shadow Root Count
        const shadowCount = await getShadowRootCount(page);

        await page.addScriptTag({ content: axe.source });

        const axeRules = [
            "color-contrast",
            "document-title",
            "html-has-lang",
            "html-lang-valid",
            "image-alt",
            "label",
            "link-name",
            "aria-valid-attr",
            "aria-required-attr",
            "aria-allowed-attr",
        ];

        const axeResults = await page.evaluate(async (rules) => {
            return await window.axe.run(document, {
                runOnly: { type: "rule", values: rules },
                resultTypes: ["violations", "passes", "incomplete", "inapplicable"],
            });
        }, axeRules);

        const axeSummary = buildAxeSummary(axeResults);
        const contrast = extractContrastFromAxe(axeResults);

        const metrics = {
            ok: true,
            error: "",
            started_at: startedAt,
            collected_at: nowIso(),
            url,
            final_url: finalUrl,
            http_status: httpStatus,
            preset,
            viewport: pconf.viewport,
            load_time_ms: timing.load_time_ms,
            dom_content_loaded_ms: timing.dom_content_loaded_ms,
            ttfb_ms: timing.ttfb_ms,
            lcp_ms: lcpc?.lcp_ms != null ? round1(Number(lcpc.lcp_ms)) : null,
            cls: safeNum(Number(lcpc?.cls)),
            shadow_root_count: shadowCount,

            // contrast + axe summaries
            contrast_violation_count: contrast.contrast_violation_count,
            contrast_pass_count: contrast.contrast_pass_count,
            contrast_ratio_stats: contrast.contrast_ratio_stats, // new stats
            min_contrast_ratio: contrast.min_contrast_ratio,

            ...axeSummary,
        };

        await writeJson(outMetrics, metrics);
        await browser.close();
        process.exit(0);
    } catch (e) {
        const msg = e && typeof e.stack === "string" ? e.stack : (e && typeof e.message === "string" ? e.message : String(e));

        try {
            await writeJson(outMetrics, {
                ok: false,
                error: msg.slice(0, 4000),
                started_at: startedAt,
                collected_at: nowIso(),
                url,
                shadow_root_count: null,
                contrast_violation_count: null,
                min_contrast_ratio: null,
            });
        } catch {
            // best effort
        }

        if (browser) {
            try {
                await browser.close();
            } catch {
                // ignore
            }
        }

        console.error(msg);
        process.exit(2);
    }
}

run();

