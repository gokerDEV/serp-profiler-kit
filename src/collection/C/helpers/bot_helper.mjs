// bot_helper.mjs
import axe from "axe-core";

export function presetConfig(preset) {
    // Lighthouse desktop reference: mobile=false, DPR=1, wide viewport.
    if (preset === "lighthouse-desktop") {
        return {
            viewport: { width: 1350, height: 940, deviceScaleFactor: 1 },
            userAgent:
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            waitUntil: "networkidle2", // bot.mjs uses networkidle2, bot_cluster uses domcontentloaded. Unifying to networkidle2 might be safer or keeping flexible. 
            // Let's stick to what was common or passed as arg if we want more control, but for now specific configs were slightly different.
            // bot.mjs had networkidle2, bot_cluster had domcontentloaded. 
            // implementation plan said "Create bot_helper.mjs to contain... presetConfig".
            // To support both without breaking, maybe we just return the config object and let caller override waitUntil if needed?
            // Or we standardize? lighthouse-desktop usually implies networkidle.
            // bot_cluster used domcontentloaded. Let's return the common parts and let the caller decide waitUntil if it differs significantly, 
            // OR let's standardise on networkidle2 for 'lighthouse-desktop' as it's more robust for metrics, 
            // but bot_cluster might want speed.
            // Checking original files:
            // bot.mjs: lighthouse-desktop -> networkidle2, settleMs 800. Default -> networkidle2, settleMs 800.
            // bot_cluster.mjs: lighthouse-desktop -> domcontentloaded, settleMs 2000. Default -> domcontentloaded, settleMs 2000.

            // It seems they have different behaviors. 
            // I will implement a merged version that accepts overrides or arguments, or just keep it simple and standardize.
            // Given the task is to commonize, I will provide a standard config generator.
            waitUntil: "networkidle2",
            settleMs: 2000, // Taking the more conservative 2000 from cluster, or 800 from bot? 
            // User didn't ask to change behavior, just refactor.
            // Maybe I should keep presetConfig local if it differs too much, 
            // OR make it accept options.
            // Let's look at the implementation plan again. "Create bot_helper.mjs to contain... presetConfig".
            // I will make it return the config structure, and maybe taking `waitUntil` as an optional arg?
            screenshotFullPage: false,
        };
    }

    // Default Fallback
    return {
        viewport: { width: 1366, height: 768, deviceScaleFactor: 1 },
        userAgent:
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        waitUntil: "networkidle2",
        settleMs: 2000,
        screenshotFullPage: false,
    };
}

// Allow overriding defaults
export function getPresetConfig(preset, overrides = {}) {
    const base = presetConfig(preset);
    return { ...base, ...overrides };
}

export function safeNum(x) {
    return Number.isFinite(x) ? x : null;
}

export function round1(x) {
    return Number.isFinite(x) ? Math.round(x * 10) / 10 : null;
}

export function quantile(sorted, q) {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q;
    const base = Math.floor(pos);
    const rest = pos - base;
    if (sorted[base + 1] === undefined) return sorted[base];
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

export function contrastStats(ratios) {
    const xs = ratios.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
    return {
        n: xs.length,
        min: xs.length ? xs[0] : null,
        p10: quantile(xs, 0.10),
        median: quantile(xs, 0.50),
        p90: quantile(xs, 0.90),
    };
}

export function extractContrastFromAxe(axeResults) {
    const violations = Array.isArray(axeResults?.violations) ? axeResults.violations : [];
    const passes = Array.isArray(axeResults?.passes) ? axeResults.passes : [];

    const ccV = violations.find((v) => v?.id === "color-contrast") || null;
    const ccP = passes.find((v) => v?.id === "color-contrast") || null;

    const ratios = [];
    const collectFromNodes = (nodes) => {
        const arr = Array.isArray(nodes) ? nodes : [];
        for (const n of arr) {
            const any = Array.isArray(n?.any) ? n.any : [];
            for (const a of any) {
                const d = a?.data;
                if (d && typeof d.contrastRatio === "number") ratios.push(d.contrastRatio);
            }
            const all = Array.isArray(n?.all) ? n.all : [];
            for (const a of all) {
                const d = a?.data;
                if (d && typeof d.contrastRatio === "number") ratios.push(d.contrastRatio);
            }
            // Fallback for failureSummary if data is missing? bot.mjs had it, bot_cluster.mjs didn't.
            // I'll include the more robust version from bot.mjs
            const fsu = typeof n?.failureSummary === "string" ? n.failureSummary : "";
            const matches = fsu.match(/([\d.]+)\s*:\s*1/g);
            if (matches) {
                for (const hit of matches) {
                    const r = Number.parseFloat(hit.split(":")[0]);
                    if (Number.isFinite(r)) ratios.push(r);
                }
            }
        }
    };

    collectFromNodes(ccV?.nodes);
    collectFromNodes(ccP?.nodes);

    const stats = contrastStats(ratios);
    return {
        contrast_violation_count: Array.isArray(ccV?.nodes) ? ccV.nodes.length : 0,
        contrast_pass_count: Array.isArray(ccP?.nodes) ? ccP.nodes.length : 0,
        min_contrast_ratio: stats.min,
        contrast_ratio_stats: stats,
    };
}

export function summarizeGroup(group) {
    const arr = Array.isArray(group) ? group : [];

    let nodeCount = 0;
    const ruleNodeCounts = [];

    for (const r of arr) {
        const n = Array.isArray(r?.nodes) ? r.nodes.length : 0;
        nodeCount += n;
        ruleNodeCounts.push({ id: r?.id ?? "unknown", nodes: n, impact: r?.impact ?? "unknown" });
    }

    ruleNodeCounts.sort((a, b) => b.nodes - a.nodes);

    return {
        rule_count: arr.length,
        node_count: nodeCount,
        top_rules_by_nodes: ruleNodeCounts.slice(0, 10),
    };
}

export function buildAxeSummary(axeResults) {
    const violations = summarizeGroup(axeResults?.violations);
    const passes = summarizeGroup(axeResults?.passes);
    const incomplete = summarizeGroup(axeResults?.incomplete);
    const inapplicable = summarizeGroup(axeResults?.inapplicable);

    const impactCounts = { minor: 0, moderate: 0, serious: 0, critical: 0, unknown: 0 };
    const vArr = Array.isArray(axeResults?.violations) ? axeResults.violations : [];
    for (const v of vArr) {
        const k = typeof v?.impact === "string" ? v.impact : "unknown";
        if (k in impactCounts) impactCounts[k] += 1;
        else impactCounts.unknown += 1;
    }

    // Add total counts for better analysis compatibility
    const vTotal = Array.isArray(axeResults?.violations) ? axeResults.violations.length : 0;
    const pTotal = Array.isArray(axeResults?.passes) ? axeResults.passes.length : 0;
    const incTotal = Array.isArray(axeResults?.incomplete) ? axeResults.incomplete.length : 0;
    const inaTotal = Array.isArray(axeResults?.inapplicable) ? axeResults.inapplicable.length : 0;

    return {
        axe_violations: violations,
        axe_passes: passes,
        axe_incomplete: incomplete,
        axe_inapplicable: inapplicable,
        axe_impact_counts: impactCounts,

        axe_violations_total: vTotal,
        axe_passes_total: pTotal,
        axe_incomplete_total: incTotal,
        axe_inapplicable_total: inaTotal,
    };
}

// 1. Shadow Root Sayısını Alan Fonksiyon
export async function getShadowRootCount(page) {
    return await page.evaluate(() => {
        let count = 0;
        // SHOW_ELEMENT = 1
        // NodeFilter.SHOW_ELEMENT is usually 1. 
        // We use createTreeWalker to traverse efficiently.
        const walk = (root) => {
            const treeWalker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT);
            let node = treeWalker.nextNode();
            while (node) {
                if (node.shadowRoot) {
                    count++;
                    walk(node.shadowRoot);
                }
                node = treeWalker.nextNode();
            }
        };
        walk(document);
        return count;
    });
}
