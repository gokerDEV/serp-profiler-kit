"""
thesis_table.py
---------------
Tez icin sadece RQ9-RQ13 tablolarini ureten LaTeX generator'u.

Veri kaynaklari:
  - src/generators/A/tables.py ile ayni analiz ciktilari
  - data/analysis/C..H altindaki hazir ozet dosyalari
"""

import argparse
import itertools
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Proje kokunu path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.generators.A.tables import format_ci, format_p_val, tex_escape
from src.helpers.data_loader import get_feature_categories


ENGINE_LABELS = {
    "google": "Google",
    "brave": "Brave",
    "mojeek": "Mojeek",
}
ENGINE_ORDER = ["google", "brave", "mojeek"]

FEATURE_LABELS = {
    "lcp_ms": "LCP",
    "ttfb_ms": "TTFB",
    "cls": "CLS",
    "axe_score": "AXE Score",
    "contrast_score": "Contrast Score",
    "flesch_reading_ease": "Flesch Reading Ease",
    "flesch_kincaid_grade": "Flesch-Kincaid Grade",
    "sim_title": "Baslik Benzerligi",
    "sim_description": "Description Benzerligi",
    "sim_h1": "H1 Benzerligi",
    "sim_content": "Icerik Benzerligi",
}

CATEGORY_LABELS = {
    "semantic": "Semantik",
    "readability": "Okunabilirlik",
    "performance": "Performans",
    "accessibility": "Erisilebilirlik",
}


def _fmt_num(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_effect_with_ci(row: pd.Series) -> str:
    return f"{row['effect_size']:.3f} [{row['ci_lower_95']:.3f}, {row['ci_upper_95']:.3f}]"


def _practical_flag(row: pd.Series) -> str:
    return "Pratik" if bool(row.get("practical_flag", False)) else "-"


def _cohens_d(x: pd.Series, y: pd.Series) -> float:
    x = x.dropna()
    y = y.dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = math.sqrt((((len(x) - 1) * x.std(ddof=1) ** 2) + ((len(y) - 1) * y.std(ddof=1) ** 2)) / (len(x) + len(y) - 2))
    if pooled == 0:
        return np.nan
    return (x.mean() - y.mean()) / pooled


def _effect_interpretation(d: float) -> str:
    if pd.isna(d):
        return "Yetersiz veri"
    ad = abs(d)
    if ad >= 0.8:
        return "Buyuk"
    if ad >= 0.5:
        return "Orta"
    if ad >= 0.2:
        return "Kucuk"
    return "Ihmal edilebilir"


class ThesisTableGenerator:
    def __init__(self, results_path: str, dataset_path: str, output_dir: str):
        self.results_path = results_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_categories = get_feature_categories()

    def _write(self, filename: str, latex: str) -> None:
        path = self.output_dir / filename
        path.write_text(latex, encoding="utf-8")
        print(f"Wrote {path}")

    def _full_engine_order(self, values) -> list[str]:
        ordered = [e for e in ENGINE_ORDER if e in set(values)]
        extras = [e for e in sorted(set(values)) if e not in ordered]
        return ordered + extras

    def _rows_to_table(self, env: str, caption: str, label: str, colspec: str, header_lines: list[str], body_lines: list[str]) -> str:
        lines = [
            f"\\begin{{{env}}}[htbp!]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\small",
            f"\\begin{{tabular*}}{{\\textwidth}}{{{colspec}}}",
            "\\toprule",
            *header_lines,
            "\\midrule",
            *body_lines,
            "\\bottomrule",
            "\\end{tabular*}",
            f"\\end{{{env}}}",
        ]
        return "\n".join(lines) + "\n"

    def generate_all_thesis_tables(self) -> None:
        self.generate_tab_rq9_concentration_summary()
        self.generate_tab_rq9_concentration_effects()
        self.generate_tab_rq9_semantic_rank_trends()
        self.generate_tab_rq9_confirmatory_semantics()
        self.generate_tab_rq10_nested_readability()
        self.generate_tab_rq11_nested_perf_access()
        self.generate_tab_rq11_confirmatory_performance()
        self.generate_tab_rq11_confirmatory_accessibility()
        self.generate_tab_rq11_supplementary_perf_access()
        self.generate_tab_rq12_heterogeneity_summary()
        self.generate_tab_rq13_robustness_main()
        self.generate_tab_rq13_subdataset_sensitivity()
        self.generate_tab_rq13_ablation_ndcg()
        self.generate_tab_rq13_ablation_stability()

    def generate_tab_rq9_concentration_summary(self) -> None:
        path = PROJECT_ROOT / "data/analysis/C/rq1_concentration.parquet"
        if not path.exists():
            return
        df = pd.read_parquet(path)
        df = df[(df["status"] == "ok") & (df["subset"] == "Full")].copy()
        grouped = df.groupby("search_engine", as_index=False).agg(
            mean_gini=("domain_gini", "mean"),
            mean_entropy=("domain_entropy_norm", "mean"),
            monopoly_rate=("domain_is_monopoly", "mean"),
        )
        body = []
        for engine in self._full_engine_order(grouped["search_engine"]):
            row = grouped[grouped["search_engine"] == engine].iloc[0]
            body.append(
                f"{ENGINE_LABELS.get(engine, engine.title())} & "
                f"{row['mean_gini']:.3f} & {row['mean_entropy']:.3f} & {row['monopoly_rate'] * 100:.1f} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Sorgu$\\times$motor duzeyinde gorunurluk yogunlasmasi ve alan adi cesitliligi ozeti",
            "tab:rq9_concentration_summary",
            "l@{\\extracolsep{\\fill}}ccc",
            ["\\textbf{Arama Motoru} & \\textbf{Ortalama Gini} & \\textbf{Ortalama Normalize Entropi} & \\textbf{Tekel Orani (\\%)} \\\\"],
            body,
        )
        self._write("tab_rq9_concentration_summary.tex", latex)

    def generate_tab_rq9_concentration_effects(self) -> None:
        path = PROJECT_ROOT / "data/analysis/C/rq1_concentration.parquet"
        if not path.exists():
            return
        df = pd.read_parquet(path)
        df = df[(df["status"] == "ok") & (df["subset"] == "Full")].copy()
        metrics = {
            "domain_gini": "Gorunurluk Gini",
            "domain_entropy_norm": "Normalize Entropi",
        }
        body = []
        for left, right in itertools.combinations(self._full_engine_order(df["search_engine"]), 2):
            pair = f"{ENGINE_LABELS.get(left, left.title())} vs {ENGINE_LABELS.get(right, right.title())}"
            left_df = df[df["search_engine"] == left]
            right_df = df[df["search_engine"] == right]
            for metric, label in metrics.items():
                diff = left_df[metric].mean() - right_df[metric].mean()
                d = _cohens_d(left_df[metric], right_df[metric])
                body.append(f"{pair} & {label} & {diff:.3f} & {_fmt_num(d)} & {_effect_interpretation(d)} \\\\")
        latex = self._rows_to_table(
            "table*",
            "Motor ciftleri icin gorunurluk yogunlasmasi karsilastirmalari",
            "tab:rq9_concentration_effects",
            "l@{\\extracolsep{\\fill}}lcll",
            ["\\textbf{Karsilastirma} & \\textbf{Metrik} & \\textbf{Ortalama Fark} & \\textbf{Cohen'in \\(d\\)} & \\textbf{Yorum} \\\\"],
            body,
        )
        self._write("tab_rq9_concentration_effects.tex", latex)

    def generate_tab_rq9_semantic_rank_trends(self) -> None:
        path = PROJECT_ROOT / "data/analysis/C/rq1_feature_trends.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[(df["subset"] == "Full") & (df["category"] == "semantic")].copy()
        rank_order = ["Top 1-3", "Rank 4-10", "Rank 11-20"]
        engines = self._full_engine_order(df["search_engine"])
        features = [f for f, cat in self.feature_categories.items() if cat == "semantic" and f in set(df["feature"])]
        body = []
        for feature in features:
            for engine in engines:
                sub = df[(df["feature"] == feature) & (df["search_engine"] == engine)].copy()
                if sub.empty:
                    continue
                sub["rank_bin"] = pd.Categorical(sub["rank_bin"], categories=rank_order, ordered=True)
                sub = sub.sort_values("rank_bin")
                vals = []
                for rank_bin in rank_order:
                    row = sub[sub["rank_bin"] == rank_bin]
                    vals.append("-" if row.empty else f"{row.iloc[0]['mean']:.3f}")
                body.append(
                    f"{FEATURE_LABELS.get(feature, feature)} & {ENGINE_LABELS.get(engine, engine.title())} & "
                    + " & ".join(vals)
                    + " \\\\"
                )
        latex = self._rows_to_table(
            "table*",
            "Siralama dilimlerine gore semantik benzerlik egilimleri",
            "tab:rq9_semantic_rank_trends",
            "l@{\\extracolsep{\\fill}}lccc",
            ["\\textbf{Ozellik} & \\textbf{Motor} & \\textbf{Top 1--3} & \\textbf{Sira 4--10} & \\textbf{Sira 11--20} \\\\"],
            body,
        )
        self._write("tab_rq9_semantic_rank_trends.tex", latex)

    def generate_tab_rq9_confirmatory_semantics(self) -> None:
        path = PROJECT_ROOT / "data/analysis/D/confirmatory_coeffs_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[(df["subset"] == "Full") & (df["model_id"] == "RQ2_Semantics_R_FE")].copy()
        df = df[df["term"].isin([k for k, v in self.feature_categories.items() if v == "semantic"])].copy()
        body = []
        for _, row in df.iterrows():
            body.append(
                f"Semantik (Full) & {FEATURE_LABELS.get(row['term'], row['term'])} & "
                f"{row['effect_size']:.3f} & {format_ci(row['ci_lower_95'], row['ci_upper_95'])} & "
                f"{format_p_val(row['p_raw'])} & {_practical_flag(row)} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Semantik blok icin confirmatory regresyon sonuclari",
            "tab:rq9_confirmatory_semantics",
            "l@{\\extracolsep{\\fill}}lrlll",
            ["\\textbf{Model} & \\textbf{Yordayici} & \\textbf{Tahmin (\\(\\beta^*\\))} & \\textbf{CI95} & \\textbf{\\(p\\)} & \\textbf{Flag} \\\\"],
            body,
        )
        self._write("tab_rq9_confirmatory_semantics.tex", latex)

    def generate_tab_rq10_nested_readability(self) -> None:
        path = PROJECT_ROOT / "data/analysis/D/nested_model_fit.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[df["Added_Block"].isin(["Semantics", "Readability"])].copy()
        gain = lambda r: "Yuksek" if r["Delta_R2"] >= 0.01 else ("Sinirli" if r["Delta_R2"] >= 0.002 else "Dusuk")
        body = []
        for _, row in df.iterrows():
            body.append(
                f"{row['Model']} & {row['Added_Block']} & {row['Delta_R2']:.3f} & {row['Delta_AIC']:.1f} & "
                f"{row['Delta_BIC']:.1f} & {format_p_val(row['LR_p'])} & {gain(row)} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Okunabilirlik blogunun artimli katkisi icin ic ice model karsilastirmalari",
            "tab:rq10_nested_readability",
            "l@{\\extracolsep{\\fill}}lrrrrr",
            ["\\textbf{Model} & \\textbf{Eklenen Blok} & \\textbf{\\(\\Delta R^2\\)} & \\textbf{\\(\\Delta\\)AIC} & \\textbf{\\(\\Delta\\)BIC} & \\textbf{LR Test \\(p\\)} & \\textbf{Pratik Kazanc} \\\\"],
            body,
        )
        self._write("tab_rq10_nested_readability.tex", latex)

    def generate_tab_rq11_nested_perf_access(self) -> None:
        path = PROJECT_ROOT / "data/analysis/D/nested_model_fit.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[df["Added_Block"].isin(["Performance", "Accessibility"])].copy()
        gain = lambda r: "Yuksek" if r["Delta_R2"] >= 0.01 else ("Sinirli" if r["Delta_R2"] >= 0.002 else "Dusuk")
        body = []
        for _, row in df.iterrows():
            body.append(
                f"{row['Model']} & {row['Added_Block']} & {row['Delta_R2']:.3f} & {row['Delta_AIC']:.1f} & "
                f"{row['Delta_BIC']:.1f} & {format_p_val(row['LR_p'])} & {gain(row)} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Performans ve erisilebilirlik bloklarinin artimli katkisi icin ic ice model karsilastirmalari",
            "tab:rq11_nested_perf_access",
            "l@{\\extracolsep{\\fill}}lrrrrr",
            ["\\textbf{Model} & \\textbf{Eklenen Blok} & \\textbf{\\(\\Delta R^2\\)} & \\textbf{\\(\\Delta\\)AIC} & \\textbf{\\(\\Delta\\)BIC} & \\textbf{LR Test \\(p\\)} & \\textbf{Pratik Kazanc} \\\\"],
            body,
        )
        self._write("tab_rq11_nested_perf_access.tex", latex)

    def _generate_confirmatory_category_table(self, filename: str, caption: str, label: str, category: str, model_id: str) -> None:
        path = PROJECT_ROOT / "data/analysis/D/confirmatory_coeffs_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        terms = [k for k, v in self.feature_categories.items() if v == category]
        df = df[(df["subset"] == "Full") & (df["model_id"] == model_id) & (df["term"].isin(terms))].copy()
        body = []
        for _, row in df.iterrows():
            body.append(
                f"{CATEGORY_LABELS[category]} (Full) & {FEATURE_LABELS.get(row['term'], row['term'])} & "
                f"{row['effect_size']:.3f} & {format_ci(row['ci_lower_95'], row['ci_upper_95'])} & "
                f"{format_p_val(row['p_raw'])} & {_practical_flag(row)} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            caption,
            label,
            "l@{\\extracolsep{\\fill}}lrlll",
            ["\\textbf{Model} & \\textbf{Yordayici} & \\textbf{Tahmin (\\(\\beta^*\\))} & \\textbf{CI95} & \\textbf{\\(p\\)} & \\textbf{Flag} \\\\"],
            body,
        )
        self._write(filename, latex)

    def generate_tab_rq11_confirmatory_performance(self) -> None:
        self._generate_confirmatory_category_table(
            "tab_rq11_confirmatory_performance.tex",
            "Performans sinyalleri icin confirmatory regresyon sonuclari",
            "tab:rq11_confirmatory_performance",
            "performance",
            "RQ4_Performance_R_FE",
        )

    def generate_tab_rq11_confirmatory_accessibility(self) -> None:
        self._generate_confirmatory_category_table(
            "tab_rq11_confirmatory_accessibility.tex",
            "Erisilebilirlik sinyalleri icin confirmatory regresyon sonuclari",
            "tab:rq11_confirmatory_accessibility",
            "accessibility",
            "RQ5_Accessibility_R_FE",
        )

    def generate_tab_rq11_supplementary_perf_access(self) -> None:
        path = PROJECT_ROOT / "data/analysis/D/supplementary_coeffs_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        wanted_terms = [k for k, v in self.feature_categories.items() if v in {"performance", "accessibility"}]
        keep_models = {
            "Supp_Performance_R_FE_Full": "Supp. Performance",
            "Supp_Accessibility_R_FE_Full": "Supp. Accessibility",
        }
        df = df[(df["subset"] == "Full") & (df["model_id"].isin(keep_models)) & (df["term"].isin(wanted_terms))].copy()
        body = []
        for _, row in df.iterrows():
            body.append(
                f"{keep_models[row['model_id']]} & {FEATURE_LABELS.get(row['term'], row['term'])} & "
                f"{row['effect_size']:.3f} & {format_ci(row['ci_lower_95'], row['ci_upper_95'])} & "
                f"{format_p_val(row['p_raw'])} & {_practical_flag(row)} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Performans ve erisilebilirlik icin supplementary regresyon sonuclari",
            "tab:rq11_supplementary_perf_access",
            "l@{\\extracolsep{\\fill}}lrlll",
            ["\\textbf{Model} & \\textbf{Yordayici} & \\textbf{Tahmin (\\(\\beta^*\\))} & \\textbf{CI95} & \\textbf{\\(p\\)} & \\textbf{Flag} \\\\"],
            body,
        )
        self._write("tab_rq11_supplementary_perf_access.tex", latex)

    def generate_tab_rq12_heterogeneity_summary(self) -> None:
        path = PROJECT_ROOT / "data/analysis/E/replication_grid.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[df["subset"] == "Full"].copy()
        schema_terms = [t for t in self.feature_categories if t in set(df["term"])]
        df = df[df["term"].isin(schema_terms)].copy()
        body = []
        for _, row in df.iterrows():
            yorum = "Tutarlı" if row["all_sign_agreement"] and row["significant_agreement"] else "Kismi"
            body.append(
                f"{FEATURE_LABELS.get(row['term'], row['term'])} & "
                f"{'Evet' if row['all_sign_agreement'] else 'Hayir'} & "
                f"{'Evet' if row['significant_agreement'] else 'Hayir'} & "
                f"{'Evet' if row['ci_overlap'] else 'Hayir'} & "
                f"{row['median_coef']:.3f} & {row['mad_coef']:.3f} & {yorum} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Motorlar arasi heterojenlik icin ozet tutarlilik olcutleri",
            "tab:rq12_heterogeneity_summary",
            "l@{\\extracolsep{\\fill}}cccccc",
            ["\\textbf{Yordayici} & \\textbf{Isaret Uyumu} & \\textbf{Anlamlilik Uyumu} & \\textbf{CI Cakismasi} & \\textbf{Medyan Katsayi} & \\textbf{MAD} & \\textbf{Yorum} \\\\"],
            body,
        )
        self._write("tab_rq12_heterogeneity_summary.tex", latex)

    def generate_tab_rq13_robustness_main(self) -> None:
        path = PROJECT_ROOT / "data/analysis/G/robustness_coeffs_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        schema_terms = [t for t in self.feature_categories if t in set(df["term"])]
        body = []
        for term in schema_terms:
            sub = df[df["term"] == term]
            base = sub[sub["model_id"] == "RQ8_Robustness_Baseline_R"]
            win = sub[sub["model_id"] == "RQ8_Robustness_Winsorized_R"]
            cluster = sub[sub["model_id"] == "RQ8_Robustness_2WayCluster_R"]
            if base.empty or win.empty or cluster.empty:
                continue
            body.append(
                f"{FEATURE_LABELS.get(term, term)} & {_fmt_effect_with_ci(base.iloc[0])} & "
                f"{_fmt_effect_with_ci(win.iloc[0])} & {_fmt_effect_with_ci(cluster.iloc[0])} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Saglamlik kontrolleri: winsorizasyon ve alternatif hata yapilari",
            "tab:rq13_robustness_main",
            "l@{\\extracolsep{\\fill}}lll",
            ["\\textbf{Yordayici} & \\textbf{Temel \\(\\beta^*\\) (CI95)} & \\textbf{Winsorized} & \\textbf{Iki-Yonlu Kumeli} \\\\"],
            body,
        )
        self._write("tab_rq13_robustness_main.tex", latex)

    def generate_tab_rq13_subdataset_sensitivity(self) -> None:
        path = PROJECT_ROOT / "data/analysis/G/robustness_coeffs_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        schema_terms = [t for t in self.feature_categories if t in set(df["term"])]
        body = []
        for term in schema_terms:
            sub = df[df["term"] == term]
            base = sub[sub["model_id"] == "RQ8_Robustness_Baseline_R"]
            nosource = sub[sub["model_id"] == "RQ8_Robustness_NoSourceDomain_R"]
            if base.empty or nosource.empty:
                continue
            b = base.iloc[0]
            n = nosource.iloc[0]
            sign = "Ayni" if np.sign(b["effect_size"]) == np.sign(n["effect_size"]) else "Degisti"
            pi = "Ayni" if bool(b.get("practical_flag", False)) == bool(n.get("practical_flag", False)) else "Degisti"
            body.append(
                f"{FEATURE_LABELS.get(term, term)} & {_fmt_effect_with_ci(b)} & {_fmt_effect_with_ci(n)} & {sign} & {pi} \\\\"
            )
        latex = self._rows_to_table(
            "table*",
            "Alt veri kumesi duyarliligi: Full ve NoSource karsilastirmasi",
            "tab:rq13_subdataset_sensitivity",
            "l@{\\extracolsep{\\fill}}l l c c",
            ["\\textbf{Yordayici} & \\textbf{Temel \\(\\beta^*\\) [CI95]} & \\textbf{NoSource \\(\\beta^*\\) [CI95]} & \\textbf{Isaret} & \\textbf{PI} \\\\"],
            body,
        )
        self._write("tab_rq13_subdataset_sensitivity.tex", latex)

    def generate_tab_rq13_ablation_ndcg(self) -> None:
        path = PROJECT_ROOT / "data/analysis/H/ablation_predictive.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        df = df[~df["set_name"].astype(str).str.contains("Full Model", case=False, na=False)].copy()
        body = []
        subset_labels = {
            "Full": "Full",
            "NoSource": "NoSource",
            "google": "Google",
            "brave": "Brave",
            "mojeek": "Mojeek",
        }
        for _, row in df.iterrows():
            block = row["set_name"].replace("- ", "").replace("-", "").strip()
            delta = f"{row['effect_size']:.3f} [{row['ci_lower_95']:.3f}, {row['ci_upper_95']:.3f}]"
            family = f"{row['model_family']} / {subset_labels.get(row['subset'], row['subset'])}"
            body.append(f"{block} & {row['ndcg_mean']:.3f} & {delta} & {family} \\\\")
        latex = self._rows_to_table(
            "table*",
            "Blok-ablasyon sonuclari: tahminsel basaridaki degisim",
            "tab:rq13_ablation_ndcg",
            "l@{\\extracolsep{\\fill}}ccc",
            ["\\textbf{Cikarilan Blok} & \\textbf{Ortalama NDCG@10} & \\textbf{\\(\\Delta\\)NDCG [95\\% CI]} & \\textbf{Model Ailesi} \\\\"],
            body,
        )
        self._write("tab_rq13_ablation_ndcg.tex", latex)

    def generate_tab_rq13_ablation_stability(self) -> None:
        path = PROJECT_ROOT / "data/analysis/H/ablation_stability_r.csv"
        if not path.exists():
            return
        df = pd.read_csv(path)
        body = []
        for _, row in df.iterrows():
            unstable = "Yok" if pd.isna(row["unstable_vars"]) or str(row["unstable_vars"]).strip() == "" else str(row["unstable_vars"]).replace(";", ", ")
            body.append(f"{tex_escape(str(row['comparison']))} & {row['avg_pct_change']:.1f}\\% & {tex_escape(unstable)} \\\\")
        latex = self._rows_to_table(
            "table*",
            "Blok karsilastirmalarinda katsayi kararliligi",
            "tab:rq13_ablation_stability",
            "lll",
            ["\\textbf{Karsilastirma} & \\textbf{Ortalama \\% Degisim} & \\textbf{Kararsiz Degiskenler} \\\\"],
            body,
        )
        self._write("tab_rq13_ablation_stability.tex", latex)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="data/analysis/analysis_results.json")
    parser.add_argument("--dataset_path", default="data/processed/combined.csv")
    parser.add_argument("--output_dir", default="data/reports/thesis/tables")
    args = parser.parse_args()

    gen = ThesisTableGenerator(args.results_path, args.dataset_path, args.output_dir)
    gen.generate_all_thesis_tables()


if __name__ == "__main__":
    main()
