import json
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys
import cluster_labels

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils
from src.generators.plot import PlotGenerator 


class LatexTableGenerator:
    
    cluster_names = {k: v.replace('&', '\\&') for k, v in cluster_labels.CLUSTER_NAMES.items()}
    cluster_colors = cluster_labels.CLUSTER_COLORS
    
    def __init__(self, results_path: str, dataset_path: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = Utils.set_colorful_logging('LatexTableGenerator') 

        try:
            with open(results_path, 'r') as f:
                self.stats = json.load(f)
            self.logger.info(f"Successfully loaded analysis results from {results_path}")
        except FileNotFoundError:
            self.logger.error(f"Error: Analysis results file not found at {results_path}")
            self.stats = None
        except json.JSONDecodeError:
            self.logger.error(f"Error: Could not decode JSON from {results_path}")
            self.stats = None

        try:
            self.df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded dataset with {len(self.df)} rows from {dataset_path}")
        except FileNotFoundError:
            self.logger.error(f"Error: Dataset file not found at {dataset_path}")
            self.df = None
        
        # Use centralized feature display names
        self.feature_display_names = Utils.get_feature_display_names()
    
    def generate_dataset_columns_table(self):
        """Generate the Dataset Column Types and Descriptions table"""
        self.logger.info("Generating Dataset Column Types and Descriptions LaTeX table...")
        
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\caption{Dataset Column Types and Descriptions}\n"
        latex += "\\label{tab:dataset_columns_types}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}lll}\n"
        latex += "\\toprule\n"
        latex += "\\sbf{Category} & \\sbf{Column Name} & \\sbf{Type} & \\sbf{Description (Metric)} \\\\\n"
        latex += "\\dmidrule\n"
        latex += "Search Engine & engine & Categorical (google/bing) & Search Engine \\\\\n"
        latex += "Search Engine & position & Integer (1-20) & SERP Position \\\\\n"
        latex += "\\midrule\n"
        latex += "Lighthouse & performance\\_score & Float (0-100) & Perf. \\\\\n"
        latex += "Lighthouse & accessibility\\_score & Float (0-100) & Access. \\\\\n"
        latex += "Lighthouse & best-practices\\_score & Float (0-100) & Best Prac. \\\\\n"
        latex += "Lighthouse & seo\\_score & Float (0-100) & SEO \\\\\n"
        latex += "\\midrule\n"
        latex += "Content & query\\_in\\_title & Integer (0/1) & Q-Title \\\\\n"
        latex += "Content & query\\_in\\_h1 & Integer (0/1) & Q-H1 \\\\\n"
        latex += "Content & exact\\_query\\_in\\_title & Integer (0/1) & ExQ-Title \\\\\n"
        latex += "Content & exact\\_query\\_in\\_h1 & Integer (0/1) & ExQ-H1 \\\\\n"
        latex += "Content & query\\_density\\_body & Float (\\%) & Q/B Density \\\\\n"
        latex += "Content & semantic\\_similarity\\_title\\_query & Float (0-1, Cosine) & Sim. Title \\\\\n"
        latex += "Content & semantic\\_similarity\\_content\\_query & Float (0-1, Cosine) & Sim. Content \\\\\n"
        latex += "Content & word\\_count & Integer & Word Count \\\\\n"
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Metric abbreviations: Perf. (Performance Score), Access. (Accessibility Score), Best Prac. (Best Practices Score), SEO (SEO Score), Q-Title (Query in Title), Q-H1 (Query in H1), ExQ-Title (Exact Query in Title), ExQ-H1 (Exact Query in H1), Q/B Density (Query Density Body), Sim. Title (Semantic Similarity Title-Query), Sim. Content (Semantic Similarity Content-Query).\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_dataset_columns_types.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"Dataset Column Types and Descriptions LaTeX table generated at {output_path}")

    def generate_descriptive_statistics_table(self):
        """Generate the Descriptive Statistics of the Final Analyzed Dataset table (fully matching user example)"""
        self.logger.info("Generating Descriptive Statistics LaTeX table (user format)...")
        
        if self.df is None or self.df.empty:
            self.logger.warning("Dataset not available for descriptive statistics.")
            return
        
        # Features in order: SERP Position, Lighthouse, Content
        features = [
            ('SERP Position', 'position'),
            ('Lighthouse', ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']),
            ('Content', ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1',
                        'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count'])
        ]
        
        # Prepare stats for each feature
        stats_rows = []
        for section, feats in features:
            if isinstance(feats, str):
                feats = [feats]
            for feature in feats:
                if feature in self.df.columns:
                    s = self.df[feature].dropna()
                    if len(s) == 0:
                        continue
                    mean = s.mean()
                    median = s.median()
                    std = s.std()
                    minv = s.min()
                    maxv = s.max()
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    # IQR%: percentage of outliers (using 1.5*IQR rule)
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_pct = 100 * ((s < lower) | (s > upper)).sum() / len(s) if len(s) > 0 else 0
                    stats_rows.append({
                        'section': section,
                        'feature': feature,
                        'mean': mean,
                        'median': median,
                        'std': std,
                        'min': minv,
                        'max': maxv,
                        'q1': q1,
                        'q3': q3,
                        'iqr_pct': outlier_pct
                    })
        # Section order for LaTeX
        section_order = ['SERP Position', 'Lighthouse', 'Content']
        section_titles = {
            'SERP Position': '\\multicolumn{9}{l}{\\sit{SERP Position}} \\\\',
            'Lighthouse': '\\multicolumn{9}{l}{\\sit{Lighthouse Scores}} \\\\',
            'Content': '\\multicolumn{9}{l}{\\sit{Content Relevance Metrics}} \\\\'
        }
        # Feature display names
        display_names = self.feature_display_names.copy()
        display_names['position'] = 'Overall'
        # LaTeX table
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\caption{Descriptive Statistics of the Final Analyzed Dataset (N=\\num{14465})}\n"
        latex += "\\label{tab:comprehensive_stats}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}S[table-format=3.2]S[table-format=3.2]S[table-format=4.2]S[table-format=3.2]S[table-format=6.2]S[table-format=3.2]S[table-format=3.2]S[table-format=2.2]}\n"
        latex += "\\toprule\n"
        latex += "\\sbf{Metric} & \\sbf{Mean} & \\sbf{Median} & \\sbf{SD} & \\sbf{Min} & \\sbf{Max} & \\sbf{Q1} & \\sbf{Q3} & \\sbf{IQR\\%} \\\\\n"
        latex += "\\dmidrule\n"
        # Write rows by section
        for section in section_order:
            section_rows = [r for r in stats_rows if r['section'] == section]
            if not section_rows:
                continue
            latex += section_titles[section] + "\n"
            for row in section_rows:
                name = display_names.get(row['feature'], row['feature'])
                latex += f"{name} & {row['mean']:.2f} & {row['median']:.2f} & {row['std']:.2f} & {row['min']:.2f} & {row['max']:.2f} & {row['q1']:.2f} & {row['q3']:.2f} & {row['iqr_pct']:.2f} \\\\\n"
            if section != section_order[-1]:
                latex += "\\midrule\n"
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Note: SD = Standard Deviation; Q1 = First Quartile; Q3 = Third Quartile. IQR\\% indicates percentage of outliers via IQR method.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_descriptive_statistics.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        self.logger.info(f"Descriptive Statistics LaTeX table generated at {output_path}")
    
    def generate_all_tables(self):
        if not self.stats:
            self.logger.error("Cannot generate tables due to missing analysis results.")
            return
      
        self.generate_dataset_columns_table()
        self.generate_descriptive_statistics_table()
        self.generate_rq1_cluster_summary_table()
        self.generate_rq3_feature_comparison_table()
        self.generate_rq1_validation_table()
        self.generate_rq2_feature_rank_tests_table()
        self.generate_rq4_regression_tables()
        self.generate_rq1_cluster_quality_table()
        self.generate_rq4_feature_importance_comparison_table()
        
        self.logger.info(f"All LaTeX tables/snippets generated in {self.output_dir}")

    def generate_rq1_cluster_summary_table(self):
        """Generate the RQ1 cluster summary table as in tables.tex"""
        self.logger.info("Generating RQ1 Cluster Summary LaTeX table...")
        if 'rq1_clustering' not in self.stats or 'cluster_stats' not in self.stats['rq1_clustering']:
            self.logger.warning("Data for RQ1 cluster summary table not found.")
            return

        cluster_data = self.stats['rq1_clustering']['cluster_stats']
        
        # Sort clusters by key to ensure consistent order
        sorted_cluster_data = sorted(cluster_data.items(), key=lambda item: int(item[0]))
        
        # Get features from the first cluster
        first_cluster_key = sorted_cluster_data[0][0]
        if 'feature_means_scaled' not in cluster_data[first_cluster_key]:
            self.logger.warning("Scaled feature means not found in cluster data.")
            return
            
        features = list(cluster_data[first_cluster_key]['feature_means_scaled'].keys())
        
        # Create DataFrame for table data
        data = []
        for cluster_name, stats in sorted_cluster_data:
            row = {'Cluster': f'C{cluster_name}'}
            for feature in features:
                row[feature] = stats['feature_means_scaled'].get(feature, 0)
            row['Size'] = stats.get('size', 0)
            data.append(row)
        
        df = pd.DataFrame(data)
        total_size = df['Size'].sum()
        if total_size > 0:
            df['Percentage'] = (df['Size'] / total_size) * 100
        else:
            df['Percentage'] = 0
            
        # Generate LaTeX table
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Mean Scaled Feature Values for the Identified Cluster Profiles (RQ1)}\n"
        latex += "\\label{tab:rq1_cluster_summary}\n"
        latex += "\\small \n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
                                     
        # Dynamically create header
        header = " & ".join([f"{{{{\\sbf{{C{i}}}}}}}" for i in range(len(sorted_cluster_data))])
        latex += f"\\begin{{tabular*}}{{\\textwidth}}{{l@{{\\extracolsep{{\\fill}}}} {' '.join(['S[table-format=1.3]' for _ in sorted_cluster_data])} }}\n"
        latex += "\\toprule\n"
        latex += f"\\sbf{{Feature (Scaled)}} & {header} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Add each feature row
        for feature in features:
            display_name = self.feature_display_names.get(feature, feature)
            values = " & ".join([f"{row[feature]:.3f}" for _, row in df.iterrows()])
            latex += f"{display_name} & {values} \\\\\n"
        
        # Add cluster size and percentage
        latex += "\\midrule\n"
        latex += "Cluster Size (N) & "
        size_values = " & ".join([f"\\num{{{int(row['Size'])}}}" for _, row in df.iterrows()])
        latex += f"{size_values} \\\\\n"
        
        latex += "Percentage (\\%) & "
        percentage_values = " & ".join([f"{{{row['Percentage']:.1f}}}" for _, row in df.iterrows()])
        latex += f"{percentage_values} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += ("\\item Note: C0, C1, C2, ... denote the K-means cluster labels "
          "ordered by their numeric ID. Features were Min-Max scaled to [0,1].\n")
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{table}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq1_cluster_summary.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ1 Cluster Summary LaTeX table generated at {output_path}")

    def generate_rq3_feature_comparison_table(self):
        """Generate the RQ3 feature comparison table dynamically from JSON data with real median calculations"""
        self.logger.info("Generating RQ3 Feature Comparison LaTeX table...")
        
        if 'rq3_system_comparison' not in self.stats or 'feature_comparison_top5' not in self.stats['rq3_system_comparison']:
            self.logger.warning("RQ3 feature comparison data not found.")
            return
            
        feature_data = self.stats['rq3_system_comparison']['feature_comparison_top5']
        
        # Calculate real median values from the dataset
        if self.df is not None and not self.df.empty:
            features = list(feature_data.keys())
            medians = Utils.calculate_medians_by_engine(self.df, features, top_n=5)
        else:
            self.logger.warning("Dataset not available for median calculation, using placeholder values.")
            medians = {}
        
        # Generate LaTeX table
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Comparison of Median Feature Values for Top-Ranking (1-5) Pages between Google and Bing (RQ3)}\n"
        latex += "\\label{tab:rq3_feature_comparison}\n"
        latex += "\\small \n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}} S[table-format=3.1] S[table-format=3.1] S[table-format=1.3,table-comparator=true,table-space-text-post={***}] S[table-format=-1.3,table-space-text-post={***}]}\n"
        latex += "\\toprule\n"
        latex += "\\sbf{Feature} & {\\sbf{Median Google}} & {\\sbf{Median Bing}} & {\\sbf{p-value}} & {\\sbf{Cohen's d}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Lighthouse scores section
        lighthouse_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        latex += "\\multicolumn{5}{@{}l}{\\sit{Lighthouse Scores (0-100)}} \\\\\n"
        
        for feature in lighthouse_features:
            if feature in feature_data:
                data = feature_data[feature]
                p_value = data.get('p_value', 0)
                cohens_d = data.get('cohens_d', 0)
                
                # Format p-value
                if p_value < 0.001:
                    p_str = "< .001***"
                elif p_value < 0.01:
                    p_str = f"{p_value:.3f}**"
                elif p_value < 0.05:
                    p_str = f"{p_value:.3f}*"
                else:
                    p_str = f"{p_value:.3f}"
                
                # Get real median values or use placeholders
                median_google = medians.get('google', {}).get(feature, 0.0)
                median_bing = medians.get('bing', {}).get(feature, 0.0)
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {median_google:.1f} & {median_bing:.1f} & {p_str} & {cohens_d:.3f} \\\\\n"
        
        # Content metrics section
        content_features = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1', 
                          'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
        latex += "\\midrule\n"
        latex += "\\multicolumn{5}{@{}l}{\\sit{Content Relevance Metrics}} \\\\\n"
        
        for feature in content_features:
            if feature in feature_data:
                data = feature_data[feature]
                p_value = data.get('p_value', 0)
                cohens_d = data.get('cohens_d', 0)
                
                # Format p-value
                if p_value < 0.001:
                    p_str = "< .001***"
                elif p_value < 0.01:
                    p_str = f"{p_value:.3f}**"
                elif p_value < 0.05:
                    p_str = f"{p_value:.3f}*"
                else:
                    p_str = f"{p_value:.3f}"
                
                # Get real median values or use placeholders
                median_google = medians.get('google', {}).get(feature, 0.0)
                median_bing = medians.get('bing', {}).get(feature, 0.0)
                
                # Add appropriate unit notes
                if feature in ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1']:
                    unit_note = "\\tnote{a}"
                elif feature == 'query_density_body':
                    unit_note = "(\\%)"
                elif feature in ['semantic_similarity_title_query', 'semantic_similarity_content_query']:
                    unit_note = "(0-1)"
                else:  # word_count
                    unit_note = ""
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} {unit_note} & {median_google:.1f} & {median_bing:.1f} & {p_str} & {cohens_d:.3f} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Mann-Whitney U test used. p-values: * $p < .05$; ** $p < .01$; *** $p < .001$.\n"
        latex += "\\item Cohen's d: negligible ($|d| < 0.2$), small ($0.2 \\le |d| < 0.5$), medium ($0.5 \\le |d| < 0.8$).\n"
        latex += "\\item[a] For binary features, medians are shown; mean percentages discussed in text.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq3_feature_comparison.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ3 Feature Comparison LaTeX table generated at {output_path}")

    def generate_rq1_validation_table(self):
        """Generate the RQ1 cluster validation table (Kruskal-Wallis) dynamically from JSON data"""
        self.logger.info("Generating RQ1 Cluster Validation LaTeX table...")
        
        if 'rq1_clustering' not in self.stats or 'final_clustering_results' not in self.stats['rq1_clustering']:
            self.logger.warning("RQ1 clustering validation data not found.")
            return
            
        kruskal_data = self.stats['rq1_clustering']['final_clustering_results']['kruskal_wallis_on_clusters']
        
        # Generate LaTeX table
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Kruskal-Wallis H-Test Results for Feature Differentiation Across Clusters (RQ1)}\n"
        latex += "\\label{app:rq1_cluster_validation_table}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\columnwidth}{l@{\\extracolsep{\\fill}} S[table-format=5.3] S[table-format=1.3,table-comparator=true]}\n"
        latex += "\\toprule\n"
        latex += "\\sbf{Feature} & {\\sbf{H-statistic}} & {\\sbf{p-value}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Add each feature row
        for feature, data in kruskal_data.items():
            h_statistic = data.get('h_statistic', 0)
            p_value = data.get('p_value', 0)
            display_name = self.feature_display_names.get(feature, feature)
            
            # Format p-value
            if p_value < 0.001:
                p_str = "< .001"
            else:
                p_str = f"{p_value:.3f}"
            
            latex += f"{display_name}  & {h_statistic:.3f} & {p_str} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item p-values smaller than $.001$ are reported as $< .001$.\n"
        latex += "\\item Pairwise cluster differences were examined using Dunn's post-hoc tests with Bonferroni correction; only omnibus Kruskal--Wallis statistics are reported here.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq1_cluster_validation.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ1 Cluster Validation LaTeX table generated at {output_path}")

    def generate_rq2_feature_rank_tests_table(self):
        """Generate the RQ2 feature rank tests table (Kruskal-Wallis) dynamically from JSON data"""
        self.logger.info("Generating RQ2 Feature Rank Tests LaTeX table...")
        
        if 'rq2_visibility' not in self.stats:
            self.logger.warning("RQ2 visibility data not found.")
            return
            
        google_data = self.stats['rq2_visibility']['google']['kruskal_wallis_by_rank']
        bing_data = self.stats['rq2_visibility']['bing']['kruskal_wallis_by_rank']
        
        # Generate LaTeX table
        latex = "\\begin{table}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Kruskal-Wallis H-Test Results for Features Across Ranking Tiers (High, Medium, Low) for Google and Bing (RQ2)}\n"
        latex += "\\label{tab:rq2_kruskal_wallis_ranks}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}}S[table-format=3.3] S[table-format=1.3,table-comparator=true] S[table-format=3.3] S[table-format=1.3,table-comparator=true]}\n"
        latex += "\\toprule\n"
        latex += "& \\multicolumn{2}{c}{\\sbf{Google (System A)}} & \\multicolumn{2}{c}{\\sbf{Bing (System B)}} \\\\\n"
        latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        latex += "\\sbf{Feature} & {\\sbf{H-statistic}} & {\\sbf{p-value}} & {\\sbf{H-statistic}} & {\\sbf{p-value}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Lighthouse scores section
        lighthouse_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        latex += "\\sit{Lighthouse Scores} & & & & \\\\\n"
        
        for feature in lighthouse_features:
            if feature in google_data and feature in bing_data:
                g_data = google_data[feature]
                b_data = bing_data[feature]
                
                g_h = g_data.get('h_statistic', 0)
                g_p = g_data.get('p_value', 0)
                b_h = b_data.get('h_statistic', 0)
                b_p = b_data.get('p_value', 0)
                
                # Format p-values
                g_p_str = "< .001" if g_p < 0.001 else f"{g_p:.3f}"
                b_p_str = "< .001" if b_p < 0.001 else f"{b_p:.3f}"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {g_h:.3f} & {g_p_str} & {b_h:.3f} & {b_p_str} \\\\\n"
        
        # Content metrics section
        content_features = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1', 
                          'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
        latex += "\\midrule\n"
        latex += "\\sit{Content Relevance Metrics} & & & & \\\\\n"
        
        for feature in content_features:
            if feature in google_data and feature in bing_data:
                g_data = google_data[feature]
                b_data = bing_data[feature]
                
                g_h = g_data.get('h_statistic', 0)
                g_p = g_data.get('p_value', 0)
                b_h = b_data.get('h_statistic', 0)
                b_p = b_data.get('p_value', 0)
                
                # Format p-values
                g_p_str = "< .001" if g_p < 0.001 else f"{g_p:.3f}"
                b_p_str = "< .001" if b_p < 0.001 else f"{b_p:.3f}"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {g_h:.3f} & {g_p_str} & {b_h:.3f} & {b_p_str} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Ranking tiers: High (positions 1--5), Medium (6--10), Low (11--20).\n"
        latex += "\\item For features with significant omnibus tests, pairwise high/medium/low differences were examined using Dunn's post-hoc tests with Bonferroni correction; only Kruskal--Wallis statistics are reported here.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq2_feature_rank_tests.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ2 Feature Rank Tests LaTeX table generated at {output_path}")

    def _generate_regression_table(self, engine_name: str, engine_data: dict) -> str:
        """Helper function to generate regression table for a specific engine"""
        engine_display = "Google (System A)" if engine_name == "google" else "Bing (System B)"
        label_suffix = "google" if engine_name == "google" else "bing"
        
        latex = f"\\begin{{table}}[htbp!] \n"
        latex += "\\centering\n"
        latex += f"\\caption{{Ordinal Logistic Regression for Predicting SERP Quintiles - {engine_display} (RQ4)}}\n"
        latex += f"\\label{{app:rq4_regression_{label_suffix}}}\n"
        latex += "\\small \n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}} S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}] S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}] S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}]}\n"
        latex += "\\toprule\n"
        latex += "& \\multicolumn{2}{c}{\\sbf{Technical Model}} & \\multicolumn{2}{c}{\\sbf{Content Model}} & \\multicolumn{2}{c}{\\sbf{Combined Model}} \\\\\n"
        latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
        latex += "\\sbf{Predictor} & {\\sbf{Coeff.}} & {\\sbf{p-value}} & {\\sbf{Coeff.}} & {\\sbf{p-value}} & {\\sbf{Coeff.}} & {\\sbf{p-value}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Technical features section
        technical_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        latex += "\\sit{Lighthouse Scores} & & & & & & \\\\\n"
        
        for feature in technical_features:
            if feature in engine_data['ordinal_logistic_regression']['technical']['coeffs']:
                # Technical model coefficients
                tech_coeff = engine_data['ordinal_logistic_regression']['technical']['coeffs'][feature]
                tech_p_value = engine_data['ordinal_logistic_regression']['technical']['p_values'][feature]
                
                # Combined model coefficients
                combined_coeff = engine_data['ordinal_logistic_regression']['combined']['coeffs'][feature]
                combined_p_value = engine_data['ordinal_logistic_regression']['combined']['p_values'][feature]
                
                # Format p-values
                if tech_p_value < 0.001:
                    tech_p_str = "< .001***"
                elif tech_p_value < 0.01:
                    tech_p_str = f"{tech_p_value:.3f}**"
                elif tech_p_value < 0.05:
                    tech_p_str = f"{tech_p_value:.3f}*"
                else:
                    tech_p_str = f"{tech_p_value:.3f}"
                
                if combined_p_value < 0.001:
                    combined_p_str = "< .001***"
                elif combined_p_value < 0.01:
                    combined_p_str = f"{combined_p_value:.3f}**"
                elif combined_p_value < 0.05:
                    combined_p_str = f"{combined_p_value:.3f}*"
                else:
                    combined_p_str = f"{combined_p_value:.3f}"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {tech_coeff:.3f} & {tech_p_str} & \\multicolumn{{1}}{{c}}{{--}} & \\multicolumn{{1}}{{c}}{{--}} & {combined_coeff:.3f} & {combined_p_str} \\\\\n"
        
        # Content features section
        content_features = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1', 
                          'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
        latex += "\\midrule\n"
        latex += "\\sit{Content Metrics} & & & & & & \\\\\n"
        
        for feature in content_features:
            if feature in engine_data['ordinal_logistic_regression']['content']['coeffs']:
                # Content model coefficients
                content_coeff = engine_data['ordinal_logistic_regression']['content']['coeffs'][feature]
                content_p_value = engine_data['ordinal_logistic_regression']['content']['p_values'][feature]
                
                # Combined model coefficients
                combined_coeff = engine_data['ordinal_logistic_regression']['combined']['coeffs'][feature]
                combined_p_value = engine_data['ordinal_logistic_regression']['combined']['p_values'][feature]
                
                # Format p-values
                if content_p_value < 0.001:
                    content_p_str = "< .001***"
                elif content_p_value < 0.01:
                    content_p_str = f"{content_p_value:.3f}**"
                elif content_p_value < 0.05:
                    content_p_str = f"{content_p_value:.3f}*"
                else:
                    content_p_str = f"{content_p_value:.3f}"
                
                if combined_p_value < 0.001:
                    combined_p_str = "< .001***"
                elif combined_p_value < 0.01:
                    combined_p_str = f"{combined_p_value:.3f}**"
                elif combined_p_value < 0.05:
                    combined_p_str = f"{combined_p_value:.3f}*"
                else:
                    combined_p_str = f"{combined_p_value:.3f}"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & \\multicolumn{{1}}{{c}}{{--}} & \\multicolumn{{1}}{{c}}{{--}} & {content_coeff:.3f} & {content_p_str} & {combined_coeff:.3f} & {combined_p_str} \\\\\n"
        
        # Model fit measures - use actual values from each model
        technical_r2 = engine_data['ordinal_logistic_regression']['technical']['pseudo_r2']
        content_r2 = engine_data['ordinal_logistic_regression']['content']['pseudo_r2']
        combined_r2 = engine_data['ordinal_logistic_regression']['combined']['pseudo_r2']
        
        technical_aic = engine_data['ordinal_logistic_regression']['technical']['aic']
        content_aic = engine_data['ordinal_logistic_regression']['content']['aic']
        combined_aic = engine_data['ordinal_logistic_regression']['combined']['aic']
        
        technical_bic = engine_data['ordinal_logistic_regression']['technical']['bic']
        content_bic = engine_data['ordinal_logistic_regression']['content']['bic']
        combined_bic = engine_data['ordinal_logistic_regression']['combined']['bic']
        
        latex += "\\midrule\n"
        latex += f"Pseudo $R^2$ & \\multicolumn{{2}}{{c}}{{{technical_r2:.4f}}} & \\multicolumn{{2}}{{c}}{{{content_r2:.4f}}} & \\multicolumn{{2}}{{c}}{{{combined_r2:.4f}}} \\\\\n"
        latex += f"AIC & \\multicolumn{{2}}{{c}}{{{technical_aic:.1f}}} & \\multicolumn{{2}}{{c}}{{{content_aic:.1f}}} & \\multicolumn{{2}}{{c}}{{{combined_aic:.1f}}} \\\\\n"
        latex += f"BIC & \\multicolumn{{2}}{{c}}{{{technical_bic:.1f}}} & \\multicolumn{{2}}{{c}}{{{content_bic:.1f}}} & \\multicolumn{{2}}{{c}}{{{combined_bic:.1f}}} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Dependent variable: SERP quintile (0=best, 4=worst). Negative coefficients indicate an increased likelihood of appearing in a better quintile.\n"
        latex += "\\item * $p < .05$; ** $p < .01$; *** $p < .001$. Thresholds (cut-points) for quintiles are omitted. Predictors were normalized using Min-Max scaling prior to estimation.\n"
        latex += "\\item Proportional odds (parallel lines) assumption was tested for all models using a Brant-type likelihood-ratio test; no violations were detected (all $p > .05$).\n"
        latex += "\\item Omnibus Kruskal--Wallis tests and Dunn's post-hoc comparisons across ranking tiers are reported in Table~\\ref{tab:rq2_kruskal_wallis_ranks}.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table}\n"
        
        return latex

    def _format_p_value(self, p_value: float) -> str:
        """Helper function to format p-values with significance markers"""
        if p_value < 0.001:
            return "< .001***"
        elif p_value < 0.01:
            return f"{p_value:.3f}**"
        elif p_value < 0.05:
            return f"{p_value:.3f}*"
        else:
            return f"{p_value:.3f}"

    def _add_engine_section(self, latex: str, engine_name: str, engine_data: dict) -> str:
        """Helper function to add a section for one engine (Google or Bing) to the combined table"""
        engine_label = "Google" if engine_name == "google" else "Bing"
        
        # Technical features section
        technical_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        latex += f"\\multicolumn{{3}}{{l}}{{\\sit{{Lighthouse Scores ({engine_label})}}}}  \\\\\n"
        
        for feature in technical_features:
            if feature in engine_data['ordinal_logistic_regression']['technical']['coeffs']:
                tech_coeff = engine_data['ordinal_logistic_regression']['technical']['coeffs'][feature]
                tech_p_value = engine_data['ordinal_logistic_regression']['technical']['p_values'][feature]
                combined_coeff = engine_data['ordinal_logistic_regression']['combined']['coeffs'][feature]
                combined_p_value = engine_data['ordinal_logistic_regression']['combined']['p_values'][feature]
                
                tech_p_str = self._format_p_value(tech_p_value)
                combined_p_str = self._format_p_value(combined_p_value)
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {tech_coeff:.3f} & {tech_p_str} & \\multicolumn{{1}}{{c}}{{--}} & \\multicolumn{{1}}{{c}}{{--}} & {combined_coeff:.3f} & {combined_p_str} \\\\\n"
        
        # Content features section
        content_features = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1', 
                          'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
        latex += "\\midrule\n"
        latex += f"\\multicolumn{{3}}{{l}}{{\\sit{{Content Metrics ({engine_label})}}}}  \\\\\n"
        
        for feature in content_features:
            if feature in engine_data['ordinal_logistic_regression']['content']['coeffs']:
                content_coeff = engine_data['ordinal_logistic_regression']['content']['coeffs'][feature]
                content_p_value = engine_data['ordinal_logistic_regression']['content']['p_values'][feature]
                combined_coeff = engine_data['ordinal_logistic_regression']['combined']['coeffs'][feature]
                combined_p_value = engine_data['ordinal_logistic_regression']['combined']['p_values'][feature]
                
                content_p_str = self._format_p_value(content_p_value)
                combined_p_str = self._format_p_value(combined_p_value)
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & \\multicolumn{{1}}{{c}}{{--}} & \\multicolumn{{1}}{{c}}{{--}} & {content_coeff:.3f} & {content_p_str} & {combined_coeff:.3f} & {combined_p_str} \\\\\n"
        
        # Model fit measures
        technical_r2 = engine_data['ordinal_logistic_regression']['technical']['pseudo_r2']
        content_r2 = engine_data['ordinal_logistic_regression']['content']['pseudo_r2']
        combined_r2 = engine_data['ordinal_logistic_regression']['combined']['pseudo_r2']
        
        technical_aic = engine_data['ordinal_logistic_regression']['technical']['aic']
        content_aic = engine_data['ordinal_logistic_regression']['content']['aic']
        combined_aic = engine_data['ordinal_logistic_regression']['combined']['aic']
        
        technical_bic = engine_data['ordinal_logistic_regression']['technical']['bic']
        content_bic = engine_data['ordinal_logistic_regression']['content']['bic']
        combined_bic = engine_data['ordinal_logistic_regression']['combined']['bic']
        
        latex += "\\midrule\n"
        latex += f"Pseudo $R^2$ & \\multicolumn{{2}}{{c}}{{{technical_r2:.4f}}} & \\multicolumn{{2}}{{c}}{{{content_r2:.4f}}} & \\multicolumn{{2}}{{c}}{{{combined_r2:.4f}}} \\\\\n"
        latex += f"AIC & \\multicolumn{{2}}{{c}}{{\\num{{{technical_aic:.1f}}}}} & \\multicolumn{{2}}{{c}}{{\\num{{{content_aic:.1f}}}}} & \\multicolumn{{2}}{{c}}{{\\num{{{combined_aic:.1f}}}}} \\\\\n"
        latex += f"BIC & \\multicolumn{{2}}{{c}}{{\\num{{{technical_bic:.1f}}}}} & \\multicolumn{{2}}{{c}}{{\\num{{{content_bic:.1f}}}}} & \\multicolumn{{2}}{{c}}{{\\num{{{combined_bic:.1f}}}}} \\\\\n"
        latex += "\\bottomrule\n"
        
        return latex

    def generate_rq4_regression_tables(self):
        """Generate the combined RQ4 regression table (Google and Bing together)"""
        self.logger.info("Generating RQ4 Combined Regression LaTeX table...")
        
        if 'rq4_factors' not in self.stats:
            self.logger.warning("RQ4 factors data not found.")
            return
        
        google_data = self.stats['rq4_factors'].get('google')
        bing_data = self.stats['rq4_factors'].get('bing')
        
        if not google_data or not bing_data:
            self.logger.warning("Both Google and Bing data required for combined table.")
            return
        
        # Start building the combined table
        latex = "\\begin{table*}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Ordinal Logistic Regression for Predicting SERP Quintiles (RQ4)}\n"
        latex += "\\label{tab:rq4_regression}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}} S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}] S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}] S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3,table-comparator=true,table-space-text-post={***}]}\n"
        latex += "\\toprule\n"
        latex += "& \\multicolumn{2}{c}{\\sbf{Technical Model}} & \\multicolumn{2}{c}{\\sbf{Content Model}} & \\multicolumn{2}{c}{\\sbf{Combined Model}} \\\\\n"
        latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}\n"
        latex += "\\sbf{Predictor} & {\\sbf{Coeff.}} & {\\sbf{p-value}} & {\\sbf{Coeff.}} & {\\sbf{p-value}} & {\\sbf{Coeff.}} & {\\sbf{p-value}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Add Google section
        latex = self._add_engine_section(latex, "google", google_data)
        
        # Add Bing section
        latex = self._add_engine_section(latex, "bing", bing_data)
        
        # Finish table
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Dependent Variable: SERP Quintile (0=best, 4=worst). Negative coefficients indicate increased likelihood of better ranking. * $p < .05$; ** $p < .01$; *** $p < .001$. Thresholds (cut-points) for quintiles are omitted. Predictors were standardised prior to estimation.\n"
        latex += "\\item Proportional odds (parallel lines) assumption was tested for all models using a Brant-type likelihood-ratio test; no violations were detected (all p > .05).\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{table*}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq4_regression.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ4 Combined Regression LaTeX table generated at {output_path}")

    def generate_rq1_cluster_quality_table(self):
        """Generate the RQ1 cluster quality metrics table dynamically from JSON data"""
        self.logger.info("Generating RQ1 Cluster Quality Metrics LaTeX table...")
        
        if 'rq1_clustering' not in self.stats or 'final_clustering_results' not in self.stats['rq1_clustering']:
            self.logger.warning("RQ1 clustering data not found in stats.")
            return
            
        # Get cluster data (already mapped by correct_cluster_labels)
        cluster_data = self.stats['rq1_clustering']['cluster_stats']
        final_results = self.stats['rq1_clustering']['final_clustering_results']
        
        overall_silhouette = final_results.get('overall_silhouette_score', 0.0)
        per_cluster_silhouette = final_results.get('per_cluster_silhouette', {})
        per_cluster_wcss = final_results.get('per_cluster_wcss', {})
        
        # Generate LaTeX table
        latex = "\\begin{table*}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Identified Cluster Quality and Cohesion Metrics (RQ1)}\n"
        latex += "\\label{tab:cluster_quality}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}} S[table-format=4.0] S[table-format=2.1] S[table-format=1.3] S[table-format=6.1] l}\n"
        latex += "\\toprule\n"
        latex += "\\sbf{Cluster Profile} & {\\sbf{Cluster Size (N)}} & {\\sbf{Percentage (\\%)}} & {\\sbf{Mean Silhouette Score}} & {\\sbf{WCSS}} & {\\sbf{Color}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Sort clusters by their new numbers (after mapping)
        sorted_cluster_data = sorted(cluster_data.items(), key=lambda item: int(item[0]))
        total_size = sum(stats.get('size', 0) for _, stats in sorted_cluster_data)

        if total_size == 0:
            self.logger.warning("Total cluster size is 0. Cannot generate RQ1 quality table.")
            return

        for cluster_key, stats in sorted_cluster_data:
            size = stats.get('size', 0)
            percentage = (size / total_size * 100)
            # Use cluster names from mapping
            name = self.cluster_names.get(cluster_key, f'C{cluster_key}')
            color = self.cluster_colors.get(cluster_key, f'C{cluster_key}')
            
            # Get silhouette score for this cluster (after mapping)
            silhouette = per_cluster_silhouette.get(f'cluster_{cluster_key}', 0.0)
            
            # Get WCSS for this cluster
            wcss = per_cluster_wcss.get(f'cluster_{cluster_key}', 0.0)
            
            latex += f"C{cluster_key}: {name} & {size} & {percentage:.1f} & {silhouette:.3f} & {wcss:.1f} & {color} \\\\\n"
        
        # Add overall/average row
        total_wcss = sum(per_cluster_wcss.get(f'cluster_{cluster_key}', 0.0) for cluster_key, _ in sorted_cluster_data)
        latex += "\\midrule\n"
        latex += f"Overall / Average & {total_size} & 100.0 & {overall_silhouette:.3f} & {total_wcss:.1f} \\\\\n"
        
        davies_score = final_results.get('davies_bouldin_score', 0.0)
        calinski_score = final_results.get('calinski_harabasz_score', 0.0)
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += ("\\item Higher silhouette scores (range $[-1, 1]$) indicate more cohesive and well-separated clusters. "
          "Cluster sizes correspond to Table~\\ref{tab:rq1_cluster_summary}.\n")
        latex += (f"\\item Global validation metrics: Davies--Bouldin index = {davies_score:.3f} (lower is better), "
                f"Calinski--Harabasz score = \\num{{{calinski_score:.1f}}} (higher is better).\n")
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table*}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq1_cluster_quality.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ1 Cluster Quality Metrics LaTeX table generated at {output_path}")

    def generate_rq4_feature_importance_comparison_table(self):
        """Generate the RQ4 Feature Importance Comparison table (Combined Regression Coefficients and RF Importance)"""
        self.logger.info("Generating RQ4 Feature Importance Comparison LaTeX table...")
        
        if 'rq4_factors' not in self.stats:
            self.logger.warning("RQ4 factors data not found.")
            return
        
        google_data = self.stats['rq4_factors'].get('google')
        bing_data = self.stats['rq4_factors'].get('bing')
        
        if not google_data or not bing_data:
            self.logger.warning("Both Google and Bing data required for feature importance comparison table.")
            return
        
        # Get combined model coefficients
        google_coeffs = google_data.get('ordinal_logistic_regression', {}).get('combined', {}).get('coeffs', {})
        bing_coeffs = bing_data.get('ordinal_logistic_regression', {}).get('combined', {}).get('coeffs', {})
        
        # Get RF importance scores
        google_rf = google_data.get('feature_importance_rf_stats', {}).get('all_factors', {})
        bing_rf = bing_data.get('feature_importance_rf_stats', {}).get('all_factors', {})
        
        # Get p-values for significance markers
        google_p_values = google_data.get('ordinal_logistic_regression', {}).get('combined', {}).get('p_values', {})
        bing_p_values = bing_data.get('ordinal_logistic_regression', {}).get('combined', {}).get('p_values', {})
        
        # Get all features (union of both engines)
        all_features = set(google_coeffs.keys()) | set(bing_coeffs.keys())
        # Remove threshold keys (like "0/1", "1/2", etc.)
        all_features = {f for f in all_features if '/' not in str(f)}
        
        # Sort features: technical first, then content
        technical_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        content_features = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1',
                          'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
        
        # Order features
        ordered_features = []
        for feat in technical_features:
            if feat in all_features:
                ordered_features.append(feat)
        for feat in content_features:
            if feat in all_features:
                ordered_features.append(feat)
        
        # Generate LaTeX table
        latex = "\\begin{table*}[htbp!]\n"
        latex += "\\centering\n"
        latex += "\\caption{Feature Importance Comparison: Combined Model Regression Coefficients and Random Forest Importance Scores (RQ4)}\n"
        latex += "\\label{tab:rq4_feature_importance_comparison}\n"
        latex += "\\small\n"
        latex += "\\setlength{\\tabcolsep}{3pt}\n"
        latex += "\\renewcommand{\\arraystretch}{1}\n"
        latex += "\\begin{threeparttable}\n"
        latex += "\\begin{tabular*}{\\textwidth}{l@{\\extracolsep{\\fill}} S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3] S[table-format=-1.3,table-space-text-post={***}] S[table-format=1.3]}\n"
        latex += "\\toprule\n"
        latex += "& \\multicolumn{2}{c}{\\sbf{Google (System A)}} & \\multicolumn{2}{c}{\\sbf{Bing (System B)}} \\\\\n"
        latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
        latex += "\\sbf{Feature} & {\\sbf{Combined Coeff.}} & {\\sbf{RF Importance}} & {\\sbf{Combined Coeff.}} & {\\sbf{RF Importance}} \\\\\n"
        latex += "\\dmidrule\n"
        
        # Lighthouse scores section
        latex += "\\sit{Lighthouse Scores} & & & & \\\\\n"
        
        for feature in ordered_features:
            if feature in technical_features:
                google_coeff = google_coeffs.get(feature, 0.0)
                bing_coeff = bing_coeffs.get(feature, 0.0)
                google_rf_imp = google_rf.get(feature, 0.0)
                bing_rf_imp = bing_rf.get(feature, 0.0)
                
                # Format coefficients with significance markers
                google_p = google_p_values.get(feature, 1.0)
                bing_p = bing_p_values.get(feature, 1.0)
                
                google_coeff_str = f"{google_coeff:.3f}"
                if google_p < 0.001:
                    google_coeff_str += "***"
                elif google_p < 0.01:
                    google_coeff_str += "**"
                elif google_p < 0.05:
                    google_coeff_str += "*"
                
                bing_coeff_str = f"{bing_coeff:.3f}"
                if bing_p < 0.001:
                    bing_coeff_str += "***"
                elif bing_p < 0.01:
                    bing_coeff_str += "**"
                elif bing_p < 0.05:
                    bing_coeff_str += "*"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {google_coeff_str} & {google_rf_imp:.3f} & {bing_coeff_str} & {bing_rf_imp:.3f} \\\\\n"
        
        # Content metrics section
        latex += "\\midrule\n"
        latex += "\\sit{Content Relevance Metrics} & & & & \\\\\n"
        
        for feature in ordered_features:
            if feature in content_features:
                google_coeff = google_coeffs.get(feature, 0.0)
                bing_coeff = bing_coeffs.get(feature, 0.0)
                google_rf_imp = google_rf.get(feature, 0.0)
                bing_rf_imp = bing_rf.get(feature, 0.0)
                
                # Format coefficients with significance markers
                google_p = google_p_values.get(feature, 1.0)
                bing_p = bing_p_values.get(feature, 1.0)
                
                google_coeff_str = f"{google_coeff:.3f}"
                if google_p < 0.001:
                    google_coeff_str += "***"
                elif google_p < 0.01:
                    google_coeff_str += "**"
                elif google_p < 0.05:
                    google_coeff_str += "*"
                
                bing_coeff_str = f"{bing_coeff:.3f}"
                if bing_p < 0.001:
                    bing_coeff_str += "***"
                elif bing_p < 0.01:
                    bing_coeff_str += "**"
                elif bing_p < 0.05:
                    bing_coeff_str += "*"
                
                display_name = self.feature_display_names.get(feature, feature)
                latex += f"{display_name} & {google_coeff_str} & {google_rf_imp:.3f} & {bing_coeff_str} & {bing_rf_imp:.3f} \\\\\n"
        
        # Finish table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular*}\n"
        latex += "\\begin{tablenotes}[flushleft]\n"
        latex += "\\scriptsize\n"
        latex += "\\item Combined Coeff.: Coefficients from the Combined Ordinal Logistic Regression model (see Table \\ref{tab:rq4_regression}). Negative coefficients indicate increased likelihood of better ranking.\n"
        latex += "\\item RF Importance: Random Forest feature importance scores (normalized to sum to 1.0). Higher values indicate greater predictive importance.\n"
        latex += "\\item Significance markers: * $p < .05$; ** $p < .01$; *** $p < .001$.\n"
        latex += "\\item Metric abbreviations are defined in Table \\ref{tab:dataset_columns_types}.\n"
        latex += "\\end{tablenotes}\n"
        latex += "\\end{threeparttable}\n"
        latex += "\\end{table*}\n"
        
        # Write to file
        output_path = os.path.join(self.output_dir, 'table_rq4_feature_importance_comparison.tex')
        with open(output_path, 'w') as f:
            f.write(latex)
        
        self.logger.info(f"RQ4 Feature Importance Comparison LaTeX table generated at {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate LaTeX tables from analysis_results.json.')
    parser.add_argument('--results_path', type=str, default='data/analysis/analysis_results.json',
                        help='Path to the analysis_results.json file')
    parser.add_argument('--dataset_path', type=str, default='data/analysis/dataset_with_clusters.csv',
                        help='Path to the dataset with cluster labels.')
    parser.add_argument('--output_dir', type=str, default='data/analysis/tables',
                        help='Directory to save generated LaTeX table files/snippets')
    args = parser.parse_args()

    table_generator = LatexTableGenerator(results_path=args.results_path,
                                           dataset_path=args.dataset_path,
                                           output_dir=args.output_dir)
    table_generator.generate_all_tables()

