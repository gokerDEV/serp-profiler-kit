import pandas as pd
import numpy as np
import scikit_posthocs as sp
from scipy.stats import kruskal, chi2_contingency, mannwhitneyu, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import json
import logging
import argparse
from datetime import datetime
import os
import subprocess
from typing import Dict
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from kneed import KneeLocator

class DatasetAnalyzer:
    """
    Performs the full statistical analysis pipeline for the research paper.
-   This definitive version includes all RQ1-RQ4 analyses, is free of plotting code,
-   and preserves all original analytical logic while improving structure.
+   This definitive version includes all RQ1–RQ4 analyses, uses log1p + MinMax scaling
+   for mixed-type features, and is free of plotting code.
    """
    def __init__(self, input_file: str, output_dir: str, alpha: float = 0.05):
        self.input_file = input_file
        self.output_dir = output_dir
        self.alpha = alpha
        
        self.results_path = os.path.join(self.output_dir, 'analysis_results.json')
        self.dataset_with_clusters_path = os.path.join(self.output_dir, 'dataset_with_clusters.csv')
        
        self.df = None
        self.scaled_features_df = None
        self.results = {}
        
        os.makedirs(self.output_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.feature_cols = [
            'performance_score', 'accessibility_score', 'best-practices_score', 'seo_score',
            'query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1',
            'query_density_body', 'semantic_similarity_title_query',
            'semantic_similarity_content_query', 'word_count'
        ]
        self.technical_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        self.content_features = [col for col in self.feature_cols if col not in self.technical_features]

    def _load_and_prepare_data(self):
        """Loads the dataset and assigns the rank tier, without scaling."""
        self.logger.info(f"Loading dataset from {self.input_file}")
        self.df = pd.read_csv(self.input_file)
        
        if 'cluster' in self.df.columns:
            self.df = self.df.drop('cluster', axis=1)
        
        def assign_rank_tier(pos):
            if 1 <= pos <= 5: return 'high'
            if 6 <= pos <= 10: return 'medium'
            return 'low'
        self.df['rank_tier'] = self.df['position'].apply(assign_rank_tier)
        self.logger.info("Data loaded and prepared successfully.")

    def _analyze_rq1_clustering(self):
        """
        Performs RQ1 analysis, calculating all necessary metrics (WCSS, Silhouette, Calinski-Harabasz)
        in a unified loop to serve both plotting and table generation needs, aligned with the original methodology.
        """
        self.logger.info("Starting RQ1: Clustering Analysis (Unified Methodology)...")
        
        # 1. Prepare data locally: select features, handle missing values, and scale.
        X = self.df[self.feature_cols].copy()
        X = X.fillna(X.mean())
        
        
        # Apply log1p to word_count (heavy right skew) and MinMaxScaler to map all
        # features into [0, 1], keeping 0–1 and 0/1 variables in a comparable range.
        if 'word_count' in X.columns:
            X['word_count'] = np.log1p(X['word_count'])
            self.logger.info("Applied log1p transformation to word_count to normalize distribution")
        
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        # 2. Find Optimal K by calculating all required metrics in a single loop.
        self.logger.info("Calculating WCSS and Silhouette scores for K=2 to 10 (Calinski-Harabasz calculated for optimal K only)...")
        max_k = 10

        
        # 3. Perform final clustering.
        optimal_k, wcss, silhouette_scores, davies_bouldin_scores = self._find_optimal_clusters(X_scaled, max_k)
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(X_scaled)
        self.df['cluster_label'] = final_labels
        self.logger.info(f"Final clustering performed with K={optimal_k}")
        
        # As requested, calculate the overall silhouette score for the final clustering.
        overall_silhouette_score = silhouette_score(X_scaled, final_labels)
        self.logger.info(f"Overall silhouette score for final clustering (K={optimal_k}): {overall_silhouette_score}")
        
        # Calculate additional validation metrics for the final clustering
        calinski_score = calinski_harabasz_score(X_scaled, final_labels)
        davies_score = davies_bouldin_score(X_scaled, final_labels)
        
        # Calculate per-cluster silhouette scores
        per_cluster_silhouette = {}
        per_cluster_wcss = {}
        # Note: Per-cluster silhouette scores are not meaningful since silhouette score
        # requires at least 2 different labels. Instead, we'll calculate the overall
        # silhouette score and individual sample silhouette scores.
        silhouette_samples_scores = silhouette_samples(X_scaled, final_labels)
        for i in range(optimal_k):
            cluster_mask = final_labels == i
            if sum(cluster_mask) > 1:  # Check if cluster has more than one sample
                # Calculate mean silhouette score for samples in this cluster
                cluster_sil = np.mean(silhouette_samples_scores[cluster_mask])
                per_cluster_silhouette[f'cluster_{i}'] = float(cluster_sil)
                
                # Calculate WCSS for this cluster
                cluster_data = X_scaled[cluster_mask]
                cluster_center = kmeans_final.cluster_centers_[i]
                cluster_wcss = np.sum(cdist(cluster_data, [cluster_center], 'euclidean')**2)
                per_cluster_wcss[f'cluster_{i}'] = float(cluster_wcss)
        
        self.logger.info(f"Calinski-Harabasz score: {calinski_score}")
        self.logger.info(f"Davies-Bouldin score: {davies_score}")
        self.logger.info(f"Per-cluster silhouette scores: {per_cluster_silhouette}")
        self.logger.info(f"Per-cluster WCSS scores: {per_cluster_wcss}")
        
        # 5. Calculate cluster statistics for analysis
        cluster_stats = {}
        self.df['cluster_label'] = self.df['cluster_label'].astype(int)
        
        for cluster_id in range(optimal_k):
            cluster_mask = self.df['cluster_label'] == cluster_id
            cluster_stats[str(cluster_id)] = {
                'size': self.df[cluster_mask].shape[0],
                'feature_means': self.df[cluster_mask][self.feature_cols].mean().to_dict(),
                'feature_means_scaled': X_scaled[cluster_mask].mean().to_dict()
            }
        
        # 6. Perform statistical validation.
        kruskal_results = {}
        for feature in self.feature_cols:
            # Create groups for each cluster, filtering out empty groups
            groups = []
            group_labels = []
            for i in range(optimal_k):
                cluster_data = self.df[feature][self.df['cluster_label'] == i]
                # Remove NaN values and ensure we have data
                cluster_data = cluster_data.dropna()
                if len(cluster_data) > 0:
                    groups.append(cluster_data)
                    group_labels.append(i)
            
            # Only perform test if we have at least 2 groups with data
            if len(groups) >= 2:
                try:
                    # Kruskal-Wallis omnibus test
                    h_stat, p_val = kruskal(*groups)
                    result_entry = {'h_statistic': h_stat, 'p_value': p_val}
                    
                    # Dunn's post-hoc pairwise comparisons with Bonferroni correction
                    dunn_pvalues = None
                    if p_val < 0.05:
                        try:
                            # Flatten values and labels for scikit-posthocs
                            all_values = np.concatenate([g.values for g in groups])
                            all_labels = np.concatenate(
                                [[str(lbl)] * len(g) for lbl, g in zip(group_labels, groups)]
                            )
                            # Create DataFrame for posthoc_dunn
                            dunn_data = pd.DataFrame({
                                'value': all_values,
                                'group': all_labels
                            })
                            dunn_df = sp.posthoc_dunn(
                                dunn_data,
                                val_col='value',
                                group_col='group',
                                p_adjust='bonferroni'
                            )
                            # Ensure JSON-serializable output
                            dunn_df.index = dunn_df.index.astype(str)
                            dunn_df.columns = dunn_df.columns.astype(str)
                            dunn_pvalues = dunn_df.to_dict()
                        except Exception as e:
                            self.logger.warning(
                                f"Dunn post-hoc failed for feature {feature}: {str(e)}"
                            )
                    
                    if dunn_pvalues is not None:
                        result_entry['dunn_posthoc_pvalues'] = dunn_pvalues
                    
                    kruskal_results[feature] = result_entry

                except Exception as e:
                    self.logger.warning(
                        f"Kruskal-Wallis test failed for {feature}: {str(e)}"
                    )
                    kruskal_results[feature] = {
                        'h_statistic': np.nan,
                        'p_value': np.nan
                    }
            else:
                self.logger.warning(
                    f"Insufficient groups for Kruskal-Wallis test on {feature}"
                )
                kruskal_results[feature] = {
                    'h_statistic': np.nan,
                    'p_value': np.nan
                }
        
        # 7. Feature importance for clusters (RandomForestClassifier)
        #    To generate the 'rf_cluster_feature_importance' structure seen in the new analysis results.
        try:
            rf_clf = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            rf_clf.fit(X_scaled, final_labels)
            rf_importances = dict(zip(
                X_scaled.columns,
                rf_clf.feature_importances_
            ))
            # Sort by importance in descending order
            sorted_feats = sorted(
                rf_importances.items(),
                key=lambda x: x[1],
                reverse=True
            )
            rf_cluster_feature_importance = {
                'scores': {feat: float(score) for feat, score in rf_importances.items()},
                # List at least the first 5–10 features as in the analysis JSON
                'top_features': [feat for feat, _ in sorted_feats[:10]]
            }
        except Exception as e:
            self.logger.warning(f"RandomForestClassifier for cluster feature importance failed: {str(e)}")
            rf_cluster_feature_importance = {
                'error': str(e)
            }

        # 8. Store results in the original structure expected by the plot script.
        self.results['rq1_clustering'] = {
            'cluster_stats': cluster_stats,
            'wcss': wcss,
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'final_clustering_results': {
                'k_used': optimal_k,
                'kruskal_wallis_on_clusters': kruskal_results,
                'overall_silhouette_score': float(overall_silhouette_score),
                'calinski_harabasz_score': float(calinski_score),
                'davies_bouldin_score': float(davies_score),
                'per_cluster_silhouette': per_cluster_silhouette,
                'per_cluster_wcss': per_cluster_wcss
            },
            # Same field name as in the new analysis JSON:
            'rf_cluster_feature_importance': rf_cluster_feature_importance
        }
        self.logger.info("Finished RQ1 Analysis with original methodology.")
        


        
    def _find_optimal_clusters(self, X: pd.DataFrame, max_clusters: int) -> tuple:
        """
        Find optimal number of clusters using multiple methods:
        1. Elbow method (WCSS)
        2. Silhouette analysis
        3. Practical interpretability check
        
        Args:
            X: Scaled feature matrix
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            tuple: (optimal_k, wcss_dict, silhouette_scores_dict, davies_bouldin_scores_dict)
        """
        wcss = []
        silhouette_scores = []
        davies_bouldin_scores = [] 
        
        for k in range(2, max_clusters + 1):
            # Fit k-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            
            # Calculate WCSS
            wcss.append(kmeans.inertia_)
            
            # Silhouette score (mark with -1.0 if labels fall into a single class)
            if len(np.unique(kmeans.labels_)) > 1:
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            else:
                silhouette_scores.append(-1.0)

            # Davies-Bouldin index (may error in some edge cases)
            try:
                db_index = davies_bouldin_score(X, kmeans.labels_)
            except Exception:
                db_index = np.nan
            davies_bouldin_scores.append(db_index)
            
       
        # Find optimal k using the elbow method
        kl = KneeLocator(range(2, max_clusters + 1), wcss, curve='convex', direction='decreasing')
        optimal_k = kl.elbow
        
        # If no clear elbow is found, use the k with highest silhouette score
        if optimal_k is None:
            optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
        
        # Create dictionaries with k values as keys
        wcss_dict = {str(k): float(score) for k, score in zip(range(2, max_clusters + 1), wcss)}
        silhouette_scores_dict = {str(k): float(score) for k, score in zip(range(2, max_clusters + 1), silhouette_scores)}
        davies_bouldin_scores_dict = {
            str(k): (float(score) if not pd.isna(score) else None)
            for k, score in zip(range(2, max_clusters + 1), davies_bouldin_scores)
        } 
           
            
        return optimal_k, wcss_dict, silhouette_scores_dict, davies_bouldin_scores_dict

    def _analyze_rq2_visibility(self):
        """Performs RQ2 analysis: Associates profiles with ranking tiers."""
        self.logger.info("Running RQ2: Visibility Patterns...")
        self.results['rq2_visibility'] = {
            'google': self._run_visibility_analysis_for_engine('google'),
            'bing': self._run_visibility_analysis_for_engine('bing')
        }
        self.logger.info("Finished RQ2 Analysis.")
        
    def _run_visibility_analysis_for_engine(self, engine: str) -> Dict:
        """Helper to run RQ2 analysis for a given engine."""
        engine_df = self.df[self.df['engine'] == engine].copy()
        
        # Chi-square test
        contingency_table = pd.crosstab(engine_df['cluster_label'], engine_df['rank_tier'])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        
        # Kruskal-Wallis test for each feature by rank tier
        kruskal_results = {}
        # Ensure 'rank_tier' column exists and get unique tiers
        if 'rank_tier' in engine_df.columns:
            rank_tiers = sorted(engine_df['rank_tier'].dropna().unique())
            for feature in self.feature_cols:
                # Build groups per rank tier
                groups = []
                group_labels = []
                for tier in rank_tiers:
                    data = engine_df.loc[engine_df['rank_tier'] == tier, feature].dropna()
                    if len(data) > 0:
                        groups.append(data)
                        group_labels.append(tier)

                # Perform test only if there are multiple groups with data
                if len(groups) > 1:
                    try:
                        # Kruskal-Wallis omnibus test
                        h_stat, p_val = kruskal(*groups)
                        result_entry = {'h_statistic': h_stat, 'p_value': p_val}

                        # Dunn's post-hoc pairwise comparisons with Bonferroni correction
                        dunn_pvalues = None
                        if p_val < 0.05:
                            try:
                                all_values = np.concatenate([g.values for g in groups])
                                all_labels = np.concatenate([[str(lbl)] * len(g) for lbl, g in zip(group_labels, groups)])
                                # Create DataFrame for posthoc_dunn
                                dunn_data = pd.DataFrame({
                                    'value': all_values,
                                    'group': all_labels
                                })
                                dunn_df = sp.posthoc_dunn(
                                    dunn_data,
                                    val_col='value',
                                    group_col='group',
                                    p_adjust='bonferroni'
                                )
                                dunn_df.index = dunn_df.index.astype(str)
                                dunn_df.columns = dunn_df.columns.astype(str)
                                dunn_pvalues = dunn_df.to_dict()
                            except Exception as e:
                                self.logger.warning(f"Dunn post-hoc (rank_tier) failed for feature {feature}: {str(e)}")

                        if dunn_pvalues is not None:
                            result_entry['dunn_posthoc_pvalues'] = dunn_pvalues

                        kruskal_results[feature] = result_entry
                    except ValueError:
                        # This can happen if a group has no samples or only one sample
                        kruskal_results[feature] = {'h_statistic': np.nan, 'p_value': np.nan}

        return {
            'profile_distribution_vs_rank': {
                'chi2_statistic': chi2,
                'p_value': p,
                'contingency_table': contingency_table.to_dict()
            },
            'kruskal_wallis_by_rank': kruskal_results
        }

    def _analyze_rq3_comparison(self):
        """Performs RQ3 analysis: Compares top-ranking pages between Google and Bing."""
        self.logger.info("Running RQ3: Comparative System Analysis...")
        top5_df = self.df[self.df['position'] <= 5].copy()
        google_top5 = top5_df[top5_df['engine'] == 'google']
        bing_top5 = top5_df[top5_df['engine'] == 'bing']

        # Profile comparison
        contingency_table = pd.crosstab(top5_df['cluster_label'], top5_df['engine'])
        chi2, p, _, _ = chi2_contingency(contingency_table)

        # Feature comparison
        feature_comp_results = {}
        for feature in self.feature_cols:
            g_data, b_data = google_top5[feature], bing_top5[feature]
            if len(g_data) > 0 and len(b_data) > 0:
                u_stat, p_val = mannwhitneyu(g_data, b_data, alternative='two-sided')
                # Cohen's d calculation
                dof = len(g_data) + len(b_data) - 2
                cohens_d = (np.mean(g_data) - np.mean(b_data)) / np.sqrt(((len(g_data)-1)*np.std(g_data, ddof=1)**2 + (len(b_data)-1)*np.std(b_data, ddof=1)**2) / dof)
                feature_comp_results[feature] = {'u_statistic': u_stat, 'p_value': p_val, 'cohens_d': cohens_d}
        
        self.results['rq3_system_comparison'] = {
            'profile_comparison_top5': {
                'chi2_statistic': chi2, 
                'p_value': p,
                'contingency_table': contingency_table.to_dict()
            },
            'feature_comparison_top5': feature_comp_results
        }
        self.logger.info("Finished RQ3 Analysis.")

    def _analyze_rq4_priorities(self):
        """Performs RQ4 analysis: Infers factor priorities."""
        self.logger.info("Running RQ4: Factor Priorities...")
        self.results['rq4_factors'] = {
            'google': self._run_priority_analysis_for_engine('google'),
            'bing': self._run_priority_analysis_for_engine('bing')
        }
        self.logger.info("Finished RQ4 Analysis.")
    
    def _run_priority_analysis_for_engine(self, engine: str) -> Dict:
        """Helper to run RQ4 analysis for a given engine."""
        engine_df = self.df[self.df['engine'] == engine].copy()

        corr_matrix = engine_df[self.feature_cols + ['position']].corr(method='spearman')
        
        scaled_engine_df = engine_df.copy()
        
        if 'word_count' in scaled_engine_df.columns:
            scaled_engine_df['word_count'] = np.log1p(scaled_engine_df['word_count'])
            self.logger.info(f"Applied log1p transformation to word_count for {engine} engine")
        
        scaler = MinMaxScaler()
        scaled_engine_df[self.feature_cols] = scaler.fit_transform(scaled_engine_df[self.feature_cols])
        scaled_engine_df['serp_quintile'] = pd.qcut(scaled_engine_df['position'], 5, labels=False, duplicates='drop')
        
        regression_results = {}
        feature_sets = {'technical': self.technical_features, 'content': self.content_features, 'combined': self.feature_cols}
        for name, features in feature_sets.items():
            X = scaled_engine_df[features]
            y = scaled_engine_df['serp_quintile']
            try:
                model = OrderedModel(y, X, distr='logit').fit(method='bfgs', disp=False)
                
                # Test proportional odds assumption
                poa_test_result = self._test_proportional_odds_assumption(model, X, y)
                
                regression_results[name] = {
                    'coeffs': model.params.to_dict(), 'p_values': model.pvalues.to_dict(),
                    'pseudo_r2': model.prsquared, 'aic': model.aic, 'bic': model.bic,
                    'proportional_odds_test': poa_test_result
                }
            except Exception as e:
                regression_results[name] = {'error': str(e)}

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(engine_df[self.feature_cols], engine_df['position'])
        importances = {feat: imp for feat, imp in zip(self.feature_cols, rf.feature_importances_)}

        # Organize feature importance by category
        technical_factors = {k: v for k, v in importances.items() if k in self.technical_features}
        content_factors = {k: v for k, v in importances.items() if k in self.content_features}

        return {
            'spearman_correlation_matrix': corr_matrix.to_dict(),
            'ordinal_logistic_regression': regression_results,
            'feature_importance_rf_stats': {
                'technical_factors': technical_factors,
                'content_factors': content_factors,
                'all_factors': importances
            }
        }

    def _test_proportional_odds_assumption(self, model, X, y):
        """
        Brant testini R tarafında (MASS::polr + brant::brant) ile çalıştır,
        sonucu Python'a taşı.
        """
        # 1) Modelde kullanılan veriyi R'a göndermek için DataFrame hazırla
        df_model = X.copy()
        df_model["serp_quintile"] = y

        try:
            brant_res = self._run_brant_via_r(df_model)
        except Exception as e:
            # R tarafında bir hata olursa test yapılmadı olarak işaretle
            return {
                "test_performed": False,
                "reason": f"Brant test via R failed: {e}",
                "assumption_met": None,
                "alpha_level": self.alpha
            }

        chi2_stat = float(brant_res["chi2_statistic"])
        df_val    = int(brant_res["degrees_of_freedom"])
        p_val     = float(brant_res["p_value"])

        assumption_met = bool(p_val > self.alpha)

        return {
            "test_performed": True,
            "chi2_statistic": chi2_stat,
            "degrees_of_freedom": df_val,
            "p_value": p_val,
            "assumption_met": assumption_met,
            "alpha_level": self.alpha
        }
            
            
    def _export_for_brant(self, df_model):
        in_path = os.path.join(self.output_dir, "brant_model_data.csv")
        df_model.to_csv(in_path, index=False)
        return in_path
    
    
    def _run_brant_via_r(self, df_model) -> dict:
        in_path = self._export_for_brant(df_model)
        out_path = os.path.join(self.output_dir, "brant_result.json")

        # Convert to absolute paths so working directory doesn't matter
        in_path = os.path.abspath(in_path)
        out_path = os.path.abspath(out_path)

        # Rscript komutunu çalıştır - use absolute path for R script
        # Get the directory where this Python file is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        r_script_path = os.path.join(current_dir, "rq4_brant.R")
        
        # Ensure the script exists
        if not os.path.exists(r_script_path):
            raise FileNotFoundError(f"R script not found at {r_script_path}")
        
        # Use absolute path for R script too
        r_script_path = os.path.abspath(r_script_path)
        
        cmd = [
            "Rscript",
            r_script_path,
            in_path,
            out_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"R script failed with exit code {e.returncode}"
            if e.stdout:
                error_msg += f"\nSTDOUT: {e.stdout}"
            if e.stderr:
                error_msg += f"\nSTDERR: {e.stderr}"
            raise RuntimeError(error_msg) from e

        if not os.path.exists(out_path):
            raise FileNotFoundError(f"R script did not create output file at {out_path}")

        with open(out_path, "r", encoding="utf-8") as f:
            res = json.load(f)

        return res

    def _add_dataset_info(self):
        """Adds dataset information for plotting functions."""
        self.results['dataset_info'] = {
            'lighthouse_metrics_found': self.technical_features,
            'content_metrics_found': self.content_features,
            'ranking_groups': {'high': (1, 5), 'medium': (6, 10), 'low': (11, 20)},
            'position_column': 'position',
            'total_samples': len(self.df),
            'engines_present': self.df['engine'].unique().tolist() if 'engine' in self.df.columns else []
        }

    def save_outputs(self):
        """Saves all outputs: the results JSON and the dataset with cluster labels."""
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer): return int(obj)
                if isinstance(obj, np.floating): return float(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        with open(self.results_path, 'w') as f:
            json.dump(self.results, f, indent=4, cls=NpEncoder)
        self.logger.info(f"Analysis results successfully saved to {self.results_path}")

        self.df.to_csv(self.dataset_with_clusters_path, index=False)
        self.logger.info(f"Dataset with cluster labels saved to {self.dataset_with_clusters_path}")

    def run(self):
        """Executes the entire analysis pipeline from start to finish."""
        self.logger.info("Starting analysis pipeline...")
        self._load_and_prepare_data()
        self._analyze_rq1_clustering()
        self._analyze_rq2_visibility()
        self._analyze_rq3_comparison()
        self._analyze_rq4_priorities()
        self._add_dataset_info()
        self.save_outputs()
        self.logger.info("Analysis pipeline completed successfully.")

def main():
    parser = argparse.ArgumentParser(description='Run complete analysis pipeline for the SERP study.')
    parser.add_argument('--input', type=str, default='data/processed/dataset_processed.csv', help='Input dataset CSV file')
    parser.add_argument('--output', type=str, default='data/analysis', help='Output directory for analysis results')
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(input_file=args.input, output_dir=args.output)
    
    start_time = datetime.now()
    analyzer.run()
    duration = datetime.now() - start_time
    logging.info(f"Total analysis duration: {duration}")

if __name__ == "__main__":
    main()