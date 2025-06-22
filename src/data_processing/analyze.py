import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, levene, mannwhitneyu, kruskal, chi2_contingency
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import os
import json
import logging
from datetime import datetime
import argparse

# Add the root directory to the Python path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils # Assuming Utils.set_colorful_logging is still used

class DatasetAnalyzer:
    """
    Advanced dataset analyzer for research questions RQ1-RQ4.
    This class implements comprehensive statistical analyses and saves results to JSON.
    Plotting and LaTeX table generation are handled by separate scripts.
    """
    
    def __init__(self,
                 input_file: str,
                 output_dir: str,
                 alpha: float = 0.05,
                 ranking_groups: Optional[Dict[str, Tuple[int, int]]] = None):
        self.input_file = input_file
        self.output_dir = output_dir
        self.alpha = alpha
        self.ranking_groups = ranking_groups or {
            'high': (1, 5),
            'medium': (6, 10),
            'low': (11, 20)
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        for subdir in ['tables', 'plots', 'latex_tables']: 
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        self.logger = Utils.set_colorful_logging('DatasetAnalyzerCore')
        
        self.lighthouse_scores = [
            'performance_score', 'accessibility_score',
            'best-practices_score', 'seo_score'
        ]
        self.content_metrics = [
            'query_in_title', 'query_in_h1', 'exact_query_in_title',
            'exact_query_in_h1', 'query_density_body',
            'semantic_similarity_title_query',
            'semantic_similarity_content_query', 'word_count'
        ]
        self.binary_metrics = [
            'query_in_title', 'query_in_h1',
            'exact_query_in_title', 'exact_query_in_h1'
        ]
        self.categorical_columns = ['engine']
        self.position_column = 'position'
        self.info_columns = ['url', 'file_name']
        self.numerical_columns = []
        
        self.stats = {
            'dataset_info': {},
            'rq1_clustering': {},
            'rq2_visibility': {},
            'rq3_comparison': {},
            'rq4_factors': {}
        }
        self.df = None
        self.scaled_feature_df_for_clustering = None

    def run(self):
        self.logger.info("Starting comprehensive analysis...")
        self._load_and_preprocess_data()
        if self.df is not None:
            self._analyze_rq1_clustering()
            self._analyze_rq2_visibility()
            self._analyze_rq3_comparison()
            self._analyze_rq4_factors()
            self._save_results()
            self.logger.info("Core analysis completed. Results saved to JSON and processed CSV.")
        else:
            self.logger.error("DataFrame not loaded. Aborting analysis.")

    def _load_and_preprocess_data(self):
        self.logger.info("Loading and preprocessing data...")
        try:
            self.df = pd.read_csv(self.input_file)
            self.stats['dataset_info']['initial_rows'] = len(self.df)

            if self.position_column not in self.df.columns and 'serp_position' in self.df.columns:
                self.logger.info(f"Renaming 'serp_position' to '{self.position_column}'")
                self.df.rename(columns={'serp_position': self.position_column}, inplace=True)
            
            if 'url' in self.df.columns and 'hostname' not in self.df.columns:
                self.df['hostname'] = self.df['url'].apply(
                    lambda x: Utils.get_hostname(x) if pd.notnull(x) else None
                )

            numerical_cols_set = set()
            available_lighthouse = [col for col in self.lighthouse_scores if col in self.df.columns]
            numerical_cols_set.update(available_lighthouse)
            if self.position_column in self.df.columns:
                numerical_cols_set.add(self.position_column)
            available_content = [col for col in self.content_metrics if col in self.df.columns]
            numerical_cols_set.update(available_content)
            self.numerical_columns = list(numerical_cols_set)

            for binary_metric in self.binary_metrics:
                if binary_metric in self.df.columns and self.df[binary_metric].dtype != np.int64:
                    self.df[binary_metric] = self.df[binary_metric].astype(int)
            
            self.stats['dataset_info']['processed_rows'] = len(self.df)
            self.stats['dataset_info']['lighthouse_metrics_found'] = available_lighthouse
            self.stats['dataset_info']['content_metrics_found'] = available_content
            self.logger.info(f"Loaded and preprocessed dataset with {len(self.df)} rows.")

        except FileNotFoundError:
            self.logger.error(f"Input file not found: {self.input_file}")
            self.df = None
        except Exception as e:
            self.logger.error(f"Error loading data: {repr(e)}")
            self.df = None

    def _analyze_rq1_clustering(self):
        self.logger.info("RQ1: Starting Resource Profiling and Clustering Analysis...")
        features_for_clustering = [col for col in self.lighthouse_scores + self.content_metrics if col in self.df.columns]
        
        if not features_for_clustering:
            self.logger.warning("No features available for clustering. Skipping RQ1.")
            self.stats['rq1_clustering']['error'] = "No features for clustering."
            return

        X = self.df[features_for_clustering].copy()
        for col in X.select_dtypes(include=np.number).columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].mean(), inplace=True)
        
        scaler = MinMaxScaler()
        X_scaled_array = scaler.fit_transform(X)
        self.scaled_feature_df_for_clustering = pd.DataFrame(X_scaled_array, columns=X.columns, index=X.index)

        wcss_scores_dict, silhouette_k_scores_dict = self._calculate_optimal_k_metrics(self.scaled_feature_df_for_clustering, max_clusters=10)
        
        optimal_k = 6 
        try:
            from kneed import KneeLocator
            k_values_wcss = sorted([int(k) for k in wcss_scores_dict.keys()])
            wcss_values_list = [wcss_scores_dict[k] for k in k_values_wcss]

            optimal_k_elbow = None
            if len(k_values_wcss) >= 3: 
                kl = KneeLocator(k_values_wcss, wcss_values_list, curve='convex', direction='decreasing', S=1.0)
                optimal_k_elbow = kl.elbow
            
            if optimal_k_elbow is not None:
                optimal_k = int(optimal_k_elbow)
                self.logger.info(f"Optimal K from Elbow method (KneeLocator): {optimal_k}")
            else:
                valid_silhouette_scores = {k: v for k, v in silhouette_k_scores_dict.items() if v != -1.0 and pd.notna(v)}
                if valid_silhouette_scores:
                    optimal_k = max(valid_silhouette_scores, key=valid_silhouette_scores.get)
                    self.logger.info(f"Elbow/KneeLocator not definitive, Optimal K from Max Silhouette: {optimal_k}")
                else:
                    self.logger.warning(f"No valid silhouette scores, K remains fallback {optimal_k}")
        except ImportError:
            self.logger.warning("kneed library not installed. Using max silhouette score for optimal K.")
            valid_silhouette_scores = {k: v for k, v in silhouette_k_scores_dict.items() if v != -1.0 and pd.notna(v)}
            if valid_silhouette_scores:
                optimal_k = max(valid_silhouette_scores, key=valid_silhouette_scores.get)
            else:
                self.logger.warning(f"No valid silhouette scores, K remains fallback {optimal_k}")
        except Exception as e:
            self.logger.error(f"Error determining optimal K with KneeLocator/Silhouette: {repr(e)}. Defaulting to 6.")
            optimal_k = 6
        
        self.stats['rq1_clustering']['optimal_k_metrics'] = {
            'wcss': {str(k): v for k,v in wcss_scores_dict.items()},
            'silhouette_scores_for_k': {str(k): v for k,v in silhouette_k_scores_dict.items()}
        }
        self.stats['rq1_clustering']['selected_optimal_k'] = int(optimal_k)
        
        kmeans = KMeans(n_clusters=int(optimal_k), random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10)
        cluster_labels = kmeans.fit_predict(self.scaled_feature_df_for_clustering)
        self.df['cluster_label'] = cluster_labels
        
        self.stats['rq1_clustering']['cluster_labels_assigned'] = True
        self.stats['rq1_clustering']['cluster_stats'] = self._analyze_clusters(features_for_clustering)
        self.stats['rq1_clustering']['cluster_validation'] = self._validate_clusters(features_for_clustering)
        self.stats['rq1_clustering']['feature_importance_rf_classifier'] = self._analyze_feature_importance_for_clusters(self.scaled_feature_df_for_clustering, cluster_labels)
        self.logger.info(f"RQ1: Clustering analysis complete with K={optimal_k}.")

    def _calculate_optimal_k_metrics(self, X_scaled: pd.DataFrame, max_clusters: int) -> Tuple[Dict[int, float], Dict[int, float]]:
        wcss = {}
        silhouette_k_scores = {}
        for k_val in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k_val, random_state=42, n_init='auto' if hasattr(KMeans(), 'n_init') and KMeans().n_init == 'auto' else 10)
            labels = kmeans.fit_predict(X_scaled)
            wcss[k_val] = float(kmeans.inertia_)
            if len(np.unique(labels)) > 1: 
                 silhouette_k_scores[k_val] = float(silhouette_score(X_scaled, labels))
            else:
                silhouette_k_scores[k_val] = -1.0 
        return wcss, silhouette_k_scores

    def _analyze_clusters(self, features: List[str]) -> Dict:
        # ... (Implementation from previous version, ensuring JSON serializable output) ...
        cluster_stats_data = {}
        if 'cluster_label' not in self.df.columns:
            return {'error': 'Cluster labels not found in DataFrame.'}
        
        unique_labels = sorted(self.df['cluster_label'].unique())
        for cluster_id in unique_labels:
            cluster_data_df = self.df[self.df['cluster_label'] == cluster_id]
            scaled_cluster_data_df = self.scaled_feature_df_for_clustering.loc[cluster_data_df.index] if self.scaled_feature_df_for_clustering is not None else pd.DataFrame()

            feature_stats_original = {}
            for feature in features:
                if feature in cluster_data_df:
                    feature_stats_original[feature] = {
                        'mean': float(cluster_data_df[feature].mean(numeric_only=True)),
                        'median': float(cluster_data_df[feature].median(numeric_only=True)),
                        'std': float(cluster_data_df[feature].std(numeric_only=True))
                    }
            
            feature_means_scaled = {}
            if not scaled_cluster_data_df.empty:
                feature_means_scaled = {col: float(val) for col, val in scaled_cluster_data_df[features].mean().to_dict().items()}


            cluster_stats_data[f'cluster_{cluster_id}'] = {
                'size': int(len(cluster_data_df)),
                'percentage': float((len(cluster_data_df) / len(self.df)) * 100 if len(self.df) > 0 else 0),
                'feature_stats_original_scale': feature_stats_original,
                'feature_means_scaled': feature_means_scaled
            }
        return cluster_stats_data


    def _validate_clusters(self, features: List[str]) -> Dict:
        # ... (Implementation from previous version, ensuring JSON serializable output) ...
        validation_results = {}
        if 'cluster_label' not in self.df.columns:
            return {'error': 'Cluster labels not found for validation.'}

        for feature in features:
            if feature not in self.df.columns:
                self.logger.warning(f"Feature {feature} not in DataFrame for cluster validation.")
                continue
            
            groups = []
            for _, group_df in self.df.groupby('cluster_label'):
                feature_data = group_df[feature].dropna()
                if not feature_data.empty: 
                    groups.append(feature_data.values)
            
            if len(groups) < 2:
                self.logger.warning(f"Not enough groups with data for Kruskal-Wallis on feature '{feature}'.")
                validation_results[feature] = {'error': 'Not enough groups with data for comparison'}
                continue
            
            is_identical = False
            if len(groups) > 0:
                first_group_unique = np.unique(groups[0])
                if len(first_group_unique) == 1: 
                    is_identical = all(len(np.unique(g)) == 1 and np.array_equal(np.unique(g),first_group_unique) for g in groups)

            h_stat, pval = np.nan, np.nan
            test_name = 'Kruskal-Wallis'
            if is_identical:
                pval = 1.0 
                test_name = 'Kruskal-Wallis (Identical Groups)'
            else:
                try:
                    valid_groups_for_kruskal = [g for g in groups if len(g) > 0]
                    if len(valid_groups_for_kruskal) >= 2:
                         h_stat, pval = stats.kruskal(*valid_groups_for_kruskal)
                    else:
                        pval = np.nan 
                except ValueError as e:
                    self.logger.warning(f"Kruskal-Wallis ValueError for {feature}: {repr(e)}. Assigning NaN.")
                    test_name = 'Kruskal-Wallis (Error)'
            
            validation_results[feature] = {
                'cluster_difference_test': {
                    'test_name': test_name,
                    'statistic': float(h_stat) if pd.notna(h_stat) else 'NaN',
                    'pvalue': float(pval) if pd.notna(pval) else 'NaN',
                    'significant': (pval < self.alpha) if pd.notna(pval) else 'N/A'
                }
            }
        return validation_results

    def _analyze_feature_importance_for_clusters(self, X_scaled: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        # ... (Implementation from previous version) ...
        if X_scaled.empty or len(cluster_labels) == 0 or len(np.unique(cluster_labels)) < 2:
            return {'error': 'Insufficient data or classes for cluster feature importance.'}
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, cluster_labels)
            importance_scores = dict(zip(X_scaled.columns, rf.feature_importances_))
            sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            return {
                'scores': {k: float(v) for k, v in importance_scores.items()},
                'top_features': [feature for feature, _ in sorted_features[:5]]
            }
        except Exception as e:
            self.logger.error(f"Error in _analyze_feature_importance_for_clusters: {repr(e)}")
            return {'error': str(e)}


    def _analyze_rq2_visibility(self):
        # ... (Implementation from previous version) ...
        self.logger.info("RQ2: Starting Visibility Patterns Analysis...")
        if self.position_column not in self.df.columns:
            self.stats['rq2_visibility']['error'] = "Position column not found."
            return

        for engine in self.df['engine'].unique():
            engine_data_df = self.df[self.df['engine'] == engine].copy()
            if engine_data_df.empty: continue
            
            engine_data_df['ranking_group'] = engine_data_df[self.position_column].apply(self._get_ranking_group)
            
            self.stats['rq2_visibility'][engine] = {
                'profile_distribution_stats': self._analyze_profile_distribution(engine_data_df),
                'feature_analysis_by_ranking_stats': self._analyze_features_by_ranking(
                    engine_data_df, 
                    features_to_analyze=self.lighthouse_scores + self.content_metrics,
                    group_col='ranking_group'
                )
            }
        self.logger.info("RQ2: Visibility patterns analysis complete.")


    def _get_ranking_group(self, position: int) -> str:
        # ... (Implementation from previous version) ...
        for group, (start, end) in self.ranking_groups.items():
            if start <= int(position) <= end: 
                return group
        return 'other'

    def _analyze_profile_distribution(self, data: pd.DataFrame) -> Dict:
        # ... (Implementation from previous version) ...
        if 'cluster_label' not in data.columns or 'ranking_group' not in data.columns:
            return {'error': 'Required columns for profile distribution missing.'}
        
        contingency_table = pd.crosstab(data['ranking_group'], data['cluster_label'])
        
        if contingency_table.sum().sum() < 10 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
             self.logger.warning("Not enough data for Chi-square test in profile distribution.")
             chi2, pval, dof, significant = np.nan, np.nan, np.nan, 'N/A (Insufficient Data)'
        else:
            try:
                chi2, pval, dof, expected = chi2_contingency(contingency_table)
                significant = pval < self.alpha
            except ValueError as e: 
                self.logger.warning(f"Chi-square test failed for profile distribution: {repr(e)}")
                chi2, pval, dof, significant = np.nan, np.nan, np.nan, 'N/A (Error)'

        return {
            'contingency_table_dict': contingency_table.to_dict(),
            'percentages_by_profile_in_rank_group': (contingency_table.T / contingency_table.sum(axis=1)).T.mul(100).fillna(0).to_dict(),
            'percentages_of_rank_group_in_profile': (contingency_table / contingency_table.sum(axis=0)).mul(100).fillna(0).to_dict(),
            'chi_square_test': {
                'statistic': float(chi2) if pd.notna(chi2) else 'NaN',
                'p_value': float(pval) if pd.notna(pval) else 'NaN',
                'degrees_of_freedom': int(dof) if pd.notna(dof) else 'NaN',
                'significant': significant
            }
        }

    def _analyze_features_by_ranking(self, data: pd.DataFrame, features_to_analyze: List[str], group_col: str) -> Dict:
        # ... (Implementation from previous version, with .map instead of .applymap) ...
        analysis_results = {}
        if group_col not in data.columns:
            return {"error": f"Grouping column '{group_col}' not found."}

        for feature in features_to_analyze:
            if feature not in data.columns: continue

            grouped_data = {name: group_df[feature].dropna() for name, group_df in data.groupby(group_col)}
            valid_groups_data = {name: g_data for name, g_data in grouped_data.items() if len(g_data) >= 3}
            
            if len(valid_groups_data) < 2:
                analysis_results[feature] = {'error': 'Not enough valid groups for comparison'}
                continue

            normality_tests = {name: (stats.shapiro(g_data)[1] > self.alpha if len(g_data) >=3 else 'N/A') 
                               for name, g_data in valid_groups_data.items()}
            all_normal = all(res for res in normality_tests.values() if isinstance(res, bool)) # Check only boolean results
            
            test_name, stat, pval = "N/A", np.nan, np.nan
            group_values_list = [g_data.values for g_data in valid_groups_data.values()]

            try:
                if len(valid_groups_data) == 2:
                    g1_data, g2_data = group_values_list[0], group_values_list[1]
                    if all_normal:
                        levene_stat, levene_pval = stats.levene(g1_data, g2_data)
                        test_name = "Welch's t-test" if levene_pval <= self.alpha else "Student's t-test"
                        stat, pval = stats.ttest_ind(g1_data, g2_data, equal_var=(levene_pval > self.alpha))
                    else:
                        stat, pval = stats.mannwhitneyu(g1_data, g2_data, alternative='two-sided')
                        test_name = "Mann-Whitney U"
                elif len(valid_groups_data) > 2:
                    test_name = "Kruskal-Wallis" if not all_normal else "ANOVA"
                    stat, pval = stats.kruskal(*group_values_list) if not all_normal else stats.f_oneway(*group_values_list)
            except ValueError as e:
                self.logger.warning(f"Stat test error for {feature} by {group_col}: {repr(e)}")
            
            desc_stats = data.groupby(group_col)[feature].agg(['count', 'mean', 'std', 'median', 'min', 'max'])
            
            analysis_results[feature] = {
                'normality_tests': normality_tests,
                'statistical_test': {
                    'name': test_name,
                    'statistic': float(stat) if pd.notna(stat) else 'NaN',
                    'p_value': float(pval) if pd.notna(pval) else 'NaN',
                    'significant': (pval < self.alpha) if pd.notna(pval) else 'N/A'
                },
                'descriptive_stats_by_group': desc_stats.map(lambda x: float(x) if isinstance(x, (int, float, np.number)) else x).to_dict()
            }
        return analysis_results

    def _analyze_rq3_comparison(self):
        # ... (Implementation from previous version) ...
        self.logger.info("RQ3: Starting Comparative System Analysis...")
        if self.position_column not in self.df.columns:
            self.stats['rq3_comparison']['error'] = "Position column not found."
            return

        top_n_ranks = self.ranking_groups['high'][1] 
        top_results_df = self.df[self.df[self.position_column] <= top_n_ranks].copy()

        if top_results_df.empty or len(top_results_df['engine'].unique()) < 2:
            self.stats['rq3_comparison']['error'] = "Not enough data or engines for top results comparison."
            return
            
        self.stats['rq3_comparison']['profile_comparison_top_ranks_stats'] = self._compare_top_profiles(top_results_df)
        self.stats['rq3_comparison']['feature_comparison_top_ranks_stats'] = self._compare_top_features(top_results_df, self.lighthouse_scores + self.content_metrics)
        self.logger.info("RQ3: Comparative system analysis complete.")

    def _compare_top_profiles(self, data: pd.DataFrame) -> Dict:
        # ... (Implementation from previous version) ...
        if 'cluster_label' not in data.columns or 'engine' not in data.columns:
            return {'error': 'Required columns for top profile comparison missing.'}
        
        contingency_table = pd.crosstab(data['engine'], data['cluster_label'])
        if contingency_table.sum().sum() < 10 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
             self.logger.warning("Not enough data for Chi-square test in top profile comparison.")
             chi2, pval, dof, significant = np.nan, np.nan, np.nan, 'N/A (Insufficient Data)'
        else:
            try:
                chi2, pval, dof, expected = chi2_contingency(contingency_table)
                significant = pval < self.alpha
            except ValueError as e:
                self.logger.warning(f"Chi-square test failed for top profile comparison: {repr(e)}")
                chi2, pval, dof, significant = np.nan, np.nan, np.nan, 'N/A (Error)'
        
        return {
            'contingency_table_dict': contingency_table.to_dict(),
            'chi_square_test': {
                'statistic': float(chi2) if pd.notna(chi2) else 'NaN',
                'p_value': float(pval) if pd.notna(pval) else 'NaN',
                'degrees_of_freedom': int(dof) if pd.notna(dof) else 'NaN',
                'significant': significant
            }
        }

    def _compare_top_features(self, data: pd.DataFrame, features_to_compare: List[str]) -> Dict:
        # ... (Implementation from previous version with NaN handling for Cohen's d) ...
        comparison_results = {}
        engines = data['engine'].unique()
        if len(engines) < 2:
            return {'error': 'Need at least two engines to compare features.'}
        
        data_google = data[data['engine'] == 'google']
        data_bing = data[data['engine'] == 'bing']

        if data_google.empty or data_bing.empty:
            return {'error': 'Data missing for one or both engines.'}

        for feature in features_to_compare:
            if feature not in data.columns: continue
            
            google_feature_data = data_google[feature].dropna()
            bing_feature_data = data_bing[feature].dropna()

            if len(google_feature_data) < 3 or len(bing_feature_data) < 3:
                comparison_results[feature] = {'error': 'Insufficient data for comparison after dropna'}
                continue

            _, google_norm_pval = stats.shapiro(google_feature_data)
            _, bing_norm_pval = stats.shapiro(bing_feature_data)
            both_normal = (google_norm_pval > self.alpha and bing_norm_pval > self.alpha)

            homogeneous_var = None
            levene_p = np.nan
            if both_normal:
                if len(google_feature_data) > 1 and len(bing_feature_data) > 1:
                    _, levene_p = stats.levene(google_feature_data, bing_feature_data)
                    homogeneous_var = levene_p > self.alpha
                else:
                    homogeneous_var = False 
            
            test_name, stat, pval = "N/A", np.nan, np.nan
            try:
                if both_normal:
                    equal_v = homogeneous_var if homogeneous_var is not None else False
                    stat, pval = stats.ttest_ind(google_feature_data, bing_feature_data, equal_var=equal_v)
                    test_name = "Student's t-test" if equal_v else "Welch's t-test"
                else:
                    stat, pval = stats.mannwhitneyu(google_feature_data, bing_feature_data, alternative='two-sided')
                    test_name = "Mann-Whitney U"
            except ValueError as e:
                self.logger.warning(f"Stat test error for {feature} in RQ3: {repr(e)}")

            mean_google, mean_bing = google_feature_data.mean(), bing_feature_data.mean()
            std_google, std_bing = google_feature_data.std(), bing_feature_data.std()
            n_google, n_bing = len(google_feature_data), len(bing_feature_data)
            effect_size = np.nan
            if n_google > 1 and n_bing > 1 and pd.notna(std_google) and pd.notna(std_bing) and (std_google > 0 or std_bing > 0) :
                # Ensure pooled_std denominator is not zero
                if (n_google + n_bing - 2) > 0:
                    pooled_std_variance = ((n_google - 1) * std_google**2 + (n_bing - 1) * std_bing**2) / (n_google + n_bing - 2)
                    if pooled_std_variance > 0:
                        pooled_std = np.sqrt(pooled_std_variance)
                        effect_size = (mean_google - mean_bing) / pooled_std if pooled_std > 0 else 0.0
                    else: # Handles case where pooled_std_variance is 0 (e.g. all values in both groups are identical)
                        effect_size = 0.0 if mean_google == mean_bing else np.nan 
            
            comparison_results[feature] = {
                'normality_test': {'google_pval': float(google_norm_pval), 'bing_pval': float(bing_norm_pval)},
                'variance_homogeneity_pval': float(levene_p) if pd.notna(levene_p) else 'N/A',
                'statistical_test': {'name': test_name, 'statistic': float(stat) if pd.notna(stat) else 'NaN', 'p_value': float(pval) if pd.notna(pval) else 'NaN', 'significant': (pval < self.alpha) if pd.notna(pval) else 'N/A'},
                'effect_size': {'cohens_d': float(effect_size) if pd.notna(effect_size) else 'NaN', 'interpretation': self._interpret_effect_size(effect_size)},
                'descriptive_stats': {
                    'google': {'mean': float(mean_google), 'median': float(google_feature_data.median()), 'std': float(std_google) if pd.notna(std_google) else 'NaN', 'n': int(n_google)},
                    'bing': {'mean': float(mean_bing), 'median': float(bing_feature_data.median()), 'std': float(std_bing) if pd.notna(std_bing) else 'NaN', 'n': int(n_bing)}
                }
            }
        return comparison_results

    def _interpret_effect_size(self, d: float) -> str:
        # ... (Implementation from previous version) ...
        if pd.isna(d): return "N/A"
        d_abs = abs(d)
        if d_abs < 0.2: return "negligible"
        elif d_abs < 0.5: return "small"
        elif d_abs < 0.8: return "medium"
        else: return "large"

    def _analyze_rq4_factors(self):
        # ... (Implementation from previous version) ...
        self.logger.info("RQ4: Starting Factor Priority Inference...")
        if self.position_column not in self.df.columns:
            self.stats['rq4_factors']['error'] = "Position column not found."
            return

        for engine in self.df['engine'].unique():
            engine_data_df = self.df[self.df['engine'] == engine].copy()
            if engine_data_df.empty: continue

            self.stats['rq4_factors'][engine] = {
                'correlations_stats': self._analyze_correlations(engine_data_df, self.lighthouse_scores + self.content_metrics),
                'regression_stats': self._perform_regression_analysis(engine_data_df),
                'feature_importance_rf_stats': self._analyze_feature_importance_rq4(engine_data_df)
            }
        self.logger.info("RQ4: Factor priority inference complete.")

    def _analyze_correlations(self, data: pd.DataFrame, features_to_correlate: List[str]) -> Dict:
        # ... (Implementation from previous version with NaN handling) ...
        correlations = {}
        if self.position_column not in data.columns:
            return {'error': 'Position column missing for correlation.'}

        for feature in features_to_correlate:
            if feature not in data.columns: continue
            data_pos = data[self.position_column].dropna()
            data_feat = data[feature].dropna()
            
            common_indices = data_pos.index.intersection(data_feat.index)
            corr_coeff, p_val = np.nan, np.nan 
            
            if len(common_indices) >= 2: 
                data_pos_aligned = data_pos.loc[common_indices]
                data_feat_aligned = data_feat.loc[common_indices]
                if len(data_pos_aligned.unique()) > 1 and len(data_feat_aligned.unique()) > 1:
                    try:
                        corr_coeff, p_val = stats.spearmanr(data_pos_aligned, data_feat_aligned)
                    except Exception as e:
                        self.logger.warning(f"Spearman correlation failed for {feature}: {repr(e)}")
                else:
                    self.logger.warning(f"Not enough variance for Spearman correlation on {feature}")
            else:
                 self.logger.warning(f"Not enough common non-NaN pairs for Spearman correlation on {feature}")

            correlations[feature] = {
                'coefficient': float(corr_coeff) if pd.notna(corr_coeff) else 'NaN',
                'p_value': float(p_val) if pd.notna(p_val) else 'NaN',
                'significant': (p_val < self.alpha) if pd.notna(p_val) else 'N/A'
            }
        return correlations
        
    def _perform_regression_analysis(self, data: pd.DataFrame) -> Dict:
        # ... (Implementation from previous version with minor adjustments) ...
        if self.position_column not in data.columns or data[self.position_column].isna().all():
            return {'error': 'No position data for regression'}

        data_copy = data.copy()
        try:
            data_copy['position_category'] = pd.qcut(
                data_copy[self.position_column],
                q=5, labels=['top', 'high', 'medium', 'low', 'bottom'], duplicates='drop'
            )
        except ValueError as e:
            return {'error': f'Failed to create position categories: {repr(e)}'}

        tech_features = [f for f in self.lighthouse_scores if f in data_copy.columns]
        content_features = [f for f in self.content_metrics if f in data_copy.columns]
        
        regression_outputs = {}
        scaler = MinMaxScaler()

        for model_name, feature_set in [
            ('technical_only', tech_features),
            ('content_only', content_features),
            ('combined', tech_features + content_features)
        ]:
            if not feature_set:
                regression_outputs[model_name] = {'error': f'No features for {model_name} model.'}
                continue

            cols_to_check_for_na = ['position_category'] + feature_set
            data_clean = data_copy.dropna(subset=cols_to_check_for_na)

            if len(data_clean) < max(10, len(feature_set) * 2 + 5) or len(data_clean['position_category'].unique()) < 2 :
                regression_outputs[model_name] = {'error': 'Insufficient data or category diversity after NaN removal.'}
                continue
            
            X = data_clean[feature_set].copy()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
            y_ord = data_clean['position_category']
            
            regression_outputs[model_name] = self._fit_ordinal_regression(X_scaled, y_ord)
            
        return regression_outputs

    def _fit_ordinal_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        # ... (Implementation from previous version, ensure robust error handling) ...
        try:
            category_map = {'top': 0, 'high': 1, 'medium': 2, 'low': 3, 'bottom': 4}
            y_numeric = y.map(category_map) if y.dtype.name == 'category' or y.dtype == 'object' else y
            
            X_numeric = X.select_dtypes(include=np.number)
            if X_numeric.shape[1] != X.shape[1]:
                self.logger.warning("Non-numeric columns dropped from X in _fit_ordinal_regression")
            if X_numeric.empty:
                 return {'error': 'No numeric features left for regression model'}

            model = OrderedModel(y_numeric, X_numeric, distr='logit')
            result = model.fit(method='bfgs', disp=0, maxiter=200, gtol=1e-4) 
            
            coeffs = {str(k): float(v) for k, v in result.params.items()}
            pvals = {str(k): float(v) for k, v in result.pvalues.items()}
            
            significant_features = [f for f in X_numeric.columns if f in pvals and pvals[f] < self.alpha]
            
            return {
                'pseudo_r2': float(result.prsquared) if hasattr(result, 'prsquared') and pd.notna(result.prsquared) else None,
                'aic': float(result.aic) if pd.notna(result.aic) else None, 
                'bic': float(result.bic) if pd.notna(result.bic) else None, 
                'log_likelihood': float(result.llf) if pd.notna(result.llf) else None,
                'coefficients': coeffs, 'p_values': pvals,
                'significant_features': significant_features,
                'model_type': 'OrderedModel (proportional odds)',
                'coding': category_map,
                'interpretation': ('With our coding (0=top, 4=bottom), negative coefficients indicate features that '
                                 'increase the probability of being in a better rank category (lower numerical value). '
                                 'Positive coefficients indicate features that increase the probability of being in a '
                                 'worse rank category (higher numerical value).')
            }
        except Exception as e:
            self.logger.error(f"Exception in ordinal regression fitting: {repr(e)}")
            return {'error': f'Failed to fit ordinal regression model: {str(e)}'}


    def _analyze_feature_importance_rq4(self, data: pd.DataFrame) -> Dict:
        # ... (Implementation from previous version) ...
        tech_features = [f for f in self.lighthouse_scores if f in data.columns]
        content_features = [f for f in self.content_metrics if f in data.columns]
        features = tech_features + content_features
        if not features:
            return {'error': 'No features for RF importance analysis'}

        if self.position_column not in data.columns or data[self.position_column].isna().all():
            return {'error': 'No position data for RF importance'}

        data_clean = data.dropna(subset=[self.position_column] + features)
        if len(data_clean) < 5: 
            return {'error': f'Insufficient data for RF after NaN removal: {len(data_clean)} rows'}
            
        X = data_clean[features]
        y = data_clean[self.position_column]
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        try:
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            importance_scores = dict(zip(features, rf.feature_importances_))
        except Exception as e:
            self.logger.error(f"RandomForestRegressor failed for RQ4: {repr(e)}")
            return {'error': str(e)}

        grouped_importance = {
            'technical_factors': {f: float(importance_scores[f]) for f in tech_features if f in importance_scores},
            'content_factors': {f: float(importance_scores[f]) for f in content_features if f in importance_scores}
        }
        
        tech_imp_sum = sum(grouped_importance['technical_factors'].values())
        content_imp_sum = sum(grouped_importance['content_factors'].values())
        total_imp = tech_imp_sum + content_imp_sum
        
        grouped_importance['aggregate'] = {
            'technical_importance_pct': float(tech_imp_sum / total_imp * 100) if total_imp > 0 else 0.0,
            'content_importance_pct': float(content_imp_sum / total_imp * 100) if total_imp > 0 else 0.0
        }
        return grouped_importance

    def _save_results(self):
        # ... (Implementation from previous version) ...
        self.logger.info("Saving analysis results...")
        self.stats['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        serializable_stats = self._convert_to_serializable(self.stats)
        
        stats_file_path = os.path.join(self.output_dir, 'analysis_results.json')
        try:
            with open(stats_file_path, 'w') as f:
                json.dump(serializable_stats, f, indent=4)
            self.logger.info(f"Full analysis results saved to: {stats_file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save JSON results: {repr(e)}")

        if self.df is not None:
            processed_df_path = os.path.join(self.output_dir, 'dataset_processed.csv')
            try:
                self.df.to_csv(processed_df_path, index=False)
                self.logger.info(f"Processed DataFrame saved to: {processed_df_path}")
            except Exception as e:
                self.logger.error(f"Failed to save processed DataFrame: {repr(e)}")
        
        self._generate_summary_report()


    def _convert_to_serializable(self, obj):
        # ... (Implementation from previous version) ...
        if isinstance(obj, dict):
            return {str(k): self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer, np.int64)): 
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)): 
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        elif pd.isna(obj): 
            return None 
        return obj 
        
    def _generate_summary_report(self):
        # ... (Implementation from previous version) ...
        report_file = os.path.join(self.output_dir, 'analysis_summary_text_report.txt')
        try:
            with open(report_file, 'w') as f:
                f.write("=== Web Content Analysis Summary (from analyze.py) ===\n\n")
                ds_info = self.stats.get('dataset_info', {})
                f.write(f"Input File: {self.input_file}\n")
                f.write(f"Initial Rows: {ds_info.get('initial_rows', 'N/A')}\n")
                f.write(f"Processed Rows: {ds_info.get('processed_rows', 'N/A')}\n")
                
                rq1 = self.stats.get('rq1_clustering', {})
                f.write("\n--- RQ1: Clustering ---\n")
                f.write(f"Optimal K: {rq1.get('selected_optimal_k', 'N/A')}\n")
                if 'cluster_stats' in rq1 and isinstance(rq1['cluster_stats'], dict):
                    for c_name, c_data in rq1['cluster_stats'].items():
                        if isinstance(c_data, dict): 
                           f.write(f"  {c_name}: Size={c_data.get('size', 'N/A')}, Percent={c_data.get('percentage', 0.0):.1f}%\n")

                rq4 = self.stats.get('rq4_factors', {})
                f.write("\n--- RQ4: Factor Priorities (Random Forest Aggregates) ---\n")
                for engine, data in rq4.items():
                    if engine == 'error' or not isinstance(data, dict): continue
                    f.write(f"Engine: {engine.capitalize()}\n")
                    agg = data.get('feature_importance_rf_stats', {}).get('aggregate', {})
                    if isinstance(agg, dict): 
                        f.write(f"  Technical Importance: {agg.get('technical_importance_pct', 0.0):.1f}%\n")
                        f.write(f"  Content Importance: {agg.get('content_importance_pct', 0.0):.1f}%\n")
            self.logger.info(f"Text summary report saved to: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {repr(e)}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive dataset analysis and save results to JSON.')
    parser.add_argument('--input', type=str, default='data/datasets/dataset.csv',
                        help='Input dataset CSV file')
    parser.add_argument('--output', type=str, default='data/analysis', 
                        help='Output directory for analysis results')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='Significance level for statistical tests')
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(
        input_file=args.input,
        output_dir=args.output,
        alpha=args.alpha
    )
    
    start_time = datetime.now()
    analyzer.run()
    duration = datetime.now() - start_time
    analyzer.logger.info(f"⏱️ Analysis completed in: {duration}")

if __name__ == "__main__":
    main()
