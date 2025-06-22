import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import logging
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# Add the parent directory to the path to import Utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.
    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.
    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

class PlotGenerator:
    def __init__(self, results_path: str, dataset_path: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger = Utils.set_colorful_logging('PlotGenerator')
        
        # Load analysis results
        try:
            with open(results_path, 'r') as f:
                self.stats = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading analysis results: {repr(e)}")
            self.stats = None

        # Load processed data
        try:
            self.df = pd.read_csv(dataset_path)
            self.logger.info(f"Loaded processed data with {len(self.df)} rows from {dataset_path}")
        except Exception as e:
            self.logger.error(f"Error loading processed data: {repr(e)}")
            self.df = None

        # --- User's original attributes are preserved ---
        self.sunset_sunrise_palette = ['#3b5b92', '#7a5195', '#ce5a8f', '#f0ad5f', '#e27850', '#cc4040', '#a42c33']
        self.feature_display_names = Utils.get_feature_display_names()
        self.technical_features = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        self.content_features = [col for col in self.feature_display_names.keys() if col not in self.technical_features]
        self.ranking_groups_order = ['high', 'medium', 'low']

        # Initialize additional attributes from stats
        if self.stats:
            dataset_info = self.stats.get('dataset_info', {})
            self.lighthouse_scores = dataset_info.get('lighthouse_metrics_found', [
                'performance_score', 'accessibility_score',
                'best-practices_score', 'seo_score'
            ])
            self.content_metrics = dataset_info.get('content_metrics_found', [
                'query_in_title', 'query_in_h1', 'exact_query_in_title',
                'exact_query_in_h1', 'query_density_body',
                'semantic_similarity_title_query',
                'semantic_similarity_content_query', 'word_count'
            ])
            self.ranking_groups = dataset_info.get('ranking_groups', {
                'high': (1, 5), 'medium': (6, 10), 'low': (11, 20)
            })
            self.position_column = dataset_info.get('position_column', 'position')
        else: 
            # Fallbacks if stats couldn't be loaded
            self.lighthouse_scores = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
            self.content_metrics = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1', 'query_density_body', 'semantic_similarity_title_query', 'semantic_similarity_content_query', 'word_count']
            self.ranking_groups = {'high': (1, 5), 'medium': (6, 10), 'low': (11, 20)}
            self.position_column = 'position'

        self.binary_metrics = [
            'query_in_title', 'query_in_h1',
            'exact_query_in_title', 'exact_query_in_h1'
        ]

    def generate_all_plots(self):
        """Generates all plots based on the analysis results, respecting original styles."""
        # --- USER: Insert your manual mapping here if needed ---
        # Example: self.correct_cluster_labels({0: 3, 1: 5, 2: 4, 3: 1, 4: 2, 5: 0})
        # Uncomment and edit the line below as needed:
        mapping = {0: 3, 1: 2, 2: 4, 3: 1, 4: 5, 5: 0}
        self.stats, self.df = Utils.correct_cluster_labels(self.stats, self.df, mapping, self.logger)
        
        # Cluster labels based on mapping analysis (same as table.py)
        # mapping: {old: new} -> cluster_labels: {new: label}
        self.cluster_names = {
            '0': 'Balanced Profile',      
            '1': 'Low Relevance',         
            '2': 'Content Focused',       
            '3': 'Mixed Performance',     
            '4': 'High Relevance',        
            '5': 'Technical Excellence'   
        }
        
        if not self.stats or self.df is None:
            self.logger.error("Cannot generate plots due to missing data or analysis results.")
            return

        self.logger.info("Starting plot generation...")
        
        # RQ1 plots
        self.plot_rq1_optimal_clusters()
        self.plot_rq1_cluster_radar()
        self.plot_rq1_cluster_pca()

        # RQ2 plots
        if 'engine' in self.df.columns:
            engines_present = [e for e in self.df['engine'].unique() if pd.notna(e)]
            for engine in engines_present:
                if engine in self.stats.get('rq2_visibility', {}):
                    self.plot_rq2_profile_distribution(engine)
                else:
                    self.logger.warning(f"No RQ2 visibility stats found for engine: {engine}")
        else:
            self.logger.error("'engine' column not found in DataFrame. Skipping engine-specific plots for RQ2.")
        
        # RQ3 plots
        self.plot_rq3_profile_comparison()
        
        # RQ4 plots
        self.plot_rq4_combined_correlation_heatmaps()
        self.plot_rq4_combined_feature_importance()
        
        self.logger.info(f"All plots successfully generated in '{self.output_dir}'")

    def plot_rq1_optimal_clusters(self):
        self.logger.info("Generating RQ1 optimal clusters plot...")
        if 'rq1_clustering' not in self.stats:
            self.logger.warning("Data for optimal clusters plot not found in stats.")
            return
        
        # Get silhouette scores from the new structure
        silhouette_scores = self.stats['rq1_clustering'].get('silhouette_scores', {})
        wcss = self.stats['rq1_clustering'].get('wcss', {})
        
        if not silhouette_scores:
            self.logger.warning("Silhouette scores missing for optimal_k plot.")
            return

        k_values_wcss = sorted([int(k) for k in wcss.keys()])
        wcss_values = [wcss[str(k)] for k in k_values_wcss]
        
        k_values_sil = sorted([int(k) for k in silhouette_scores.keys()])
        silhouette_values = [silhouette_scores[str(k)] for k in k_values_sil]
        
        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 4))
        
        # Add gray grid lines for better tracking of y-values
        ax1.grid(axis='y', linestyle='--', alpha=0.3, color='gray')

         # Use sunset-sunrise colors
        wcss_color = self.sunset_sunrise_palette[0]  # Dark blue
        silhouette_color = self.sunset_sunrise_palette[-2]  # Red (second to last)
        
        ax1.set_xlabel('Number of Clusters (K)')
        ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)', color=wcss_color)
        ax1.plot(k_values_wcss, wcss_values, marker='o', color=wcss_color, label='WCSS (Elbow)')
        ax1.tick_params(axis='y', labelcolor=wcss_color)
        ax1.set_ylim(bottom=min(wcss_values)*0.95 if wcss_values else 0, top=8000) # Set WCSS y-axis limit

        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Silhouette Score', color=silhouette_color)
        ax2.plot(k_values_sil, silhouette_values, marker='s', linestyle='--', color=silhouette_color, label='Silhouette Score')
        ax2.tick_params(axis='y', labelcolor=silhouette_color)
        
        plt.title('Optimal Cluster Number Determination') # Use normal title instead of suptitle
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='center right') 
        
        plt.tight_layout() 
        plt.savefig(os.path.join(self.output_dir, 'rq1_optimal_clusters_combined.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info("RQ1 optimal clusters plot generated.")

    def plot_rq1_cluster_radar(self):
        self.logger.info("Generating RQ1 cluster radar plot (normalized)...")
        if 'rq1_clustering' not in self.stats or 'cluster_stats' not in self.stats['rq1_clustering']:
            self.logger.warning("Data for radar plot not found in stats.")
            return

        cluster_data = self.stats['rq1_clustering']['cluster_stats']
        
        first_cluster_key = next(iter(cluster_data), None)
        if not first_cluster_key or 'feature_means_scaled' not in cluster_data[first_cluster_key] or not cluster_data[first_cluster_key]['feature_means_scaled']:
            self.logger.warning("Scaled feature means not found or empty for radar plot in the first cluster.")
            return
            
        features = list(cluster_data[first_cluster_key]['feature_means_scaled'].keys())
        if not features:
            self.logger.warning("No features found for radar plot based on scaled_feature_means.")
            return

        # Toplam feature sayısını al
        num_vars = len(features)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles_closed = angles + angles[:1] 

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar')) 
        
        # Use sunset-sunrise palette
        palette = self.sunset_sunrise_palette
        
        # Sort cluster_data by key (cluster number) to ensure consistent plotting order
        sorted_cluster_data = sorted(cluster_data.items(), key=lambda item: int(item[0]))

        # Plot each cluster with simple numeric labels
        for i, (cluster_name_key, stats_dict) in enumerate(sorted_cluster_data):
            if 'feature_means_scaled' in stats_dict and isinstance(stats_dict['feature_means_scaled'], dict):
                values = [stats_dict['feature_means_scaled'].get(f, 0.0) for f in features] 
                values += values[:1] 
                
                # Use the actual cluster number for color and label
                cluster_num = int(cluster_name_key)
                color_idx = cluster_num % len(palette)
                
                ax.plot(angles_closed, values, linewidth=2, linestyle='solid', 
                       # Use C-prefixed labels for clusters based on their actual number
                       label=f'C{cluster_num}',
                       color=palette[color_idx])
                ax.fill(angles_closed, values, alpha=0.2, color=palette[color_idx])
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles) # Use original angles for ticks
        
        # Use feature display names
        xticklabels = ax.set_xticklabels([self.feature_display_names.get(feature, feature) for feature in features], 
                                         fontsize=10, va='center')
        
        # Adjust label positions for better readability
        for i, label in enumerate(xticklabels):
            angle_rad = angles[i]
            
            # Top position (0 radians)
            if np.isclose(angle_rad, 0, atol=0.1):
                label.set_ha('center')
                label.set_va('bottom')
                label.set_rotation(0)
            # Bottom position (pi radians)
            elif np.isclose(angle_rad, np.pi, atol=0.1):
                label.set_ha('center')
                label.set_va('top')
                label.set_rotation(0)
            # Right side
            elif 0 < angle_rad < np.pi:
                label.set_ha('left')
                label.set_va('center')
                label.set_rotation(np.rad2deg(angle_rad + np.pi/2))
            # Left side
            else:
                label.set_ha('right')
                label.set_va('center')
                label.set_rotation(np.rad2deg(angle_rad + np.pi/2))

        ax.set_ylim(0, 1) 
        
        # Find index for 9 position (left) for radial labels
        left_idx = num_vars * 3 // 4  # 9 saat pozisyonu (sol)
        if left_idx < len(angles):
            angle_rad_left = angles[left_idx]
            ax.set_rlabel_position(np.degrees(angle_rad_left))
        else:
            # Fallback to default if feature doesn't exist
            ax.set_rlabel_position(0)
            
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=['0.2', '0.4', '0.6', '0.8'])
        plt.title("Cluster Profiles Radar Plot (Normalized Features)", size=14, y=1.08) 
        
        # Move legend to upper right
        ax.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rq1_cluster_radar_normalized.png'), bbox_inches='tight', dpi=300)
        plt.close(fig)
        self.logger.info("RQ1 cluster radar plot (normalized) generated.")

    def plot_rq1_cluster_pca(self):
        """
        Create PCA visualization of clusters to show their distribution in 2D space.
        This helps visualize how well-separated the clusters are and their internal structure.
        """
        if not isinstance(self.stats, dict):
            self.logger.error("Analysis results not loaded properly")
            return
            
        # Get the features from lighthouse scores and content metrics
        lighthouse_scores = [
            'performance_score',
            'accessibility_score',
            'best-practices_score',
            'seo_score'
        ]
        
        content_metrics = [
            'query_in_title',
            'query_in_h1',
            'exact_query_in_title',
            'exact_query_in_h1',
            'query_density_body',
            'semantic_similarity_title_query',
            'semantic_similarity_content_query',
            'word_count'
        ]
        
        # Get available features from the data
        available_features = [f for f in lighthouse_scores + content_metrics if f in self.df.columns]
        
        if not available_features:
            self.logger.warning("No features available for PCA visualization")
            return
            
        if 'cluster_label' not in self.df.columns:
            self.logger.warning("No cluster labels found in the data")
            return
            
        # Prepare feature matrix
        X = self.df[available_features].copy()
        
        # Handle missing values if any
        X = X.fillna(X.mean())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for 2D
        pca_2d = PCA(n_components=2)
        X_pca_2d = pca_2d.fit_transform(X_scaled)
        
        # Calculate PCA loadings for 2D
        loadings_2d = pd.DataFrame(
            pca_2d.components_.T,
            columns=['PC1', 'PC2'],
            index=available_features
        )
        
        # Get top contributing features for each component
        def get_top_features(loadings_df, pc_column, n=2):
            pos_features = loadings_df.nlargest(n, pc_column).index.tolist()
            neg_features = loadings_df.nsmallest(n, pc_column).index.tolist()
            pos_loadings = loadings_df.nlargest(n, pc_column)[pc_column].tolist()
            neg_loadings = loadings_df.nsmallest(n, pc_column)[pc_column].tolist()
            return pos_features, neg_features, pos_loadings, neg_loadings
        
        # Get interpretable axis labels for 2D
        pc1_pos, pc1_neg, pc1_pos_vals, pc1_neg_vals = get_top_features(loadings_2d, 'PC1')
        pc2_pos, pc2_neg, pc2_pos_vals, pc2_neg_vals = get_top_features(loadings_2d, 'PC2')
        
        # Create axis labels
        def create_axis_label(pos_features, neg_features, pos_vals, neg_vals, pc_num, var_ratio):
            pos_str = ' & '.join([f.replace('_', ' ').title() for f in pos_features])
            neg_str = ' & '.join([f.replace('_', ' ').title() for f in neg_features])
            return f"PC{pc_num} ({var_ratio:.1%} var.): {pos_str} (+)\nvs {neg_str} (-)"
        
        # Get number of unique clusters
        n_clusters = len(self.df['cluster_label'].unique())
        
        # Use sunset-sunrise palette for clusters
        colors = self.sunset_sunrise_palette[:n_clusters]
        
        # 2D Plot with interpretable labels
        plt.figure(figsize=(12, 6))
        
        # Plot each cluster in 2D
        for cluster_idx in range(n_clusters):
            mask = self.df['cluster_label'] == cluster_idx
            plt.scatter(
                X_pca_2d[mask, 0],
                X_pca_2d[mask, 1],
                c=[colors[cluster_idx]],
                label=f'Cluster {cluster_idx}',
                alpha=0.6,
                s=100
            )
            
            # Add cluster centroid
            centroid = X_pca_2d[mask].mean(axis=0)
            plt.scatter(
                centroid[0],
                centroid[1],
                c=[colors[cluster_idx]],
                marker='o',
                s=110,
                edgecolor='black',
                linewidth=1,
                label=f'Centroid {cluster_idx}'
            )
        
        # Calculate and display explained variance for 2D
        explained_var_ratio_2d = pca_2d.explained_variance_ratio_
        total_var_explained_2d = sum(explained_var_ratio_2d)
        
        # Add interpretable labels
        plt.xlabel(create_axis_label(pc1_pos, pc1_neg, pc1_pos_vals, pc1_neg_vals, 1, explained_var_ratio_2d[0]).replace('\n', ''))
        plt.ylabel(create_axis_label(pc2_pos, pc2_neg, pc2_pos_vals, pc2_neg_vals, 2, explained_var_ratio_2d[1]))
        plt.title(f'2D PCA Visualization of Clusters Total Variance Explained: {total_var_explained_2d:.1%}', 
                 pad=20)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout to accommodate multi-line y-axis labels
        plt.subplots_adjust(left=0.15, right=0.85, bottom=0.1, top=0.9)
        
        # Save 2D plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rq1_cluster_pca_2d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"PCA visualization generated with {len(available_features)} features and {n_clusters} clusters")

    def plot_rq2_profile_distribution(self, engine: str):
        self.logger.info(f"Generating RQ2 profile distribution plot for {engine}...")
        if self.df is None or 'cluster_label' not in self.df.columns:
            self.logger.warning(f"DataFrame or cluster labels missing for RQ2 profile distribution plot for {engine}.")
            return
        
        engine_data = self.df[self.df['engine'] == engine].copy()
        if engine_data.empty:
            self.logger.warning(f"No data for engine {engine} for RQ2 profile distribution.")
            return
        
        engine_data['ranking_group'] = engine_data[self.position_column].apply(self._get_ranking_group_from_stats)
        
        if 'ranking_group' not in engine_data.columns or 'cluster_label' not in engine_data.columns:
            self.logger.error("Ranking group or cluster label column missing after processing for RQ2 profile distribution.")
            return

        try:
            profile_dist = pd.crosstab(engine_data['ranking_group'], engine_data['cluster_label'])
            profile_dist = profile_dist.reindex(self.ranking_groups_order, axis=0).fillna(0)
            # Calculate percentage of each cluster within each ranking group
            profile_dist_pct = profile_dist.apply(lambda x: x / x.sum() * 100, axis=1).fillna(0) 
            
            # Store distribution for combined plot
            if not hasattr(self, 'profile_dist_data'):
                self.profile_dist_data = {}
            self.profile_dist_data[engine] = profile_dist_pct
            
            # Create combined plot if we have both engines
            if hasattr(self, 'profile_dist_data') and len(self.profile_dist_data) >= 2 and 'google' in self.profile_dist_data and 'bing' in self.profile_dist_data:
                self.plot_rq2_profile_distribution_combined()
                
            self.logger.info(f"RQ2 profile distribution plot for {engine} processed.")
            
        except Exception as e:
            self.logger.error(f"Error generating RQ2 profile distribution plot for {engine}: {repr(e)}")

    def plot_rq2_profile_distribution_combined(self):
        """Combines profile distribution plots for Google and Bing in one figure"""
        self.logger.info("Generating combined RQ2 profile distribution plot for Google and Bing...")
        
        try:
            if not hasattr(self, 'profile_dist_data') or 'google' not in self.profile_dist_data or 'bing' not in self.profile_dist_data:
                self.logger.warning("Missing data for combined profile distribution plot.")
                return
                
            # Get data from both engines
            google_data = self.profile_dist_data['google']
            bing_data = self.profile_dist_data['bing']
            
            # Get cluster labels (columns from either dataset)
            cluster_labels = sorted(set(google_data.columns) | set(bing_data.columns))
            
            # Create a single figure
            fig, ax = plt.subplots(figsize=(12, 4.5))
            
            # Add a light gray grid for easier tracking
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            
            # Define the group order: from low to high rank
            groups = []
            for rank in reversed(self.ranking_groups_order):  # reversed to get low, medium, high
                for engine in ['bing', 'google']:
                    groups.append(f"{engine}_{rank}")
            
            # Set up bar positions
            x = np.arange(len(groups))
            
            # Use sunset-sunrise palette
            colors = self.sunset_sunrise_palette[:len(cluster_labels)]
            
            # Transform data for stacked bars
            stacked_data = []
            for cluster in cluster_labels:
                cluster_values = []
                for group in groups:
                    engine, rank = group.split('_')
                    if engine == 'google':
                        if cluster in google_data.columns and rank in google_data.index:
                            cluster_values.append(google_data.loc[rank, cluster])
                        else:
                            cluster_values.append(0)
                    else:  # bing
                        if cluster in bing_data.columns and rank in bing_data.index:
                            cluster_values.append(bing_data.loc[rank, cluster])
                        else:
                            cluster_values.append(0)
                stacked_data.append(cluster_values)
            
            # Plot stacked bars
            bottom = np.zeros(len(groups))
            for i, cluster_values in enumerate(stacked_data):
                ax.bar(x, cluster_values, bottom=bottom, width=0.8, 
                       label=f'C{i}', color=colors[i % len(colors)])
                bottom += cluster_values
            
            # Add labels and title
            ax.set_ylabel('Percentage (%)')
            ax.set_xlabel('Engine and Ranking Group')
            ax.set_title('Profile Distribution Comparison Between Google and Bing')
            ax.set_xticks(x)
            
            # Create nicer x-tick labels (horizontal)
            x_labels = []
            for group in groups:
                engine, rank = group.split('_')
                x_labels.append(f"{engine.capitalize()} {rank.capitalize()}")
            ax.set_xticklabels(x_labels)  # No rotation
            
            ax.legend(title='Clusters', loc='upper right', bbox_to_anchor=(1, 1))
            
            # Add inner padding
            plt.subplots_adjust(left=0.1, right=1, bottom=0.15, top=0.9, wspace=0.3, hspace=1)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'rq2_profile_dist_combined_grouped_bar.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
            self.logger.info("Combined RQ2 profile distribution plot generated (single plot, stacked bars).")
            
        except Exception as e:
            self.logger.error(f"Error generating combined RQ2 profile distribution plot: {repr(e)}")

    def _get_ranking_group_from_stats(self, position: int) -> str:
        if not isinstance(position, (int, float, np.number)) or pd.isna(position):
             return 'other' 
        for group, (start, end) in self.ranking_groups.items():
            if start <= int(position) <= end:
                return group
        return 'other'

    def plot_rq3_profile_comparison(self):
        self.logger.info("Generating RQ3 profile comparison plot...")
        if self.df is None or 'cluster_label' not in self.df.columns:
            self.logger.warning("DataFrame or cluster labels missing for RQ3 profile comparison.")
            return
        
        top_n_ranks = self.ranking_groups['high'][1]
        top_results_df = self.df[self.df[self.position_column] <= top_n_ranks].copy()

        if top_results_df.empty or len(top_results_df['engine'].unique()) < 2:
            self.logger.warning("Not enough data for RQ3 profile comparison.")
            return

        try:
            profile_dist = pd.crosstab(top_results_df['engine'], top_results_df['cluster_label'])
            
            # Ensure cluster columns are sorted numerically for consistent plotting order
            profile_dist = profile_dist.reindex(sorted(profile_dist.columns), axis=1).fillna(0)

            profile_dist_pct = profile_dist.div(profile_dist.sum(axis=1), axis=0).mul(100).fillna(0)
            
            # Rename columns to use C0, C1, etc. format
            profile_dist_pct.columns = [f'C{col}' for col in profile_dist_pct.columns]
            
            # Create figure
            fig, ax = plt.subplots()
            
            # Add grid lines for better tracking
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='gray')
            
            # Use sunset-sunrise palette
            colors = self.sunset_sunrise_palette[:len(profile_dist_pct.columns)]
            
            # Plot bars
            profile_dist_pct.plot(kind='bar', figsize=(12, 4), rot=0, ax=ax, color=colors)
            
            plt.title('Profile Distribution Comparison in Top Positions (Ranks 1-5)')
            plt.xlabel('Search Engine')
            plt.ylabel('Percentage (%)')
            
            # Move legend inside the plot
            plt.legend(title='Clusters', loc='upper right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'rq3_profile_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info("RQ3 profile comparison plot generated.")
            
        except Exception as e:
            self.logger.error(f"Error generating RQ3 profile comparison plot: {repr(e)}")

    def plot_rq4_combined_correlation_heatmaps(self):
        self.logger.info("Generating combined RQ4 correlation heatmaps for Google and Bing...")
        if self.df is None:
            self.logger.warning("DataFrame missing for RQ4 correlation heatmaps.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(22, 10)) # Increased figsize for better annotation readability
        
        engines = ['google', 'bing']
        plot_successful = 0
        # Using a perceptually uniform diverging colormap, 'coolwarm' is a good default.
        cmap_to_use = 'coolwarm' 
        
        # Collect common features from both engines for consistent heatmap dimensions
        common_features = []
        engine_data_dict = {}
        
        # First, collect all available data
        for engine in engines:
            engine_data = self.df[self.df['engine'] == engine]
            if not engine_data.empty and self.position_column in engine_data.columns:
                features_for_corr = [f for f in self.lighthouse_scores + self.content_metrics if f in engine_data.columns]
                engine_data_dict[engine] = {
                    'data': engine_data,
                    'features': features_for_corr
                }
        
        # Find common features between both engines if both have data
        if len(engine_data_dict) == 2:
            common_features = sorted(list(set(engine_data_dict['google']['features']) & 
                                       set(engine_data_dict['bing']['features'])))
        
        # Now plot each engine
        for i, engine in enumerate(engines):
            ax = axes[i]
            
            # Skip if no data for this engine
            if engine not in engine_data_dict:
                self.logger.warning(f"No data for engine {engine} for RQ4 correlation heatmap.")
                ax.set_title(f'No Data for {engine.capitalize()}')
                ax.axis('off')
                continue
            
            engine_data = engine_data_dict[engine]['data']
            target_col = self.position_column
                
            # Use common features if available, otherwise use engine-specific features
            features_to_use = common_features if common_features else engine_data_dict[engine]['features']
            
            if not features_to_use:
                self.logger.warning(f"Not enough features for correlation heatmap for {engine}")
                ax.set_title(f'Feature/Target Missing ({engine.capitalize()})')
                ax.axis('off')
                continue

            columns_to_use = features_to_use + [target_col]
            numeric_engine_data = engine_data[columns_to_use].select_dtypes(include=np.number)
            numeric_engine_data = numeric_engine_data.dropna(axis=1, how='all')
            
            if numeric_engine_data.empty or len(numeric_engine_data.columns) < 2:
                self.logger.warning(f"Not enough numeric data for correlation heatmap for {engine}")
                ax.set_title(f'Insufficient Numeric Data ({engine.capitalize()})')
                ax.axis('off')
                continue

            # Rename columns with display names before correlation
            display_names_map = {col: self.feature_display_names.get(col, col) for col in numeric_engine_data.columns}
            display_names_map[target_col] = 'Position'  # Special case for target column
            numeric_engine_data = numeric_engine_data.rename(columns=display_names_map)
            
            corr_matrix = numeric_engine_data.corr(method='spearman')
            
            # Only show y-labels for the first plot (Google)
            yticklabels = True if i == 0 else False
            
            sns.heatmap(corr_matrix, annot=True, cmap=cmap_to_use, center=0, fmt='.2f', 
                        square=True, ax=ax, cbar=(i==1), 
                        annot_kws={"size": 7}, yticklabels=yticklabels) # square=True ensures square cells
                        
            ax.set_title(f'Spearman Correlation Heatmap ({engine.capitalize()})')
            
            # Rotate x-axis labels to 20 degrees and align to right
            ax.tick_params(axis='x', rotation=20, labelsize=8)
            # X etiketlerini sağa hizala
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment('right')
                
            # Y-etiketleri için boyut ayarla
            ax.tick_params(axis='y', rotation=0, labelsize=8)
            
            plot_successful +=1
        
        if plot_successful > 0:
            fig.suptitle('Feature Correlation with SERP Position (Spearman $\\rho$)', fontsize=16, y=1.02)
            
            for ax in axes:
                pos1 = axes[0].get_position()
                pos2 = axes[1].get_position()
                
                if pos1.height != pos2.height:
                    # Use the width of the first plot (Google) as reference
                    new_pos = [pos2.x0-0.06, pos1.y0, pos1.width, pos1.height]
                    axes[1].set_position(new_pos)
            
            
            plt.savefig(os.path.join(self.output_dir, 'rq4_correlation_heatmaps_combined.png'), dpi=300)
            self.logger.info("Combined RQ4 correlation heatmaps generated with consistent dimensions.")
        else:
            self.logger.warning("Could not generate any correlation heatmaps for RQ4.")
        plt.close(fig)

    def plot_rq4_combined_feature_importance(self):
        self.logger.info("Generating combined RQ4 feature importance plots for Google and Bing...")
        if 'rq4_factors' not in self.stats:
            self.logger.warning("RQ4 factor data not found in stats.")
            return

        engines = ['google', 'bing']
        all_feature_names_global = set()
        all_scores_data_global = {}

        for engine in engines:
            if engine in self.stats['rq4_factors'] and 'feature_importance_rf_stats' in self.stats['rq4_factors'][engine]:
                importance_data = self.stats['rq4_factors'][engine]['feature_importance_rf_stats']
                tech_factors = importance_data.get('technical_factors', {})
                content_factors = importance_data.get('content_factors', {})
                current_engine_scores = {**tech_factors, **content_factors}
                if current_engine_scores:
                    all_scores_data_global[engine] = current_engine_scores
                    all_feature_names_global.update(current_engine_scores.keys())
            else:
                self.logger.warning(f"Data for RQ4 feature importance plot not found for {engine}.")
        
        if not all_scores_data_global or not all_feature_names_global:
            self.logger.warning("No feature importance data found for any engine in RQ4 for combined plot.")
            return

        # Check if we have both engines
        if len(all_scores_data_global) < 2:
            self.logger.warning("Missing data for at least one engine, cannot create side-by-side comparison.")
            return 
            
        # Sort features by average importance across engines for consistent ordering
        avg_importances = {
            feat: np.mean([all_scores_data_global[eng].get(feat, 0) for eng in all_scores_data_global if feat in all_scores_data_global[eng]])
            for feat in all_feature_names_global
        }
        # Sort features in descending order of importance (for top-to-bottom display in horizontal bars)
        sorted_all_feature_names = sorted(list(all_feature_names_global), key=lambda f: avg_importances.get(f,0), reverse=True)

        # Create a single plot with side-by-side bars
        # fig, ax = plt.subplots(figsize=(8, max(6, len(sorted_all_feature_names) * 0.4)))
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Define bar positions
        y_pos = np.arange(len(sorted_all_feature_names))
        bar_width = 0.30  # Narrower bars to create more space between groups
        bar_gap = 0.05    # Add gap between bars in the same group
        
        # Add grid lines for x-axis
        ax.grid(axis='x', linestyle='--', alpha=0.3, color='gray')
        
        # Use sunset-sunrise palette - use first color for technical, last for content
        tech_color = self.sunset_sunrise_palette[0]  # Blue
        content_color = self.sunset_sunrise_palette[-1]  # Dark red
        
        # Define different opacity levels for engines instead of patterns
        opacity = [1.0, 0.65]  # Google full opacity, Bing more transparent
        
        # Determine categories for coloring (Technical or Content)
        categories = ['Technical' if name in self.lighthouse_scores else 'Content' for name in sorted_all_feature_names]
        
        # Plot bars for each engine side by side
        for i, engine in enumerate(engines):
            if engine in all_scores_data_global:
                scores = [all_scores_data_global[engine].get(feat, 0) for feat in sorted_all_feature_names]
                
                # Plot with different colors for Technical vs Content
                bars = ax.barh(
                    y_pos + (bar_width/2 + bar_gap/2) * (1 if i==0 else -1),  # Increased offset for more space
                    scores,
                    bar_width,
                    label=engine.capitalize(),
                    alpha=opacity[i],  # Use opacity instead of patterns
                    color=[tech_color if cat == 'Technical' else content_color for cat in categories],
                    # edgecolor='black' if i == 1 else None  # Add black edge to Bing bars for better distinction
                )

        # Labels and title
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self.feature_display_names.get(feat, feat) for feat in sorted_all_feature_names])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance Analysis for Predicting SERP Position', fontsize=12)
        
        # Create legend for both engine and category
        import matplotlib.patches as mpatches
        
        # Create custom patches for engine legend
        google_patch = mpatches.Patch(facecolor='gray', label='Google', alpha=opacity[0])
        bing_patch = mpatches.Patch(facecolor='gray', label='Bing', alpha=opacity[1])
        
        # Create custom patches for category legend
        technical_patch = mpatches.Patch(facecolor=tech_color, label='Technical', alpha=0.8)
        content_patch = mpatches.Patch(facecolor=content_color, label='Content', alpha=0.8)
  
        
        engine_legend = ax.legend(
            [google_patch, bing_patch], 
            ['Google', 'Bing'], 
            loc='upper right', 
            bbox_to_anchor=(1.0, 1),
            title="Engine"
        )
        ax.add_artist(engine_legend)
        category_legend = plt.legend(
            [technical_patch, content_patch], 
            ['Technical', 'Content'], 
            loc='upper right',
            bbox_to_anchor=(1.0, 0.8),
            title="Category"
        )
        ax.add_artist(category_legend)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rq4_feature_importance_combined.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        self.logger.info("Combined RQ4 feature importance plot generated with side-by-side comparison.")


def main():
    parser = argparse.ArgumentParser(description="Generate publication-quality plots from final analysis results.")
    # REPAIRED: Argument names are corrected to match the user's run.py script
    parser.add_argument('--results_path', type=str, default='data/analysis/analysis_results.json', help='Path to the analysis results JSON file.')
    parser.add_argument('--dataset_path', type=str, default='data/analysis/dataset_with_clusters.csv', help='Path to the dataset with cluster labels.')
    parser.add_argument('--output_dir', type=str, default='figs', help='Directory to save the generated plots.')
    args = parser.parse_args()
    
    plotter = PlotGenerator(
        results_path=args.results_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
    plotter.generate_all_plots()

if __name__ == "__main__":
    main()