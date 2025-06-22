import pandas as pd
import colorlog
import logging
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from urllib.parse import urlparse


class Utils:
    def __init__(self):
        self.url_index_map = {}
        
    @staticmethod
    def plot_correlation_heatmap(
        data: pd.DataFrame,
        features: List[str],
        target_column: str,
        title: str,
        output_path: str,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = 'coolwarm',
        method: str = 'spearman'
    ) -> None:
        """
        Create and save a correlation heatmap for specified features.
        
        Args:
            data: DataFrame containing the data
            features: List of feature columns to include
            target_column: Target variable column
            title: Title for the plot
            output_path: Path to save the plot
            figsize: Figure size as (width, height)
            cmap: Colormap to use
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        # Filter to only include columns that exist in the dataframe
        available_features = [f for f in features if f in data.columns]
        
        if not available_features or target_column not in data.columns:
            logger = logging.getLogger("Utils")
            logger.warning(f"Cannot create correlation heatmap - missing required columns")
            return
            
        # Calculate correlation matrix
        columns_to_use = available_features + [target_column]
        corr_matrix = data[columns_to_use].corr(method=method)
        
        # Create figure
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            center=0,
            fmt='.2f',
            square=True
        )
        
        plt.title(title)
        plt.tight_layout()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path)
        plt.close()
    
    @staticmethod
    def plot_feature_importance(
        feature_names: List[str],
        importance_scores: List[float],
        categories: Optional[List[str]] = None,
        title: str = 'Feature Importance',
        output_path: str = None,
        figsize: Tuple[int, int] = (12, 8),
        color_map: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Create and save a horizontal bar chart of feature importance.
        
        Args:
            feature_names: List of feature names
            importance_scores: List of importance scores corresponding to features
            categories: Optional list of categories for each feature (for coloring)
            title: Title for the plot
            output_path: Path to save the plot (if None, just displays the plot)
            figsize: Figure size as (width, height)
            color_map: Dictionary mapping categories to colors
        """
        if len(feature_names) == 0 or len(importance_scores) == 0:
            logger = logging.getLogger("Utils")
            logger.warning("No features to plot importance")
            return
            
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        })
        
        # Add categories if provided
        if categories:
            if len(categories) != len(feature_names):
                logger = logging.getLogger("Utils")
                logger.warning("Categories length doesn't match feature names length. Ignoring categories.")
            else:
                plot_data['Category'] = categories
                
        # Sort by importance
        plot_data = plot_data.sort_values('Importance', ascending=True)
        
        # Set up default color map if needed
        if categories and not color_map:
            unique_categories = list(set(categories))
            default_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
            color_map = {
                cat: default_colors[i % len(default_colors)]
                for i, cat in enumerate(unique_categories)
            }
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Draw the bar plot
        if categories and 'Category' in plot_data.columns:
            sns.barplot(
                x='Importance',
                y='Feature',
                hue='Category',
                data=plot_data,
                palette=color_map
            )
        else:
            sns.barplot(
                x='Importance',
                y='Feature',
                data=plot_data,
                color='#3498db'
            )
        
        plt.title(title)
        plt.xlabel('Importance Score')
        plt.tight_layout()
        
        # Save or display
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_grouped_boxplots(
        data: pd.DataFrame,
        features: List[str],
        group_column: str,
        order: Optional[List[str]] = None,
        title: str = 'Feature Comparison by Group',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        ncols: int = 2
    ) -> None:
        """
        Create boxplots for multiple features grouped by a column.
        
        Args:
            data: DataFrame containing the data
            features: List of feature columns to plot
            group_column: Column to group by
            order: Optional ordering of groups
            title: Plot title
            output_path: Path to save the plot (if None, just displays the plot)
            figsize: Figure size as (width, height)
            ncols: Number of columns in the subplot grid
        """
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in features if f in data.columns]
        
        if not available_features or group_column not in data.columns:
            logger = logging.getLogger("Utils")
            logger.warning(f"Cannot create boxplots - missing required columns")
            return
            
        # Calculate rows needed
        n_features = len(available_features)
        nrows = (n_features + ncols - 1) // ncols
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Create each subplot
        for i, feature in enumerate(available_features):
            if i < len(axes):
                sns.boxplot(
                    x=group_column,
                    y=feature,
                    data=data,
                    order=order,
                    ax=axes[i]
                )
                axes[i].set_title(feature)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save or display
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_binary_feature_bar(
        data: pd.DataFrame, 
        binary_features: List[str],
        group_column: str,
        order: Optional[List[str]] = None,
        title: str = 'Binary Feature Distribution by Group',
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10),
        ncols: int = 2
    ) -> None:
        """
        Create bar charts showing percentage of binary features by group.
        
        Args:
            data: DataFrame containing the data
            binary_features: List of binary feature columns to plot
            group_column: Column to group by
            order: Optional ordering of groups
            title: Plot title
            output_path: Path to save the plot (if None, just displays the plot)
            figsize: Figure size as (width, height)
            ncols: Number of columns in the subplot grid
        """
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in binary_features if f in data.columns]
        
        if not available_features or group_column not in data.columns:
            logger = logging.getLogger("Utils")
            logger.warning(f"Cannot create binary feature bar charts - missing required columns")
            return
            
        # Calculate rows needed
        n_features = len(available_features)
        nrows = (n_features + ncols - 1) // ncols
        
        # Create figure
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows == 1 and ncols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Use default order if none provided
        if order is None:
            order = sorted(data[group_column].unique())
        
        # Create each subplot
        for i, feature in enumerate(available_features):
            if i < len(axes):
                ax = axes[i]
                
                # Calculate percentage for each group
                percentages = []
                for group in order:
                    if group in data[group_column].values:
                        subset = data[data[group_column] == group]
                        pct = subset[feature].mean() * 100
                        percentages.append(pct)
                    else:
                        percentages.append(0)
                
                # Plot bars
                ax.bar(range(len(order)), percentages)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order)
                ax.set_ylabel('Percentage Present (%)')
                ax.set_title(feature)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # Save or display
        if output_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def clean_filename(text: str, page: int = 0) -> str:
        """Clean a string to make it suitable for use as a filename.
        
        Args:
            text: The input string to clean
            page: Optional page number to append to filename (default: 0)
            
        Returns:
            A cleaned string with:
            - Leading/trailing whitespace removed
            - Special characters removed except alphanumeric and underscores
            - Internal spaces replaced with underscores
            - Converted to lowercase
            - Page number appended if provided
        """
        # First trim whitespace
        text = text.strip()
        
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text)
        
        # Remove any special characters except alphanumeric and underscores
        text = re.sub(r'[^\w\s-]', '', text)
        
        # Convert to lowercase
        text = text.lower() 
        
        if page > 0:
            text = text + '_' + str(page)
        
        return text
    
    @staticmethod
    def get_hostname(url):
        if pd.isnull(url):
            return None
        try:
            if isinstance(url, str) and not url.startswith(('http://', 'https://')):
                url_to_parse = 'http://' + url 
            else:
                url_to_parse = url
            
            parsed_url = urlparse(url_to_parse)
            hostname = parsed_url.netloc
            if hostname.startswith('www.'):
                hostname = hostname[4:]
            return hostname if hostname else None
        except Exception:
            return None
    
    @staticmethod
    def set_colorful_logging(name: str) -> logging.Logger:
        """Setup colorful logging for the given logger name"""
        logger = logging.getLogger(name)
            
        handler = colorlog.StreamHandler()
        handler.setFormatter(colorlog.ColoredFormatter(
            fmt='%(log_color)s%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger 

    @staticmethod
    def get_unique_urls(input_file: str):
        """Load websites and create index mapping"""
        try:
            df = pd.read_csv(input_file, low_memory=False)
            return df['link'].unique().tolist()
        except Exception as e:
            print(f"⚠️ Error loading SERPs file: {str(e)}")
        return df

    @staticmethod
    def url_to_safe_filename(url: str) -> str:
        """Convert a URL to a safe filename, preserving important parts of the URL structure.
        
        Args:
            url: The URL to convert
            
        Returns:
            A filename-safe string that preserves URL structure
        """
        # Remove protocol
        url = re.sub(r'^https?://', '', url)
        
        # Replace special characters and separators with underscores
        url = re.sub(r'[/\?=&%]', '_', url)
        
        # Remove any other unsafe filename characters
        url = re.sub(r'[<>:"|*]', '', url)
        
        # Collapse multiple underscores
        url = re.sub(r'_+', '_', url)
        
        # Remove leading/trailing underscores
        url = url.strip('_')
        
        # Convert to lowercase
        url = url.lower()
        
        return url

    @staticmethod
    def get_feature_display_names():
        """Centralized feature display names mapping to eliminate duplication across modules"""
        return {
            'performance_score': 'Perf.',
            'accessibility_score': 'Access.',
            'best-practices_score': 'Best Prac.',
            'seo_score': 'SEO',
            'query_in_title': 'Q-Title',
            'query_in_h1': 'Q-H1',
            'exact_query_in_title': 'ExQ-Title',
            'exact_query_in_h1': 'ExQ-H1',
            'query_density_body': 'Q/B Density',
            'semantic_similarity_title_query': 'Sim. Title',
            'semantic_similarity_content_query': 'Sim. Content',
            'word_count': 'Word Count'
        }
    
    @staticmethod
    def calculate_medians_by_engine(df: pd.DataFrame, features: list, engine_col: str = 'engine', 
                                   position_col: str = 'position', top_n: int = 5) -> dict:
        """
        Calculate median values for features by engine for top-ranking pages
        
        Args:
            df: DataFrame with the data
            features: List of feature columns to calculate medians for
            engine_col: Column name for engine (default: 'engine')
            position_col: Column name for position (default: 'position')
            top_n: Number of top positions to consider (default: 5)
            
        Returns:
            Dictionary with structure: {engine: {feature: median_value}}
        """
        if df is None or df.empty:
            return {}
            
        medians = {}
        
        for engine in df[engine_col].unique():
            if pd.isna(engine):
                continue
                
            # Filter for top N positions for this engine
            engine_data = df[(df[engine_col] == engine) & (df[position_col] <= top_n)]
            
            if engine_data.empty:
                continue
                
            engine_medians = {}
            for feature in features:
                if feature in engine_data.columns:
                    # Calculate median, handling NaN values
                    median_val = engine_data[feature].median()
                    engine_medians[feature] = median_val if not pd.isna(median_val) else 0.0
                else:
                    engine_medians[feature] = 0.0
                    
            medians[engine] = engine_medians
            
        return medians

    @staticmethod
    def correct_cluster_labels(stats: dict, df: pd.DataFrame, mapping: dict, logger: logging.Logger) -> Tuple[dict, pd.DataFrame]:
        """
        Corrects cluster labels in both the analysis statistics dictionary and the DataFrame.
        This centralized function ensures consistency across plots and tables.
        
        Args:
            stats: The main dictionary containing analysis results.
            df: The DataFrame containing the dataset with 'cluster_label'.
            mapping: A dictionary to map old cluster labels to new ones, e.g., {0: 3, 1: 5, ...}.
            logger: The logger instance for logging messages.
            
        Returns:
            A tuple containing the modified stats dictionary and DataFrame.
        """
        # 1. Remap cluster labels in the DataFrame
        if df is not None and 'cluster_label' in df.columns:
            df['cluster_label'] = df['cluster_label'].map(mapping)
            logger.info(f"DataFrame cluster labels corrected using mapping: {mapping}")
        
        # 2. Remap cluster labels in the statistics dictionary
        if stats and 'rq1_clustering' in stats:
            cluster_results = stats['rq1_clustering']
            
            # Remap 'cluster_stats' keys
            if 'cluster_stats' in cluster_results:
                old_stats = cluster_results['cluster_stats']
                new_stats = {}
                for old, new in mapping.items():
                    if str(old) in old_stats:
                        new_stats[str(new)] = old_stats[str(old)]
                cluster_results['cluster_stats'] = new_stats
                logger.info(f"Stats 'cluster_stats' keys corrected using mapping: {mapping}")
            
            # Remap 'per_cluster_silhouette' keys in 'final_clustering_results'
            if 'final_clustering_results' in cluster_results and 'per_cluster_silhouette' in cluster_results['final_clustering_results']:
                old_per_cluster = cluster_results['final_clustering_results']['per_cluster_silhouette']
                new_per_cluster = {}
                for old, new in mapping.items():
                    old_key = f'cluster_{old}'
                    if old_key in old_per_cluster:
                        new_per_cluster[f'cluster_{new}'] = old_per_cluster[old_key]
                cluster_results['final_clustering_results']['per_cluster_silhouette'] = new_per_cluster
                logger.info(f"Stats 'per_cluster_silhouette' keys corrected using mapping: {mapping}")
                
        return stats, df