import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json
import argparse
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from typing import Dict, List, Tuple

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

class OutlierAnalyzer:
    def __init__(self,
                 input_file: str,
                 output_dir: str,
                 skip_plots: bool = False):
        self.input_file = input_file
        self.output_dir = output_dir
        self.skip_plots = skip_plots
        self.logger = Utils.set_colorful_logging('OutlierAnalyzer')
        
        # Define numerical columns to analyze
        self.numerical_columns = [
            'serp_position',
            'performance_score',
            'accessibility_score',
            'best-practices_score',
            'seo_score',
            'pwa_score',
            # Add new feature columns
            'query_in_title',
            'query_in_h1',
            'exact_query_in_title',
            'exact_query_in_h1',
            'query_density_body',
            'semantic_similarity_title_query',
            'semantic_similarity_content_query',
            'word_count'
        ]
        
        # Define categorical columns to analyze
        self.categorical_columns = [
            'engine',
            'hostname'
        ]
        
        # Define score thresholds for recommendations
        self.thresholds = {
            'performance_score': 90,
            'accessibility_score': 90,
            'best-practices_score': 90,
            'seo_score': 90
        }
        
        # Initialize stats dictionary
        self.stats = {
            'total_rows': 0,
            'missing_values': {},
            'outliers': {},
            'distributions': {},
            'categorical_stats': {},
            'correlations': {}
        }
        
        # Initialize insights dictionary
        self.insights = {
            'dataset_overview': {},
            'quality_metrics': {},
            'distribution_analysis': {},
            'correlation_insights': {},
            'recommendations': []
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run(self):
        """Run the outlier and pilot analysis"""
        # Read the dataset
        self.logger.info("Reading dataset...")
        self.df = pd.read_csv(self.input_file)
        self.stats['total_rows'] = len(self.df)
        
        # Extract hostname from link
        if 'url' in self.df.columns:
            self.df['hostname'] = self.df['url'].apply(lambda x: Utils.get_hostname(x) if pd.notnull(x) else None)
        
        # Run basic analysis
        self.analyze_missing_values(self.df)
        self.analyze_numerical_features(self.df)
        self.analyze_categorical_features(self.df)
        self.analyze_correlations(self.df)
        
        # Run pilot analysis
        self.analyze_overview()
        self.analyze_quality_metrics()
        self.analyze_distributions()
        self.generate_recommendations()
        
        # Generate plots if not skipped
        if not self.skip_plots:
            try:
                self.generate_plots(self.df)
            except Exception as e:
                self.logger.error(f"Failed to generate plots: {str(e)}")
                self.logger.warning("Continuing without plots...")
        
        # Save results
        self.save_results()
        
    def analyze_missing_values(self, df: pd.DataFrame):
        """Analyze missing values in the dataset"""
        self.logger.info("Analyzing missing values...")
        
        missing_stats = {}
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_stats[column] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(df) * 100, 2)
                }
        
        self.stats['missing_values'] = missing_stats
        
    def analyze_numerical_features(self, df: pd.DataFrame):
        """Analyze numerical features for outliers and distributions"""
        self.logger.info("Analyzing numerical features...")
        
        for column in self.numerical_columns:
            if column not in df.columns:
                continue
                
            # Get basic statistics
            data = df[column].dropna()
            
            # Skip if no data available
            if len(data) == 0:
                self.logger.warning(f"No data available for column: {column}")
                self.stats['outliers'][column] = {
                    'iqr_outliers_count': 0,
                    'iqr_outliers_percentage': 0,
                    'z_score_outliers_count': 0,
                    'z_score_outliers_percentage': 0,
                    'lower_bound': None,
                    'upper_bound': None
                }
                self.stats['distributions'][column] = {
                    'mean': None,
                    'median': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'skewness': None,
                    'kurtosis': None
                }
                continue
            
            # Calculate IQR outliers
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Calculate z-score outliers
            z_scores = np.abs(stats.zscore(data))
            z_score_outliers = data[z_scores > 3]
            
            self.stats['outliers'][column] = {
                'iqr_outliers_count': len(outliers),
                'iqr_outliers_percentage': round(len(outliers) / len(data) * 100, 2),
                'z_score_outliers_count': len(z_score_outliers),
                'z_score_outliers_percentage': round(len(z_score_outliers) / len(data) * 100, 2),
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            }
            
            # Calculate distribution statistics
            self.stats['distributions'][column] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'min': float(data.min()),
                'max': float(data.max()),
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            }
            
    def analyze_categorical_features(self, df: pd.DataFrame):
        """Analyze categorical features"""
        self.logger.info("Analyzing categorical features...")
        
        for column in self.categorical_columns:
            if column not in df.columns:
                continue
                
            # Get non-null values
            data = df[column].dropna()
            
            # Skip if no data available
            if len(data) == 0:
                self.logger.warning(f"No data available for column: {column}")
                self.stats['categorical_stats'][column] = {
                    'unique_values': 0,
                    'top_10_values': {},
                    'top_10_percentages': {}
                }
                continue
                
            value_counts = data.value_counts()
            unique_count = len(value_counts)
            
            self.stats['categorical_stats'][column] = {
                'unique_values': unique_count,
                'top_10_values': value_counts.head(10).to_dict(),
                'top_10_percentages': (value_counts.head(10) / len(data) * 100).round(2).to_dict()
            }
            
    def analyze_correlations(self, df: pd.DataFrame):
        """Analyze correlations between numerical features"""
        self.logger.info("Analyzing correlations...")
        
        # Get numerical columns that exist in the dataset and have data
        numerical_cols = [col for col in self.numerical_columns 
                         if col in df.columns and len(df[col].dropna()) > 0]
        
        if len(numerical_cols) > 1:
            # Drop rows where all numerical columns are null
            data = df[numerical_cols].dropna()
            if len(data) > 0:
                corr_matrix = data.corr()
                
                # Convert to dictionary format
                correlations = {}
                for col1 in corr_matrix.columns:
                    correlations[col1] = {}
                    for col2 in corr_matrix.columns:
                        if col1 != col2:
                            correlations[col1][col2] = round(corr_matrix.loc[col1, col2], 3)
                
                self.stats['correlations'] = correlations
            else:
                self.logger.warning("No data available for correlation analysis")
                self.stats['correlations'] = {}
        else:
            self.logger.warning("Not enough numerical columns for correlation analysis")
            self.stats['correlations'] = {}
            
    def analyze_overview(self):
        """Analyze dataset overview"""
        self.logger.info("Analyzing dataset overview...")
        
        overview = {
            'total_rows': self.stats['total_rows'],
            'search_engines': {
                'google': len(self.df[self.df['engine'] == 'google']),
                'bing': len(self.df[self.df['engine'] == 'bing'])
            }
        }
        
        # Add hostname stats if available
        if 'hostname' in self.df.columns:
            overview['unique_hostnames'] = len(self.df['hostname'].dropna().unique())
        else:
            overview['unique_hostnames'] = 0
            
        # Add SERP position range if available
        if 'serp_position' in self.df.columns:
            overview['serp_position_range'] = {
                'min': float(self.df['serp_position'].min()),
                'max': float(self.df['serp_position'].max())
            }
        else:
            overview['serp_position_range'] = {'min': None, 'max': None}
            
        overview['missing_values'] = self.stats['missing_values']
        
        self.insights['dataset_overview'] = overview
        
    def analyze_quality_metrics(self):
        """Analyze quality metrics"""
        self.logger.info("Analyzing quality metrics...")
        
        quality_metrics = {}
        
        # Analyze PageSpeed metrics
        for metric in ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    quality_metrics[metric] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'below_threshold': len(data[data < self.thresholds[metric]]),
                        'below_threshold_percentage': round(len(data[data < self.thresholds[metric]]) / len(data) * 100, 2),
                        'distribution': self.stats['distributions'].get(metric, {})
                    }
        
        # Analyze content relevance metrics
        relevance_metrics = ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1']
        for metric in relevance_metrics:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    present_count = len(data[data > 0])
                    quality_metrics[metric] = {
                        'mean': float(data.mean()),
                        'present_count': present_count,
                        'present_percentage': round(present_count / len(data) * 100, 2),
                        'distribution': self.stats['distributions'].get(metric, {})
                    }
        
        # Analyze query density and word count
        for metric in ['query_density_body', 'word_count']:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    quality_metrics[metric] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'distribution': self.stats['distributions'].get(metric, {})
                    }
        
        # Analyze semantic similarity metrics
        similarity_metrics = ['semantic_similarity_title_query', 'semantic_similarity_content_query']
        for metric in similarity_metrics:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    high_similarity_count = len(data[data >= 0.5])
                    quality_metrics[metric] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'high_similarity_count': high_similarity_count,
                        'high_similarity_percentage': round(high_similarity_count / len(data) * 100, 2),
                        'distribution': self.stats['distributions'].get(metric, {})
                    }
        
        self.insights['quality_metrics'] = quality_metrics
        
    def analyze_distributions(self):
        """Analyze score distributions"""
        self.logger.info("Analyzing score distributions...")
        
        distribution_insights = {}
        
        # Analyze SERP position distribution
        if 'serp_position' in self.df.columns:
            distribution_insights['serp_position'] = {
                'top_3_percentage': round(len(self.df[self.df['serp_position'] <= 3]) / len(self.df) * 100, 2),
                'top_5_percentage': round(len(self.df[self.df['serp_position'] <= 5]) / len(self.df) * 100, 2),
                'top_10_percentage': round(len(self.df[self.df['serp_position'] <= 10]) / len(self.df) * 100, 2)
            }
        
        # Analyze PageSpeed score distributions
        for metric in ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    distribution_insights[metric] = {
                        'excellent': round(len(data[data >= 90]) / len(data) * 100, 2),
                        'good': round(len(data[(data >= 70) & (data < 90)]) / len(data) * 100, 2),
                        'fair': round(len(data[(data >= 50) & (data < 70)]) / len(data) * 100, 2),
                        'poor': round(len(data[data < 50]) / len(data) * 100, 2)
                    }
        
        # Analyze query presence metrics
        for metric in ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1']:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    distribution_insights[metric] = {
                        'present_percentage': round(len(data[data > 0]) / len(data) * 100, 2),
                        'absent_percentage': round(len(data[data == 0]) / len(data) * 100, 2)
                    }
        
        # Analyze semantic similarity distributions
        for metric in ['semantic_similarity_title_query', 'semantic_similarity_content_query']:
            if metric in self.df.columns:
                data = self.df[metric].dropna()
                if len(data) > 0:
                    distribution_insights[metric] = {
                        'high_similarity': round(len(data[data >= 0.7]) / len(data) * 100, 2),
                        'medium_similarity': round(len(data[(data >= 0.4) & (data < 0.7)]) / len(data) * 100, 2),
                        'low_similarity': round(len(data[data < 0.4]) / len(data) * 100, 2)
                    }
        
        # Analyze word count distribution
        if 'word_count' in self.df.columns:
            data = self.df['word_count'].dropna()
            if len(data) > 0:
                distribution_insights['word_count'] = {
                    'short_content': round(len(data[data < 300]) / len(data) * 100, 2),
                    'medium_content': round(len(data[(data >= 300) & (data < 1000)]) / len(data) * 100, 2),
                    'long_content': round(len(data[(data >= 1000) & (data < 3000)]) / len(data) * 100, 2),
                    'very_long_content': round(len(data[data >= 3000]) / len(data) * 100, 2)
                }
        
        # Analyze query density distribution
        if 'query_density_body' in self.df.columns:
            data = self.df['query_density_body'].dropna()
            if len(data) > 0:
                distribution_insights['query_density_body'] = {
                    'low_density': round(len(data[data < 0.5]) / len(data) * 100, 2),
                    'medium_density': round(len(data[(data >= 0.5) & (data < 2)]) / len(data) * 100, 2),
                    'high_density': round(len(data[data >= 2]) / len(data) * 100, 2)
                }
        
        self.insights['distribution_analysis'] = distribution_insights
        
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        self.logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Missing values recommendations
        for column, missing in self.stats['missing_values'].items():
            if missing['percentage'] > 1:
                recommendations.append({
                    'type': 'missing_values',
                    'metric': column,
                    'description': f"Handle {missing['percentage']}% missing values in {column}",
                    'suggestion': "Consider imputation or removal based on analysis needs"
                })
        
        # Outlier recommendations
        for column, outliers in self.stats['outliers'].items():
            if outliers.get('iqr_outliers_percentage', 0) > 5:
                recommendations.append({
                    'type': 'outliers',
                    'metric': column,
                    'description': f"Review {outliers['iqr_outliers_percentage']}% outliers in {column}",
                    'suggestion': "Investigate outlier values and consider treatment methods"
                })
        
        # Quality score recommendations for PageSpeed metrics
        for metric, stats in self.insights.get('quality_metrics', {}).items():
            if metric in self.thresholds and 'below_threshold_percentage' in stats:
                if stats['below_threshold_percentage'] > 20:
                    recommendations.append({
                        'type': 'quality',
                        'metric': metric,
                        'description': f"{stats['below_threshold_percentage']}% of URLs have {metric} below {self.thresholds[metric]}",
                        'suggestion': "Consider filtering or analyzing these low-scoring URLs separately"
                    })
        
        # Query presence recommendations
        for metric in ['query_in_title', 'query_in_h1']:
            if metric in self.df.columns and metric in self.insights.get('quality_metrics', {}):
                stats = self.insights['quality_metrics'][metric]
                if stats.get('present_percentage', 0) < 40:
                    recommendations.append({
                        'type': 'content_relevance',
                        'metric': metric,
                        'description': f"Only {stats['present_percentage']}% of URLs have the query in their {metric.split('_')[-1]}",
                        'suggestion': "Content relevance may be low; consider analyzing these pages for optimization opportunities"
                    })
        
        # Semantic similarity recommendations
        for metric in ['semantic_similarity_title_query', 'semantic_similarity_content_query']:
            if metric in self.df.columns and metric in self.insights.get('quality_metrics', {}):
                stats = self.insights['quality_metrics'][metric]
                if stats.get('high_similarity_percentage', 0) < 30:
                    element = 'title' if 'title' in metric else 'content'
                    recommendations.append({
                        'type': 'semantic_similarity',
                        'metric': metric,
                        'description': f"Only {stats['high_similarity_percentage']}% of URLs have high semantic similarity between {element} and query",
                        'suggestion': f"Consider enhancing {element} relevance to the search query"
                    })
        
        # Word count recommendations
        if 'word_count' in self.df.columns and 'word_count' in self.insights.get('distribution_analysis', {}):
            word_count_dist = self.insights['distribution_analysis']['word_count']
            if word_count_dist.get('short_content', 0) > 50:
                recommendations.append({
                    'type': 'content_length',
                    'metric': 'word_count',
                    'description': f"{word_count_dist['short_content']}% of URLs have short content (less than 300 words)",
                    'suggestion': "Short content may impact search performance; consider expanding or filtering out these pages"
                })
        
        # Distribution recommendations for SERP position
        if 'serp_position' in self.insights.get('distribution_analysis', {}):
            if self.insights['distribution_analysis']['serp_position']['top_3_percentage'] < 30:
                recommendations.append({
                    'type': 'distribution',
                    'metric': 'serp_position',
                    'description': "Low representation of top 3 SERP positions",
                    'suggestion': "Consider balancing the dataset for better position representation"
                })
        
        # Query density recommendations
        if 'query_density_body' in self.df.columns and 'query_density_body' in self.insights.get('distribution_analysis', {}):
            query_density_dist = self.insights['distribution_analysis']['query_density_body']
            if query_density_dist.get('low_density', 0) > 70:
                recommendations.append({
                    'type': 'query_density',
                    'metric': 'query_density_body',
                    'description': f"{query_density_dist['low_density']}% of URLs have low query density",
                    'suggestion': "Consider analyzing these pages for content optimization opportunities"
                })
        
        self.insights['recommendations'] = recommendations
        
    def generate_plots(self, df: pd.DataFrame):
        """Generate visualization plots"""
        if not HAS_SEABORN:
            self.logger.warning("Seaborn not installed. Plots will use basic matplotlib style.")
        
        self.logger.info("Generating plots...")
        
        # Set style
        if HAS_SEABORN:
            sns.set_style("whitegrid")
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Box plots for numerical features
        for column in self.numerical_columns:
            if column not in df.columns:
                continue
                
            data = df[column].dropna()
            if len(data) == 0:
                self.logger.warning(f"Skipping plot for {column}: no data available")
                continue
                
            plt.figure(figsize=(10, 6))
            if HAS_SEABORN:
                sns.boxplot(x=data)
            else:
                plt.boxplot(data)
            plt.title(f'Box Plot of {column}')
            plt.savefig(os.path.join(plots_dir, f'boxplot_{column}.png'))
            plt.close()
            
        # 2. Correlation heatmap
        numerical_cols = [col for col in self.numerical_columns 
                         if col in df.columns and len(df[col].dropna()) > 0]
        if len(numerical_cols) > 1:
            data = df[numerical_cols].dropna()
            if len(data) > 0:
                plt.figure(figsize=(12, 8))
                corr_matrix = data.corr()
                if HAS_SEABORN:
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
                else:
                    plt.imshow(corr_matrix, cmap='coolwarm', aspect='equal')
                    plt.colorbar()
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
                plt.close()
            
        # 3. Distribution plots for numerical features
        for column in self.numerical_columns:
            if column not in df.columns:
                continue
                
            data = df[column].dropna()
            if len(data) == 0:
                self.logger.warning(f"Skipping plot for {column}: no data available")
                continue
                
            plt.figure(figsize=(10, 6))
            if HAS_SEABORN:
                sns.histplot(data, kde=True)
            else:
                plt.hist(data, bins=30, density=True)
            plt.title(f'Distribution of {column}')
            plt.savefig(os.path.join(plots_dir, f'distribution_{column}.png'))
            plt.close()
            
    def save_results(self):
        """Save both stats and insights"""
        self.logger.info("Saving results...")
        
        # Add timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.stats['timestamp'] = timestamp
        self.insights['timestamp'] = timestamp
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'statistics.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        self.logger.info(f"Statistics saved to: {stats_file}")
        
        # Save insights
        insights_file = os.path.join(self.output_dir, 'insights.json')
        with open(insights_file, 'w') as f:
            json.dump(self.insights, f, indent=4)
        self.logger.info(f"Insights saved to: {insights_file}")
        
    def show_summary(self):
        """Display a comprehensive summary of the analysis"""
        print("\n" + "="*50)
        print("📊 Dataset Analysis Summary:")
        
        # Basic Statistics
        print(f"\n📈 General Statistics:")
        print(f"  • Total rows: {self.stats['total_rows']}")
        
        overview = self.insights['dataset_overview']
        if 'search_engines' in overview:
            print(f"  • Google: {overview['search_engines'].get('google', 0):,}")
            print(f"  • Bing: {overview['search_engines'].get('bing', 0):,}")
        if 'unique_hostnames' in overview:
            print(f"  • Unique Hostnames: {overview['unique_hostnames']:,}")
        
        print("\n📉 Missing Values:")
        for column, stats in self.stats['missing_values'].items():
            print(f"  • {column}: {stats['count']} ({stats['percentage']}%)")
            
        print("\n🎯 Outliers (IQR method):")
        for column, stats in self.stats['outliers'].items():
            if stats['lower_bound'] is not None:  # Only show if data was available
                print(f"  • {column}: {stats['iqr_outliers_count']} ({stats['iqr_outliers_percentage']}%)")
        
        print("\n📊 Quality Metrics:")
        for metric, stats in self.insights['quality_metrics'].items():
            print(f"  • {metric}:")
            print(f"    - Mean: {stats['mean']:.2f}")
            if 'below_threshold_percentage' in stats:
                print(f"    - Below Threshold: {stats['below_threshold_percentage']}%")
            elif 'present_percentage' in stats:
                print(f"    - Present: {stats['present_percentage']}%")
            elif 'high_similarity_percentage' in stats:
                print(f"    - High Similarity: {stats['high_similarity_percentage']}%")
        
        print("\n🎯 Distribution Analysis:")
        dist = self.insights['distribution_analysis']
        if 'serp_position' in dist:
            print("  • SERP Positions:")
            print(f"    - Top 3: {dist['serp_position']['top_3_percentage']}%")
            print(f"    - Top 5: {dist['serp_position']['top_5_percentage']}%")
            print(f"    - Top 10: {dist['serp_position']['top_10_percentage']}%")
        
        # Show content relevance metrics
        for metric in ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1']:
            if metric in dist:
                print(f"  • {metric}:")
                print(f"    - Present: {dist[metric]['present_percentage']}%")
                print(f"    - Absent: {dist[metric]['absent_percentage']}%")
        
        # Show semantic similarity distributions
        for metric in ['semantic_similarity_title_query', 'semantic_similarity_content_query']:
            if metric in dist:
                element = 'title' if 'title' in metric else 'content'
                print(f"  • {element.capitalize()}-Query Similarity:")
                print(f"    - High: {dist[metric]['high_similarity']}%")
                print(f"    - Medium: {dist[metric]['medium_similarity']}%")
                print(f"    - Low: {dist[metric]['low_similarity']}%")
        
        # Show word count distribution
        if 'word_count' in dist:
            print("  • Content Length:")
            print(f"    - Short (<300): {dist['word_count']['short_content']}%")
            print(f"    - Medium (300-1000): {dist['word_count']['medium_content']}%")
            print(f"    - Long (1000-3000): {dist['word_count']['long_content']}%")
            print(f"    - Very Long (>3000): {dist['word_count']['very_long_content']}%")
        
        print("\n🔄 Strong Correlations:")
        for col1, corrs in self.stats['correlations'].items():
            for col2, value in corrs.items():
                if abs(value) > 0.5:
                    print(f"  • {col1} ↔️ {col2}: {value:.3f}")
        
        print("\n⚠️ Key Recommendations:")
        for i, rec in enumerate(self.insights['recommendations'], 1):
            print(f"  {i}. {rec['description']}")
            print(f"     → {rec['suggestion']}")
        
        print("="*50 + "\n")
        
        print("📁 Full statistics and insights have been saved to the output directory")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset features and outliers')
    parser.add_argument('--input', type=str, default='data/processed/combined.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--output', type=str, default='data/outliers',
                       help='Output directory for outliers results')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()

    analyzer = OutlierAnalyzer(
        input_file=args.input,
        output_dir=args.output,
        skip_plots=args.skip_plots
    )
    
    start_time = datetime.now()
    analyzer.run()
    analyzer.show_summary()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}")

if __name__ == "__main__":
    main() 