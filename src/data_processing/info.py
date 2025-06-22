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
import seaborn as sns
from typing import Dict, List, Tuple

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

class DatasetAnalyzer:
    def __init__(self,
                 input_file: str,
                 output_dir: str,
                 skip_plots: bool = False):
        self.input_file = input_file
        self.output_dir = output_dir
        self.skip_plots = skip_plots
        self.logger = Utils.set_colorful_logging('DatasetAnalyzer')
        
        # Define numerical columns to analyze
        self.lighthouse_scores = [
            'performance_score',
            'accessibility_score',
            'best-practices_score',
            'seo_score'
        ]
        
        # Define content relevance metrics
        self.content_metrics = [
            'query_in_title',
            'query_in_h1',
            'exact_query_in_title',
            'exact_query_in_h1',
            'query_density_body',
            'semantic_similarity_title_query',
            'semantic_similarity_content_query',
            'word_count'
        ]
        
        # Will be set after reading the data
        self.numerical_columns = []
        
        # Define categorical columns
        self.categorical_columns = [
            'engine',
            'hostname'
        ]
        
        # Initialize stats dictionary
        self.stats = {
            'dataset_overview': {},
            'missing_values': {},
            'descriptive_stats': {},
            'outliers': {},
            'categorical_stats': {},
            'correlations': {},
            'engine_comparison': {}
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'latex'), exist_ok=True)
        
    def generate_latex_tables(self):
        """Generate LaTeX tables for the paper"""
        self.logger.info("Generating LaTeX tables...")
        
        # Table 1: Data Collection and Filtering Stages
        self._generate_data_collection_stages_table()
        
        # Table 2: Comprehensive Dataset Statistics
        self._generate_comprehensive_stats_table()
        
        # Table 3: Lighthouse Score Quality Distribution
        self._generate_quality_distribution_table()

        # Table 4: Dataset Column Types
        self._generate_column_types_table()
        
    def _generate_data_collection_stages_table(self):
        """Generate table showing data collection and filtering stages"""
        latex_dir = os.path.join(self.output_dir, 'latex')
        
        # Create the LaTeX table
        latex_content = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Data Collection and Filtering Stages Summary}",
            "\\label{tab:data_collection_stages}",
            "\\small",
            "\\begin{tabular}{clr}",
            "\\toprule",
            "\\textbf{Stage} & \\textbf{Description} & \\textbf{Count} \\\\",
            "\\midrule",
            "1 & Initial SERP API Results & \\numprint{19950} \\\\",
            "2 & After General Platform Filtering & \\numprint{15678} \\\\",
            "3 & Successfully Crawled Resources & \\numprint{8956} \\\\",
            "4 & Domain-Relevant URLs & \\numprint{7047} \\\\",
            "5 & Final Dataset with Lighthouse Scores & \\numprint{14465} \\\\",
            "\\midrule",
            "\\multicolumn{3}{l}{\\textbf{Final Dataset Composition}} \\\\",
            f"& Unique URLs & \\numprint{{{self.stats['dataset_overview']['unique_urls']}}} \\\\",
            f"& Unique Hostnames & \\numprint{{{self.stats['dataset_overview']['unique_hostnames']}}} \\\\",
            f"& System A (Google) Entries & \\numprint{{{self.stats['dataset_overview']['engine_distribution']['google']}}} \\\\",
            f"& System B (Bing) Entries & \\numprint{{{self.stats['dataset_overview']['engine_distribution']['bing']}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\scriptsize",
            "\\item Note: The final dataset (Stage 5) includes duplicate URLs as they appear in different SERP positions",
            "\\item or in results from different search engines.",
            "\\end{tablenotes}",
            "\\end{table}"
        ]
        
        # Save the table
        with open(os.path.join(latex_dir, 'data_collection_stages.tex'), 'w') as f:
            f.write('\n'.join(latex_content))
            
    def _generate_comprehensive_stats_table(self):
        """Generate comprehensive statistics table"""
        latex_dir = os.path.join(self.output_dir, 'latex')
        
        # Create the LaTeX table
        latex_content = [
            "\\begin{table}[htbp]",
            "\\setlength{\tabcolsep}{3pt} %% important",
            "\\centering",
            "\\caption{Descriptive Statistics of the Final Analyzed Dataset (N=\\numprint{" + 
            str(self.stats['dataset_overview']['total_rows']) + "})}",
            "\\label{tab:comprehensive_stats}",
            "\\small",
            "\\begin{tabular}{lrrrrrrrr}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Mean} & \\textbf{Median} & \\textbf{SD} & " +
            "\\textbf{Min} & \\textbf{Max} & \\textbf{Q1} & \\textbf{Q3} & \\textbf{IQR\\%} \\\\",
            "\\midrule",
            "\\multicolumn{9}{l}{\\textbf{SERP Position}} \\\\"
        ]
        
        # Add SERP position stats if available
        if 'position' in self.stats['descriptive_stats']:
            serp_stats = self.stats['descriptive_stats']['position']
            latex_content.append(
                f"Overall & {serp_stats['mean']:.2f} & {serp_stats['50%']:.2f} & {serp_stats['std']:.2f} & " +
                f"{serp_stats['min']:.0f} & {serp_stats['max']:.0f} & {serp_stats['25%']:.2f} & " +
                f"{serp_stats['75%']:.2f} & - \\\\"
            )
        
        # Add Lighthouse scores
        lighthouse_metrics = [m for m in self.lighthouse_scores if m in self.stats['descriptive_stats']]
        if lighthouse_metrics:
            latex_content.append("\\midrule")
            latex_content.append("\\multicolumn{9}{l}{\\textbf{Lighthouse Scores}} \\\\")
            
            for metric in lighthouse_metrics:
                stats = self.stats['descriptive_stats'][metric]
                outliers = self.stats['outliers'][metric]['iqr_method']['percentage']
                name = self._format_column_name(metric)
                latex_content.append(
                    f"{name} & {stats['mean']:.2f} & {stats['50%']:.2f} & {stats['std']:.2f} & " +
                    f"{stats['min']:.0f} & {stats['max']:.0f} & {stats['25%']:.2f} & " +
                    f"{stats['75%']:.2f} & {outliers:.2f} \\\\"
                )
        
        # Add Content Metrics
        content_present = [m for m in self.content_metrics if m in self.stats['descriptive_stats']]
        if content_present:
            latex_content.append("\\midrule")
            latex_content.append("\\multicolumn{9}{l}{\\textbf{Content Relevance Metrics}} \\\\")
            
            for metric in content_present:
                stats = self.stats['descriptive_stats'][metric]
                outliers = self.stats['outliers'][metric]['iqr_method']['percentage']
                name = self._format_column_name(metric)
                latex_content.append(
                    f"{name} & {stats['mean']:.2f} & {stats['50%']:.2f} & {stats['std']:.2f} & " +
                    f"{stats['min']:.2f} & {stats['max']:.2f} & {stats['25%']:.2f} & " +
                    f"{stats['75%']:.2f} & {outliers:.2f} \\\\"
                )
        
        # Add engine comparison if available
        if 'engine_comparison' in self.stats and 'engine' in self.df.columns:
            lighthouse_comparison = [m for m in self.lighthouse_scores if m in self.stats['engine_comparison']]
            if lighthouse_comparison:
                latex_content.extend([
                    "\\midrule",
                    "\\multicolumn{9}{l}{\\textbf{Search Engine Comparison (Median Values)}} \\\\",
                    "& \\multicolumn{4}{l}{System A (Google)} & \\multicolumn{4}{l}{System B (Bing)} \\\\"
                ])
                
                for metric in lighthouse_comparison:
                    name = self._format_column_name(metric)
                    google_stats = self.stats['engine_comparison'][metric]['google']
                    bing_stats = self.stats['engine_comparison'][metric]['bing']
                    latex_content.append(
                        f"{name} & \\multicolumn{{4}}{{l}}{{{google_stats['50%']:.1f}}} & "
                        f"\\multicolumn{{4}}{{l}}{{{bing_stats['50%']:.1f}}} \\\\"
                    )
                
                # Add content metrics comparison if available
                content_comparison = [m for m in self.content_metrics if m in self.stats['engine_comparison']]
                if content_comparison:
                    latex_content.append("\\multicolumn{9}{l}{\\textbf{Content Metrics by Search Engine (Median Values)}} \\\\")
                    
                    for metric in content_comparison:
                        name = self._format_column_name(metric)
                        google_stats = self.stats['engine_comparison'][metric]['google']
                        bing_stats = self.stats['engine_comparison'][metric]['bing']
                        latex_content.append(
                            f"{name} & \\multicolumn{{4}}{{l}}{{{google_stats['50%']:.2f}}} & "
                            f"\\multicolumn{{4}}{{l}}{{{bing_stats['50%']:.2f}}} \\\\"
                        )
        
        # Close the table
        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\scriptsize",
            "\\item Note: SD = Standard Deviation; Q1 = First Quartile; Q3 = Third Quartile",
            "\\item IQR\\% indicates the percentage of values identified as outliers using the IQR method",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        # Save the table
        with open(os.path.join(latex_dir, 'comprehensive_stats.tex'), 'w') as f:
            f.write('\n'.join(latex_content))
            
    def _generate_quality_distribution_table(self):
        """Generate table showing quality distribution of Lighthouse scores"""
        latex_dir = os.path.join(self.output_dir, 'latex')
        
        # Calculate quality categories for each score
        quality_stats = {}
        for metric in ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']:
            if metric not in self.stats['descriptive_stats']:
                continue
                
            data = self.df[metric].dropna()
            quality_stats[metric] = {
                'excellent': len(data[data >= 90]) / len(data) * 100,
                'good': len(data[(data >= 70) & (data < 90)]) / len(data) * 100,
                'fair': len(data[(data >= 50) & (data < 70)]) / len(data) * 100,
                'poor': len(data[data < 50]) / len(data) * 100
            }
        
        # Create the LaTeX table
        latex_content = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Distribution of Resources by Lighthouse Score Quality Categories}",
            "\\label{tab:quality_distribution}",
            "\\small",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "\\textbf{Metric} & \\textbf{Excellent} & \\textbf{Good} & \\textbf{Fair} & \\textbf{Poor} \\\\",
            "& (90-100\\%) & (70-89\\%) & (50-69\\%) & (<50\\%) \\\\",
            "\\midrule"
        ]
        
        for metric, stats in quality_stats.items():
            name = self._format_column_name(metric)
            latex_content.append(
                f"{name} & {stats['excellent']:.1f}\\% & {stats['good']:.1f}\\% & " +
                f"{stats['fair']:.1f}\\% & {stats['poor']:.1f}\\% \\\\"
            )
        
        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\small",
            "\\item Note: Categories are based on Google's Lighthouse score interpretation guidelines",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        # Save the table
        with open(os.path.join(latex_dir, 'quality_distribution.tex'), 'w') as f:
            f.write('\n'.join(latex_content))
            
    def _generate_column_types_table(self):
        """Generate table showing dataset column names and their types"""
        latex_dir = os.path.join(self.output_dir, 'latex')
        
        # Group columns by their type/category
        column_groups = {
            'Search Engine Features': ['engine', 'position'],
            'URL Information': ['link', 'hostname'],
            'Lighthouse Scores': self.lighthouse_scores,
            'Content Metrics': self.content_metrics
        }
        
        # Create the LaTeX table
        latex_content = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Dataset Column Types and Descriptions}",
            "\\label{tab:dataset_columns_types}",
            "\\small",
            "\\begin{tabular}{llp{7cm}}",
            "\\toprule",
            "\\textbf{Category} & \\textbf{Column} & \\textbf{Description/Type} \\\\",
            "\\midrule"
        ]
        
        for category, columns in column_groups.items():
            # Add category header
            latex_content.append(f"\\multicolumn{{3}}{{l}}{{\\textbf{{{category}}}}} \\\\")
            
            # Add columns in this category
            for col in columns:
                if col in self.df.columns:
                    dtype = self._get_friendly_dtype(col, str(self.df[col].dtype))
                    description = self._get_column_description(col, dtype)
                    latex_content.append(f"{' ' * 4}{col} & {dtype} & {description} \\\\")
            
            # Add midrule between categories
            latex_content.append("\\midrule")
        
        # Close the table
        latex_content.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\begin{tablenotes}",
            "\\scriptsize",
            "\\item Note: Data types have been simplified from their internal pandas representation:",
            "\\item string: textual data, categorical: fixed set of values,",
            "\\item float: decimal numbers, integer: whole numbers.",
            "\\end{tablenotes}",
            "\\end{table}"
        ])
        
        # Save the table
        with open(os.path.join(latex_dir, 'dataset_columns_types.tex'), 'w') as f:
            f.write('\n'.join(latex_content))

    def _get_friendly_dtype(self, column: str, dtype: str) -> str:
        """Convert pandas dtype to more readable format"""
        if dtype == 'object':
            if column == 'engine':
                return 'categorical'
            return 'string'
        elif dtype.startswith('float'):
            return 'float'
        elif dtype.startswith('int'):
            return 'integer'
        return dtype

    def _get_column_description(self, column: str, dtype: str) -> str:
        """Get description for a column based on its name and type"""
        descriptions = {
            'engine': 'Search engine identifier (google/bing)',
            'position': 'Position in search results (1-based)',
            'url': 'Complete URL of the webpage',
            'link': 'Complete URL of the webpage',
            'hostname': 'Domain name extracted from URL',
            'performance_score': 'Lighthouse performance metric (0-100)',
            'accessibility_score': 'Lighthouse accessibility metric (0-100)',
            'best-practices_score': 'Lighthouse best practices metric (0-100)',
            'seo_score': 'Lighthouse SEO metric (0-100)',
            # Content metrics
            'query_in_title': 'Whether query words appear in page title (0 or 1)',
            'query_in_h1': 'Whether query words appear in first H1 heading (0 or 1)',
            'exact_query_in_title': 'Whether exact query appears in page title (0 or 1)',
            'exact_query_in_h1': 'Whether exact query appears in first H1 heading (0 or 1)',
            'query_density_body': 'Frequency of query in main text content (%)',
            'semantic_similarity_title_query': 'Semantic similarity between title and query (0-1)',
            'semantic_similarity_content_query': 'Semantic similarity between content and query (0-1)',
            'word_count': 'Total word count of the main content'
        }
        
        # Return custom description if available, otherwise return type info
        return descriptions.get(column, f"Numerical feature of type {dtype}")
        
    def _format_column_name(self, column: str) -> str:
        """Format column names for LaTeX tables"""
        # Replace underscores and hyphens
        name = column.replace('_', ' ').replace('-', ' ')
        
        # Capitalize words
        name = ' '.join(word.capitalize() for word in name.split())
        
        # Special cases
        name = name.replace('Serp', 'SERP')
        name = name.replace('Seo', 'SEO')
        name = name.replace('Pwa', 'PWA')
        
        # Content feature special cases
        name = name.replace('Query In Title', 'Query in Title')
        name = name.replace('Query In H1', 'Query in H1')
        name = name.replace('Exact Query In Title', 'Exact Query in Title')
        name = name.replace('Exact Query In H1', 'Exact Query in H1')
        name = name.replace('Query Density Body', 'Query Density')
        name = name.replace('Semantic Similarity Title Query', 'Title-Query Similarity')
        name = name.replace('Semantic Similarity Content Query', 'Content-Query Similarity')
        name = name.replace('Word Count', 'Word Count')
        
        return name
        
    def run(self):
        """Run the comprehensive dataset analysis"""
        # Read the dataset
        self.logger.info("Reading dataset...")
        self.df = pd.read_csv(self.input_file)
        
        # Extract hostname from url if needed
        if 'url' in self.df.columns and 'hostname' not in self.df.columns:
            self.df['hostname'] = self.df['url'].apply(lambda x: Utils.get_hostname(x) if pd.notnull(x) else None)
        # For backward compatibility
        elif 'link' in self.df.columns and 'hostname' not in self.df.columns:
            self.df['hostname'] = self.df['link'].apply(lambda x: Utils.get_hostname(x) if pd.notnull(x) else None)
        
        # Set numerical columns based on available columns
        self.numerical_columns = [col for col in self.lighthouse_scores if col in self.df.columns]
        if 'position' in self.df.columns:
            self.numerical_columns.append('position')
        
        # Add content metrics that are in the dataframe
        for col in self.content_metrics:
            if col in self.df.columns:
                self.numerical_columns.append(col)
        
        # Add any additional numerical columns that might be present
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col not in self.numerical_columns and col != 'position':
                if col not in self.content_metrics:
                    self.content_metrics.append(col)
                self.numerical_columns.append(col)
        
        # Run analyses
        self.analyze_dataset_overview()
        self.analyze_missing_values()
        self.analyze_descriptive_stats()
        self.analyze_outliers()
        self.analyze_categorical_features()
        self.analyze_correlations()
        self.analyze_engine_comparison()
        
        # Generate plots if not skipped
        if not self.skip_plots:
            self.generate_plots()
        
        # Generate LaTeX tables
        self.generate_latex_tables()
        
        # Save results
        self.save_results()
        
    def analyze_dataset_overview(self):
        """Analyze general dataset information"""
        self.logger.info("Analyzing dataset overview...")
        
        overview = {
            'total_rows': len(self.df),
            'columns': list(self.df.columns),
            'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
        }
        
        # Handle URL column, whether it's 'url' or 'link'
        if 'url' in self.df.columns:
            overview['unique_urls'] = len(self.df['url'].unique())
        elif 'link' in self.df.columns:
            overview['unique_urls'] = len(self.df['link'].unique())
        else:
            overview['unique_urls'] = 0
            
        if 'hostname' in self.df.columns:
            overview['unique_hostnames'] = len(self.df['hostname'].unique())
        else:
            overview['unique_hostnames'] = 0
            
        if 'engine' in self.df.columns:
            overview['engine_distribution'] = {
                'google': len(self.df[self.df['engine'] == 'google']),
                'bing': len(self.df[self.df['engine'] == 'bing'])
            }
        else:
            overview['engine_distribution'] = {'google': 0, 'bing': 0}
            
        if 'position' in self.df.columns:
            overview['position_stats'] = {
                'overall': {
                    'min': float(self.df['position'].min()),
                    'max': float(self.df['position'].max()),
                    'mean': float(self.df['position'].mean()),
                    'median': float(self.df['position'].median())
                }
            }
            
            if 'engine' in self.df.columns:
                overview['position_stats']['by_engine'] = {
                    'google': self.df[self.df['engine'] == 'google']['position'].describe().to_dict(),
                    'bing': self.df[self.df['engine'] == 'bing']['position'].describe().to_dict()
                }
            
        self.stats['dataset_overview'] = overview
        
    def analyze_missing_values(self):
        """Analyze missing values in the dataset"""
        self.logger.info("Analyzing missing values...")
        
        missing_stats = {}
        for column in self.df.columns:
            missing_count = self.df[column].isnull().sum()
            if missing_count > 0:
                missing_stats[column] = {
                    'count': int(missing_count),
                    'percentage': round(missing_count / len(self.df) * 100, 2)
                }
        
        self.stats['missing_values'] = missing_stats
        
    def analyze_descriptive_stats(self):
        """Calculate descriptive statistics for numerical features"""
        self.logger.info("Calculating descriptive statistics...")
        
        desc_stats = {}
        for column in self.numerical_columns:
            if column not in self.df.columns:
                continue
                
            data = self.df[column].dropna()
            if len(data) == 0:
                continue
                
            stats_dict = data.describe().to_dict()
            stats_dict.update({
                'skewness': float(data.skew()),
                'kurtosis': float(data.kurtosis())
            })
            desc_stats[column] = stats_dict
            
        self.stats['descriptive_stats'] = desc_stats
        
    def analyze_outliers(self):
        """Analyze outliers using IQR and Z-score methods"""
        self.logger.info("Analyzing outliers...")
        
        outlier_stats = {}
        for column in self.numerical_columns:
            if column not in self.df.columns:
                continue
                
            data = self.df[column].dropna()
            if len(data) == 0:
                continue
                
            # IQR method
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs(stats.zscore(data))
            z_score_outliers = data[z_scores > 3]
            
            outlier_stats[column] = {
                'iqr_method': {
                    'count': len(iqr_outliers),
                    'percentage': round(len(iqr_outliers) / len(data) * 100, 2),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                },
                'z_score_method': {
                    'count': len(z_score_outliers),
                    'percentage': round(len(z_score_outliers) / len(data) * 100, 2)
                }
            }
            
        self.stats['outliers'] = outlier_stats
        
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        self.logger.info("Analyzing categorical features...")
        
        cat_stats = {}
        for column in self.categorical_columns:
            if column not in self.df.columns:
                continue
                
            value_counts = self.df[column].value_counts()
            cat_stats[column] = {
                'unique_values': len(value_counts),
                'top_10_values': value_counts.head(10).to_dict(),
                'top_10_percentages': (value_counts.head(10) / len(self.df) * 100).round(2).to_dict()
            }
            
        self.stats['categorical_stats'] = cat_stats
        
    def analyze_correlations(self):
        """Analyze correlations between numerical features"""
        self.logger.info("Analyzing correlations...")
        
        # Get numerical columns that exist in the dataset
        num_cols = [col for col in self.numerical_columns if col in self.df.columns]
        
        if len(num_cols) > 1:
            # Calculate both Pearson and Spearman correlations
            pearson_corr = self.df[num_cols].corr(method='pearson')
            spearman_corr = self.df[num_cols].corr(method='spearman')
            
            # Convert to dictionary format
            self.stats['correlations'] = {
                'pearson': pearson_corr.to_dict(),
                'spearman': spearman_corr.to_dict()
            }
            
    def analyze_engine_comparison(self):
        """Compare metrics between Google and Bing"""
        self.logger.info("Comparing engines...")
        
        comparison = {}
        for column in self.numerical_columns:
            if column not in self.df.columns:
                continue
                
            google_stats = self.df[self.df['engine'] == 'google'][column].describe().to_dict()
            bing_stats = self.df[self.df['engine'] == 'bing'][column].describe().to_dict()
            
            comparison[column] = {
                'google': google_stats,
                'bing': bing_stats
            }
            
        self.stats['engine_comparison'] = comparison
        
    def generate_plots(self):
        """Generate visualization plots"""
        self.logger.info("Generating plots...")
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # Create plots directory
        plots_dir = os.path.join(self.output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Figure 1: Search Engine and SERP Position Distribution
        self._plot_search_engine_distribution(plots_dir)
        
        # Figure 2: Lighthouse Score Distributions
        self._plot_lighthouse_distributions(plots_dir)
        
        # Figure 3: Correlation Heatmap
        self._plot_correlation_heatmap(plots_dir)
        
    def _plot_search_engine_distribution(self, plots_dir):
        """Generate search engine distribution plots"""
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2])
        
        # Plot 1a: Search Engine Distribution
        engine_counts = self.df['engine'].value_counts()
        sns.barplot(x=engine_counts.index, y=engine_counts.values, ax=ax1)
        ax1.set_title('Distribution of Entries by Search Engine')
        ax1.set_ylabel('Number of Entries')
        
        # Plot 1b: SERP Position Distribution by Engine
        serp_by_engine = pd.crosstab(self.df['position'], self.df['engine'])
        serp_by_engine.plot(kind='bar', ax=ax2)
        ax2.set_title('SERP Position Distribution by Search Engine')
        ax2.set_xlabel('SERP Position')
        ax2.set_ylabel('Number of Entries')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'search_engine_distribution.png'))
        plt.close()
        
    def _plot_lighthouse_distributions(self, plots_dir):
        """Generate Lighthouse score distribution plots"""
        metrics = ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']
        
        # Create box plots for each metric, split by engine
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            sns.boxplot(x='engine', y=metric, data=self.df, ax=axes[idx])
            axes[idx].set_title(self._format_column_name(metric))
            axes[idx].set_xlabel('')
            if idx % 2 == 0:
                axes[idx].set_ylabel('Score')
            else:
                axes[idx].set_ylabel('')
        
        plt.suptitle('Lighthouse Scores Distribution by Search Engine', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'lighthouse_distributions.png'))
        plt.close()
        
    def _plot_correlation_heatmap(self, plots_dir):
        """Generate correlation heatmap"""
        # Use only numerical columns
        num_cols = [col for col in self.numerical_columns if col in self.df.columns]
        
        if len(num_cols) > 1:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[num_cols].corr(method='spearman')
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(correlation_matrix), k=1)
            
            # Create heatmap
            sns.heatmap(correlation_matrix,
                       mask=mask,
                       annot=True,
                       cmap='coolwarm',
                       center=0,
                       fmt='.2f',
                       square=True,
                       cbar_kws={'label': 'Spearman Correlation'})
            
            plt.title('Feature Correlation Matrix (Spearman)')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'))
            plt.close()
            
    def save_results(self):
        """Save analysis results"""
        self.logger.info("Saving results...")
        
        # Add timestamp
        self.stats['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save statistics
        stats_file = os.path.join(self.output_dir, 'dataset_info.json')
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        self.logger.info(f"Statistics saved to: {stats_file}")
        
    def show_summary(self):
        """Display a comprehensive summary of the analysis"""
        print("\n" + "="*50)
        print("📊 Dataset Analysis Summary")
        print("="*50)
        
        # Dataset Overview
        overview = self.stats['dataset_overview']
        print("\n📈 General Statistics:")
        print(f"  • Total rows: {overview['total_rows']:,}")
        if 'unique_urls' in overview:
            print(f"  • Unique URLs: {overview['unique_urls']:,}")
        if 'unique_hostnames' in overview:
            print(f"  • Unique hostnames: {overview['unique_hostnames']:,}")
        if 'engine_distribution' in overview:
            print(f"  • Google entries: {overview['engine_distribution'].get('google', 0):,}")
            print(f"  • Bing entries: {overview['engine_distribution'].get('bing', 0):,}")
        
        # Missing Values
        print("\n📉 Missing Values:")
        for column, stats in self.stats['missing_values'].items():
            print(f"  • {column}: {stats['count']:,} ({stats['percentage']}%)")
            
        # Lighthouse Metrics
        lighthouse_metrics = [m for m in self.lighthouse_scores if m in self.stats.get('descriptive_stats', {})]
        if lighthouse_metrics:
            print("\n📊 Lighthouse Metrics (median values):")
            for column in lighthouse_metrics:
                stats = self.stats['descriptive_stats'][column]
                print(f"  • {column}: {stats['50%']:.2f}")
            
        # Content Metrics
        content_metrics = [m for m in self.content_metrics if m in self.stats.get('descriptive_stats', {})]
        if content_metrics:
            print("\n📄 Content Metrics (median values):")
            for column in content_metrics:
                stats = self.stats['descriptive_stats'][column]
                print(f"  • {column}: {stats['50%']:.2f}")
                
        # Binary Content Features
        binary_metrics = [m for m in ['query_in_title', 'query_in_h1', 'exact_query_in_title', 'exact_query_in_h1'] 
                         if m in self.df.columns]
        if binary_metrics:
            print("\n🔍 Content Presence Metrics (percentage present):")
            for metric in binary_metrics:
                present_pct = (self.df[metric] > 0).mean() * 100
                print(f"  • {metric}: {present_pct:.1f}%")
            
        # Semantic Similarity Metrics
        similarity_metrics = [m for m in ['semantic_similarity_title_query', 'semantic_similarity_content_query'] 
                             if m in self.df.columns]
        if similarity_metrics:
            print("\n🔄 Semantic Similarity Metrics:")
            for metric in similarity_metrics:
                high_similarity = (self.df[metric] >= 0.7).mean() * 100
                medium_similarity = ((self.df[metric] >= 0.4) & (self.df[metric] < 0.7)).mean() * 100
                print(f"  • {metric}:")
                print(f"    - High similarity (≥0.7): {high_similarity:.1f}%")
                print(f"    - Medium similarity (0.4-0.7): {medium_similarity:.1f}%")
        
        # Word Count Distribution
        if 'word_count' in self.df.columns:
            print("\n📝 Word Count Distribution:")
            short = (self.df['word_count'] < 300).mean() * 100
            medium = ((self.df['word_count'] >= 300) & (self.df['word_count'] < 1000)).mean() * 100
            long = (self.df['word_count'] >= 1000).mean() * 100
            print(f"  • Short (<300 words): {short:.1f}%")
            print(f"  • Medium (300-1000 words): {medium:.1f}%")
            print(f"  • Long (>1000 words): {long:.1f}%")
                
        # Outliers
        print("\n⚠️ Outliers (IQR method):")
        for column, stats in self.stats['outliers'].items():
            if column in self.lighthouse_scores or column in self.content_metrics:
                print(f"  • {column}: {stats['iqr_method']['percentage']}%")
                
        # Strong Correlations
        if 'position' in self.df.columns and 'correlations' in self.stats:
            print("\n🔄 Notable Correlations with SERP Position (Spearman):")
            correlations = self.stats['correlations']['spearman']
            for column in correlations:
                if column != 'position' and 'position' in correlations[column]:
                    corr = correlations[column]['position']
                    if abs(corr) > 0.1:  # Show only meaningful correlations
                        print(f"  • {column}: {corr:.3f}")
        
        # Content metric correlations
        if 'correlations' in self.stats and 'spearman' in self.stats['correlations']:
            content_correlations = []
            correlations = self.stats['correlations']['spearman']
            for col1 in self.content_metrics:
                if col1 not in correlations:
                    continue
                for col2 in self.lighthouse_scores:
                    if col2 not in correlations[col1]:
                        continue
                    corr = correlations[col1][col2]
                    if abs(corr) > 0.15:  # Show only meaningful correlations
                        content_correlations.append((col1, col2, corr))
            
            if content_correlations:
                print("\n🔄 Notable Content-Lighthouse Correlations (Spearman):")
                for col1, col2, corr in content_correlations:
                    print(f"  • {col1} ↔ {col2}: {corr:.3f}")
                    
        # Engine Comparison
        if 'engine_comparison' in self.stats and any(m in self.stats['engine_comparison'] for m in self.lighthouse_scores):
            print("\n🔍 Google vs Bing Comparison (medians):")
            for metric, stats in self.stats['engine_comparison'].items():
                if metric in self.lighthouse_scores or metric in self.content_metrics:
                    google_median = stats['google']['50%']
                    bing_median = stats['bing']['50%']
                    print(f"  • {metric}:")
                    print(f"    - Google: {google_median:.2f}")
                    print(f"    - Bing: {bing_median:.2f}")
                
        print("\n📁 Full analysis results have been saved to the output directory")
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze dataset features and generate comprehensive statistics')
    parser.add_argument('--input', type=str, default='data/processed/combined.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--output', type=str, default='data/analysis',
                       help='Output directory for analysis results')
    parser.add_argument('--skip-plots', action='store_true',
                       help='Skip generating plots')
    args = parser.parse_args()

    analyzer = DatasetAnalyzer(
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