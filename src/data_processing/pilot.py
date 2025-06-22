import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json
import argparse
from typing import Dict, List, Tuple

# Add the root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.lib.utils import Utils

class PilotAnalyzer:
    def __init__(self,
                 dataset_file: str,
                 stats_file: str,
                 output_file: str):
        self.dataset_file = dataset_file
        self.stats_file = stats_file
        self.output_file = output_file
        self.logger = Utils.set_colorful_logging('PilotAnalyzer')
        
        # Define score thresholds for recommendations
        self.thresholds = {
            'performance_score': 90,
            'accessibility_score': 90,
            'best-practices_score': 90,
            'seo_score': 90
        }
        
        # Initialize insights
        self.insights = {
            'dataset_overview': {},
            'quality_metrics': {},
            'distribution_analysis': {},
            'correlation_insights': {},
            'recommendations': []
        }
        
    def run(self):
        """Run the pilot analysis"""
        # Load data
        self.logger.info("Loading data...")
        df = pd.read_csv(self.dataset_file)
        
        with open(self.stats_file, 'r') as f:
            stats = json.load(f)
            
        # Analyze dataset overview
        self.analyze_overview(df, stats)
        
        # Analyze quality metrics
        self.analyze_quality_metrics(df, stats)
        
        # Analyze distributions
        self.analyze_distributions(df, stats)
        
        # Analyze correlations
        self.analyze_correlations(stats)
        
        # Generate recommendations
        self.generate_recommendations(df, stats)
        
        # Save insights
        self.save_insights()
        
    def analyze_overview(self, df: pd.DataFrame, stats: Dict):
        """Analyze dataset overview"""
        self.logger.info("Analyzing dataset overview...")
        
        overview = {
            'total_rows': stats['total_rows'],
            'search_engines': {
                'google': len(df[df['engine'] == 'google']),
                'bing': len(df[df['engine'] == 'bing'])
            },
            'unique_hostnames': len(df['hostname'].unique()),
            'serp_position_range': {
                'min': df['serp_position'].min(),
                'max': df['serp_position'].max()
            },
            'missing_values': stats['missing_values']
        }
        
        self.insights['dataset_overview'] = overview
        
    def analyze_quality_metrics(self, df: pd.DataFrame, stats: Dict):
        """Analyze quality metrics"""
        self.logger.info("Analyzing quality metrics...")
        
        quality_metrics = {}
        
        for metric in ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']:
            if metric in df.columns:
                data = df[metric].dropna()
                if len(data) > 0:
                    quality_metrics[metric] = {
                        'mean': float(data.mean()),
                        'median': float(data.median()),
                        'below_threshold': len(data[data < self.thresholds[metric]]),
                        'below_threshold_percentage': round(len(data[data < self.thresholds[metric]]) / len(data) * 100, 2),
                        'distribution': stats['distributions'][metric]
                    }
        
        self.insights['quality_metrics'] = quality_metrics
        
    def analyze_distributions(self, df: pd.DataFrame, stats: Dict):
        """Analyze distributions"""
        self.logger.info("Analyzing distributions...")
        
        distribution_insights = {}
        
        # Analyze SERP position distribution
        serp_dist = df['serp_position'].value_counts().sort_index()
        distribution_insights['serp_position'] = {
            'top_3_percentage': round(len(df[df['serp_position'] <= 3]) / len(df) * 100, 2),
            'top_5_percentage': round(len(df[df['serp_position'] <= 5]) / len(df) * 100, 2),
            'top_10_percentage': round(len(df[df['serp_position'] <= 10]) / len(df) * 100, 2)
        }
        
        # Analyze score distributions
        for metric in ['performance_score', 'accessibility_score', 'best-practices_score', 'seo_score']:
            if metric in df.columns:
                data = df[metric].dropna()
                if len(data) > 0:
                    distribution_insights[metric] = {
                        'excellent': round(len(data[data >= 90]) / len(data) * 100, 2),
                        'good': round(len(data[(data >= 70) & (data < 90)]) / len(data) * 100, 2),
                        'fair': round(len(data[(data >= 50) & (data < 70)]) / len(data) * 100, 2),
                        'poor': round(len(data[data < 50]) / len(data) * 100, 2)
                    }
        
        self.insights['distribution_analysis'] = distribution_insights
        
    def analyze_correlations(self, stats: Dict):
        """Analyze correlations"""
        self.logger.info("Analyzing correlations...")
        
        correlation_insights = {}
        
        if 'correlations' in stats:
            # Find strong correlations (|r| > 0.5)
            strong_correlations = []
            for col1, corrs in stats['correlations'].items():
                for col2, value in corrs.items():
                    if abs(value) > 0.5:
                        strong_correlations.append({
                            'metrics': [col1, col2],
                            'correlation': value
                        })
            
            correlation_insights['strong_correlations'] = strong_correlations
        
        self.insights['correlation_insights'] = correlation_insights
        
    def generate_recommendations(self, df: pd.DataFrame, stats: Dict):
        """Generate recommendations based on analysis"""
        self.logger.info("Generating recommendations...")
        
        recommendations = []
        
        # Missing values recommendations
        for column, missing in stats['missing_values'].items():
            if missing['percentage'] > 1:
                recommendations.append({
                    'type': 'missing_values',
                    'metric': column,
                    'description': f"Handle {missing['percentage']}% missing values in {column}",
                    'suggestion': "Consider imputation or removal based on analysis needs"
                })
        
        # Outlier recommendations
        for column, outliers in stats['outliers'].items():
            if outliers['iqr_outliers_percentage'] > 5:
                recommendations.append({
                    'type': 'outliers',
                    'metric': column,
                    'description': f"Review {outliers['iqr_outliers_percentage']}% outliers in {column}",
                    'suggestion': "Investigate outlier values and consider treatment methods"
                })
        
        # Quality score recommendations
        for metric, stats in self.insights['quality_metrics'].items():
            if stats['below_threshold_percentage'] > 20:
                recommendations.append({
                    'type': 'quality',
                    'metric': metric,
                    'description': f"{stats['below_threshold_percentage']}% of URLs have {metric} below {self.thresholds[metric]}",
                    'suggestion': "Consider filtering or analyzing these low-scoring URLs separately"
                })
        
        # Distribution recommendations
        if 'serp_position' in self.insights['distribution_analysis']:
            if self.insights['distribution_analysis']['serp_position']['top_3_percentage'] < 30:
                recommendations.append({
                    'type': 'distribution',
                    'metric': 'serp_position',
                    'description': "Low representation of top 3 SERP positions",
                    'suggestion': "Consider balancing the dataset for better position representation"
                })
        
        self.insights['recommendations'] = recommendations
        
    def save_insights(self):
        """Save insights to a JSON file"""
        self.logger.info("Saving insights...")
        
        # Add timestamp
        self.insights['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save to JSON file
        with open(self.output_file, 'w') as f:
            json.dump(self.insights, f, indent=4)
            
        self.logger.info(f"Insights saved to: {self.output_file}")
        
    def show_summary(self):
        """Display a summary of the pilot analysis"""
        print("\n" + "="*50)
        print("🔍 Pilot Analysis Summary")
        
        print("\n📊 Dataset Overview:")
        overview = self.insights['dataset_overview']
        print(f"  • Total URLs: {overview['total_rows']:,}")
        print(f"  • Google: {overview['search_engines']['google']:,}")
        print(f"  • Bing: {overview['search_engines']['bing']:,}")
        print(f"  • Unique Hostnames: {overview['unique_hostnames']:,}")
        
        print("\n📈 Quality Metrics:")
        for metric, stats in self.insights['quality_metrics'].items():
            print(f"  • {metric}:")
            print(f"    - Mean: {stats['mean']:.2f}")
            print(f"    - Below Threshold: {stats['below_threshold_percentage']}%")
        
        print("\n🎯 Distribution Analysis:")
        dist = self.insights['distribution_analysis']
        if 'serp_position' in dist:
            print("  • SERP Positions:")
            print(f"    - Top 3: {dist['serp_position']['top_3_percentage']}%")
            print(f"    - Top 5: {dist['serp_position']['top_5_percentage']}%")
            print(f"    - Top 10: {dist['serp_position']['top_10_percentage']}%")
        
        print("\n🔄 Strong Correlations:")
        correlations = self.insights['correlation_insights'].get('strong_correlations', [])
        for corr in correlations:
            print(f"  • {corr['metrics'][0]} ↔️ {corr['metrics'][1]}: {corr['correlation']:.3f}")
        
        print("\n⚠️ Key Recommendations:")
        for i, rec in enumerate(self.insights['recommendations'], 1):
            print(f"  {i}. {rec['description']}")
            print(f"     → {rec['suggestion']}")
        
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Run pilot analysis on dataset')
    parser.add_argument('--dataset', type=str, default='data/processed/dataset.csv',
                       help='Input dataset CSV file')
    parser.add_argument('--stats', type=str, default='data/analysis/statistics.json',
                       help='Statistics JSON file from outlier analysis')
    parser.add_argument('--output', type=str, default='data/analysis/pilot_insights.json',
                       help='Output JSON file for pilot insights')
    args = parser.parse_args()

    analyzer = PilotAnalyzer(
        dataset_file=args.dataset,
        stats_file=args.stats,
        output_file=args.output
    )
    
    start_time = datetime.now()
    analyzer.run()
    analyzer.show_summary()
    duration = datetime.now() - start_time
    print(f"⏱️ Processing time: {duration}")

if __name__ == "__main__":
    main() 