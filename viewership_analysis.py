"""
Comprehensive Viewership Analysis Module
建立全面的觀看數據分析模組
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class ViewershipAnalyzer:
    """Comprehensive analyzer for League of Legends viewership data"""
    
    def __init__(self, data_path: str = None):
        """Initialize with viewership data"""
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            # Load default data
            self.df = pd.read_csv('c:/Users/shiou/Documents/DataVisualization/Oracle\'s_Elixir_data_Processed/viewership_market_data.csv')
        
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and clean viewership data for analysis"""
        # Convert date column
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Extract year and month for analysis
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Calculate viewership metrics
        self.df['hours_watched_millions'] = self.df['hours_watched'] / 1000000
        self.df['peak_to_avg_ratio'] = self.df['viewership_peak'] / self.df['viewership_average'].replace(0, 1)
        
        # Parse language breakdown for analysis
        self.df['primary_language'] = self.df['language_breakdown'].apply(self._extract_primary_language)
        
        print(f"數據準備完成: {len(self.df)} 場比賽, 涵蓋 {self.df['year'].min()}-{self.df['year'].max()}")
    
    def _extract_primary_language(self, language_str: str) -> str:
        """Extract primary language from language breakdown string"""
        try:
            if pd.isna(language_str):
                return 'Unknown'
            
            # Parse format like "Korean:2M|English:1.69M|Vietnamese:805K"
            languages = language_str.split('|')
            if languages:
                first_lang = languages[0].split(':')[0]
                return first_lang
            return 'Unknown'
        except:
            return 'Unknown'
    
    def create_viewership_distribution_chart(self) -> go.Figure:
        """
        Create interactive viewership distribution chart across tournaments
        建立跨賽事觀看數據分布的互動式圖表
        """
        # Calculate tournament statistics
        tournament_stats = self.df.groupby('league').agg({
            'viewership_peak': ['mean', 'max', 'min', 'std'],
            'viewership_average': ['mean', 'max', 'min'],
            'hours_watched_millions': 'sum',
            'match_id': 'count'
        }).round(0)
        
        # Flatten column names
        tournament_stats.columns = ['_'.join(col).strip() for col in tournament_stats.columns]
        tournament_stats = tournament_stats.reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '各賽事峰值觀看人數', '平均觀看人數分布',
                '總觀看時數 (百萬小時)', '峰值/平均觀看人數比值'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Peak viewership by tournament
        fig.add_trace(
            go.Bar(
                x=tournament_stats['league'],
                y=tournament_stats['viewership_peak_mean'],
                name='平均峰值',
                marker_color='#1f77b4',
                text=[f'{x:,.0f}' for x in tournament_stats['viewership_peak_mean']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Plot 2: Box plot for average viewership distribution
        box_data = []
        tournaments = self.df['league'].unique()
        for tournament in tournaments:
            tournament_data = self.df[self.df['league'] == tournament]['viewership_average']
            fig.add_trace(
                go.Box(
                    y=tournament_data,
                    name=tournament,
                    boxpoints='outliers'
                ),
                row=1, col=2
            )
        
        # Plot 3: Total hours watched
        fig.add_trace(
            go.Bar(
                x=tournament_stats['league'],
                y=tournament_stats['hours_watched_millions_sum'],
                name='總觀看時數 (百萬小時)',
                marker_color='#ff7f0e',
                text=[f'{x:.1f}M' for x in tournament_stats['hours_watched_millions_sum']],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Plot 4: Peak to Average ratio
        avg_ratios = self.df.groupby('league')['peak_to_avg_ratio'].mean()
        fig.add_trace(
            go.Scatter(
                x=avg_ratios.index,
                y=avg_ratios.values,
                mode='markers+lines',
                name='峰值/平均比',
                marker=dict(size=10, color='#2ca02c'),
                text=[f'{x:.1f}x' for x in avg_ratios.values],
                textposition='top center'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='聯盟電競觀看數據分布分析',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="賽事", row=1, col=1)
        fig.update_yaxes(title_text="峰值觀看人數", row=1, col=1)
        fig.update_xaxes(title_text="賽事", row=1, col=2)
        fig.update_yaxes(title_text="平均觀看人數", row=1, col=2)
        fig.update_xaxes(title_text="賽事", row=2, col=1)
        fig.update_yaxes(title_text="總觀看時數 (百萬小時)", row=2, col=1)
        fig.update_xaxes(title_text="賽事", row=2, col=2)
        fig.update_yaxes(title_text="峰值/平均比", row=2, col=2)
        
        return fig
    
    def create_social_media_heatmap(self) -> go.Figure:
        """
        Create social media interaction heat map
        建立社交媒體互動熱力圖
        """
        # Prepare social media data
        social_columns = ['social_engagement_twitter', 'social_engagement_reddit', 'social_engagement_youtube']
        
        # Create correlation matrix
        social_data = self.df[social_columns + ['viewership_peak', 'viewership_average']].copy()
        
        # Normalize data for better visualization
        for col in social_columns:
            social_data[f'{col}_normalized'] = social_data[col] / 1000000  # Convert to millions
        
        # Create pivot table for heatmap
        heatmap_data = []
        tournaments = self.df['league'].unique()
        platforms = ['Twitter', 'Reddit', 'YouTube']
        
        for tournament in tournaments:
            tournament_data = self.df[self.df['league'] == tournament]
            row_data = []
            for i, platform in enumerate(['twitter', 'reddit', 'youtube']):
                col_name = f'social_engagement_{platform}'
                avg_engagement = tournament_data[col_name].mean() / 1000000  # Convert to millions
                row_data.append(avg_engagement)
            heatmap_data.append(row_data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=platforms,
            y=tournaments,
            colorscale='Viridis',
            text=[[f'{val:.1f}M' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False,
            colorbar=dict(title="互動數 (百萬)")
        ))
        
        fig.update_layout(
            title='各賽事社交媒體互動熱力圖',
            xaxis_title="社交媒體平台",
            yaxis_title="賽事",
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_market_trend_analysis(self) -> go.Figure:
        """
        Create comprehensive market trend analysis
        建立全面的市場趨勢分析
        """
        # Create subplots for different trend analyses
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '觀看人數成長趨勢', '區域市場份額',
                '共同直播影響分析', '賽事等級表現'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Viewership growth over time
        monthly_data = self.df.groupby(['year', 'month']).agg({
            'viewership_peak': 'mean',
            'viewership_average': 'mean',
            'hours_watched_millions': 'sum'
        }).reset_index()
        
        monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1))
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['viewership_peak'],
                mode='lines+markers',
                name='峰值觀看人數',
                line=dict(color='#1f77b4', width=3)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_data['date'],
                y=monthly_data['hours_watched_millions'],
                mode='lines+markers',
                name='總觀看時數 (百萬小時)',
                line=dict(color='#ff7f0e', width=2),
                yaxis='y2'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: Regional market share
        regional_data = self.df.groupby('region')['hours_watched_millions'].sum().sort_values(ascending=True)
        
        fig.add_trace(
            go.Bar(
                x=regional_data.values,
                y=[self._translate_region(r) for r in regional_data.index],
                orientation='h',
                name='總觀看時數 (百萬小時)',
                marker_color='#2ca02c',
                text=[f'{x:.0f}M' for x in regional_data.values],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Plot 3: Co-streaming impact
        costream_data = self.df.copy()
        costream_data['costream_ratio'] = costream_data['co_streaming_hours'] / costream_data['hours_watched']
        costream_analysis = costream_data.groupby('league').agg({
            'costream_ratio': 'mean',
            'viewership_peak': 'mean'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=costream_analysis['costream_ratio'],
                y=costream_analysis['viewership_peak'],
                mode='markers+text',
                text=costream_analysis['league'],
                textposition='top center',
                name='共同直播影響',
                marker=dict(size=12, color='#d62728')
            ),
            row=2, col=1
        )
        
        # Plot 4: Tournament tier performance
        tier_data = self.df.groupby('tournament_tier').agg({
            'viewership_peak': 'mean',
            'viewership_average': 'mean',
            'match_id': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=tier_data['tournament_tier'],
                y=tier_data['viewership_peak'],
                name='峰值觀看人數',
                marker_color='#9467bd',
                text=[f'{x:,.0f}' for x in tier_data['viewership_peak']],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='聯盟電競市場趨勢分析',
            height=900,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="日期", row=1, col=1)
        fig.update_yaxes(title_text="峰值觀看人數", row=1, col=1)
        fig.update_yaxes(title_text="總觀看時數 (百萬小時)", row=1, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="總觀看時數 (百萬小時)", row=1, col=2)
        fig.update_yaxes(title_text="地區", row=1, col=2)
        
        fig.update_xaxes(title_text="共同直播比例", row=2, col=1)
        fig.update_yaxes(title_text="峰值觀看人數", row=2, col=1)
        
        fig.update_xaxes(title_text="賽事等級", row=2, col=2)
        fig.update_yaxes(title_text="峰值觀看人數", row=2, col=2)
        
        return fig
    
    def _translate_region(self, region: str) -> str:
        region_map = {
            'Global': '全球',
            'Korea': '韓國',
            'China': '中國',
            'Europe': '歐洲',
            'North America': '北美',
            'Vietnam': '越南',
            'Brazil': '巴西',
            'Japan': '日本',
            'Turkey': '土耳其',
            'CIS': '獨立國協',
            'Latin America': '拉丁美洲',
            'Oceania': '大洋洲',
            'SEA': '東南亞',
            'PCS': 'PCS',
            'LMS': 'LMS',
            'Other': '其他',
        }
        return region_map.get(region, region)
    
    def create_audience_preference_analysis(self) -> go.Figure:
        """
        Analyze audience preferences across different dimensions
        分析不同維度的觀眾偏好
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '語言偏好分布', '比賽時長與觀看人數',
                '串流平台受歡迎程度', '時間分布觀看趨勢'
            ),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Language preference pie chart
        language_counts = self.df['primary_language'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=language_counts.index,
                values=language_counts.values,
                name="語言偏好",
                hole=0.3
            ),
            row=1, col=1
        )
        
        # Plot 2: Duration vs Viewership scatter
        fig.add_trace(
            go.Scatter(
                x=self.df['duration_minutes'],
                y=self.df['viewership_peak'],
                mode='markers',
                name='比賽時長影響',
                marker=dict(
                    size=8,
                    color=self.df['tournament_tier'].map({'International': '#ff7f0e', 'Regional': '#1f77b4'}),
                    opacity=0.7
                ),
                text=self.df['league'],
                hovertemplate='比賽時長: %{x} 分鐘<br>峰值: %{y:,}<br>賽事: %{text}'
            ),
            row=1, col=2
        )
        
        # Plot 3: Streaming platform bar chart
        platform_data = self.df.groupby('stream_platform').agg({
            'viewership_peak': 'mean',
            'match_id': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=platform_data['stream_platform'].replace({'Multi-platform': '多平台'}),
                y=platform_data['viewership_peak'],
                name='平台受歡迎程度',
                marker_color='#2ca02c',
                text=[f'{x:,.0f}' for x in platform_data['viewership_peak']],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Plot 4: Monthly viewing patterns
        monthly_pattern = self.df.groupby('month')['viewership_peak'].mean()
        
        fig.add_trace(
            go.Scatter(
                x=monthly_pattern.index,
                y=monthly_pattern.values,
                mode='lines+markers',
                name='月度趨勢',
                line=dict(color='#d62728', width=3),
                marker=dict(size=8)
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='觀眾偏好分析',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text="比賽時長 (分鐘)", row=1, col=2)
        fig.update_yaxes(title_text="峰值觀看人數", row=1, col=2)
        fig.update_xaxes(title_text="串流平台", row=2, col=1)
        fig.update_yaxes(title_text="平均峰值觀看人數", row=2, col=1)
        fig.update_xaxes(title_text="月份", row=2, col=2)
        fig.update_yaxes(title_text="平均峰值觀看人數", row=2, col=2)
        
        return fig
    
    def generate_viewership_summary(self) -> Dict:
        """
        Generate comprehensive viewership summary statistics
        產生全面的觀看數據摘要統計
        """
        summary = {
            'total_matches': len(self.df),
            'total_hours_watched': self.df['hours_watched'].sum(),
            'avg_peak_viewership': self.df['viewership_peak'].mean(),
            'max_peak_viewership': self.df['viewership_peak'].max(),
            'top_tournament': self.df.groupby('league')['viewership_peak'].mean().idxmax(),
            'most_popular_region': self.df.groupby('region')['hours_watched'].sum().idxmax(),
            'total_social_engagement': (
                self.df['social_engagement_twitter'].sum() +
                self.df['social_engagement_reddit'].sum() + 
                self.df['social_engagement_youtube'].sum()
            ),
            'dominant_platform': self.df['stream_platform'].mode().iloc[0] if not self.df['stream_platform'].mode().empty else 'Multi-platform',
            'growth_rate': self._calculate_growth_rate(),
            'tournament_tiers': self.df['tournament_tier'].value_counts().to_dict()
        }
        
        return summary
    
    def _calculate_growth_rate(self) -> float:
        """Calculate year-over-year growth rate"""
        yearly_data = self.df.groupby('year')['viewership_peak'].mean()
        if len(yearly_data) >= 2:
            latest_year = yearly_data.index.max()
            previous_year = yearly_data.index[yearly_data.index < latest_year].max()
            return ((yearly_data[latest_year] - yearly_data[previous_year]) / yearly_data[previous_year]) * 100
        return 0.0
    
    def export_analysis_results(self, output_dir: str = 'c:/Users/shiou/Documents/DataVisualization/analysis_results/'):
        """
        Export all analysis results to files
        將所有分析結果匯出至檔案
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Export summary statistics
        summary = self.generate_viewership_summary()
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(f'{output_dir}viewership_summary.csv', index=False)
        
        # Export detailed tournament analysis
        tournament_analysis = self.df.groupby('league').agg({
            'viewership_peak': ['mean', 'max', 'min', 'std'],
            'viewership_average': ['mean', 'max', 'min'],
            'hours_watched': 'sum',
            'social_engagement_twitter': 'mean',
            'social_engagement_reddit': 'mean',
            'social_engagement_youtube': 'mean',
            'match_id': 'count'
        }).round(2)
        
        tournament_analysis.to_csv(f'{output_dir}tournament_analysis.csv')
        
        print(f"分析結果已匯出至: {output_dir}")


# Demo function
def main():
    """Demo script showing how to use the analyzer"""
    analyzer = ViewershipAnalyzer()
    
    print("建立觀看數據分析器...")
    
    # Generate summary
    summary = analyzer.generate_viewership_summary()
    print(f"\n觀看數據摘要:")
    print(f"總比賽場數: {summary['total_matches']:,}")
    print(f"總觀看時數: {summary['total_hours_watched']:,.0f}")
    print(f"平均峰值觀看人數: {summary['avg_peak_viewership']:,.0f}")
    print(f"最高峰值觀看人數: {summary['max_peak_viewership']:,.0f}")
    print(f"最受歡迎賽事: {summary['top_tournament']}")
    print(f"最受歡迎地區: {summary['most_popular_region']}")
    
    # Create visualizations
    print("\n正在建立視覺化圖表...")
    
    # You can uncomment these to generate and save the plots
    # dist_fig = analyzer.create_viewership_distribution_chart()
    # dist_fig.write_html('viewership_distribution.html')
    
    # social_fig = analyzer.create_social_media_heatmap()
    # social_fig.write_html('social_media_heatmap.html')
    
    # market_fig = analyzer.create_market_trend_analysis()
    # market_fig.write_html('market_trend_analysis.html')
    
    print("分析完成!")


if __name__ == "__main__":
    main()