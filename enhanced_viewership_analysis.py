"""
Comprehensive Viewership Analysis Module for League of Legends Esports
建立全面的英雄聯盟電競觀看數據分析模組
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
            try:
                self.df = pd.read_csv(data_path)
            except FileNotFoundError:
                print(f"檔案 {data_path} 不存在，使用範例數據")
                self.df = self._create_sample_data()
        else:
            # 嘗試載入預設路徑的數據
            try:
                self.df = pd.read_csv("c:/Users/shiou/Documents/DataVisualization/data/processed/esports_viewership_data.csv")
                print("成功載入爬蟲數據")
            except FileNotFoundError:
                print("未找到爬蟲數據檔案，使用範例數據")
                self.df = self._create_sample_data()
        
        self._prepare_data()
    
    def _create_sample_data(self):
        """建立範例觀眾數據 (基於真實統計資料)"""
        sample_data = [
            # 2024年數據
            {'tournament_name': '2024 World Championship', 'year': 2024, 'peak_viewers': 6856769, 'tournament_type': 'World Championship', 'region': 'Global', 'league': 'Worlds', 'hours_watched': 17000000, 'date': '2024-11-02'},
            {'tournament_name': 'LCK Spring 2024', 'year': 2024, 'peak_viewers': 1200000, 'tournament_type': 'Regional League', 'region': 'Korea', 'league': 'LCK', 'hours_watched': 3000000, 'date': '2024-04-15'},
            {'tournament_name': 'LPL Spring 2024', 'year': 2024, 'peak_viewers': 2500000, 'tournament_type': 'Regional League', 'region': 'China', 'league': 'LPL', 'hours_watched': 6250000, 'date': '2024-04-20'},
            {'tournament_name': 'LEC Spring 2024', 'year': 2024, 'peak_viewers': 800000, 'tournament_type': 'Regional League', 'region': 'Europe', 'league': 'LEC', 'hours_watched': 2000000, 'date': '2024-04-14'},
            {'tournament_name': 'LCS Spring 2024', 'year': 2024, 'peak_viewers': 150000, 'tournament_type': 'Regional League', 'region': 'North America', 'league': 'LCS', 'hours_watched': 375000, 'date': '2024-04-21'},
            {'tournament_name': 'MSI 2024', 'year': 2024, 'peak_viewers': 3200000, 'tournament_type': 'MSI', 'region': 'Global', 'league': 'MSI', 'hours_watched': 8000000, 'date': '2024-05-19'},
            {'tournament_name': 'CBLOL 2024', 'year': 2024, 'peak_viewers': 400000, 'tournament_type': 'Regional League', 'region': 'Brazil', 'league': 'CBLOL', 'hours_watched': 1000000, 'date': '2024-04-28'},
            {'tournament_name': 'VCS 2024', 'year': 2024, 'peak_viewers': 500000, 'tournament_type': 'Regional League', 'region': 'Vietnam', 'league': 'VCS', 'hours_watched': 1250000, 'date': '2024-04-25'},
            
            # 2023年數據  
            {'tournament_name': '2023 World Championship', 'year': 2023, 'peak_viewers': 4308901, 'tournament_type': 'World Championship', 'region': 'Global', 'league': 'Worlds', 'hours_watched': 10700000, 'date': '2023-11-19'},
            {'tournament_name': 'LCK Spring 2023', 'year': 2023, 'peak_viewers': 1000000, 'tournament_type': 'Regional League', 'region': 'Korea', 'league': 'LCK', 'hours_watched': 2500000, 'date': '2023-04-09'},
            {'tournament_name': 'LPL Spring 2023', 'year': 2023, 'peak_viewers': 2200000, 'tournament_type': 'Regional League', 'region': 'China', 'league': 'LPL', 'hours_watched': 5500000, 'date': '2023-04-16'},
            {'tournament_name': 'LEC Spring 2023', 'year': 2023, 'peak_viewers': 750000, 'tournament_type': 'Regional League', 'region': 'Europe', 'league': 'LEC', 'hours_watched': 1875000, 'date': '2023-04-09'},
            {'tournament_name': 'MSI 2023', 'year': 2023, 'peak_viewers': 2800000, 'tournament_type': 'MSI', 'region': 'Global', 'league': 'MSI', 'hours_watched': 7000000, 'date': '2023-05-21'},
            {'tournament_name': 'CBLOL 2023', 'year': 2023, 'peak_viewers': 350000, 'tournament_type': 'Regional League', 'region': 'Brazil', 'league': 'CBLOL', 'hours_watched': 875000, 'date': '2023-04-23'},
            
            # 2022年數據
            {'tournament_name': '2022 World Championship', 'year': 2022, 'peak_viewers': 3800000, 'tournament_type': 'World Championship', 'region': 'Global', 'league': 'Worlds', 'hours_watched': 9500000, 'date': '2022-11-05'},
            {'tournament_name': 'LCK Spring 2022', 'year': 2022, 'peak_viewers': 900000, 'tournament_type': 'Regional League', 'region': 'Korea', 'league': 'LCK', 'hours_watched': 2250000, 'date': '2022-04-10'},
            {'tournament_name': 'LPL Spring 2022', 'year': 2022, 'peak_viewers': 2000000, 'tournament_type': 'Regional League', 'region': 'China', 'league': 'LPL', 'hours_watched': 5000000, 'date': '2022-04-17'},
            {'tournament_name': 'LEC Spring 2022', 'year': 2022, 'peak_viewers': 700000, 'tournament_type': 'Regional League', 'region': 'Europe', 'league': 'LEC', 'hours_watched': 1750000, 'date': '2022-04-10'},
            {'tournament_name': 'MSI 2022', 'year': 2022, 'peak_viewers': 2500000, 'tournament_type': 'MSI', 'region': 'Global', 'league': 'MSI', 'hours_watched': 6250000, 'date': '2022-05-29'},
        ]
        
        df = pd.DataFrame(sample_data)
        
        # 添加社交媒體數據 (基於報告中的數據)
        df['social_engagement_twitter'] = df['peak_viewers'] * np.random.uniform(0.12, 0.18, len(df))
        df['social_engagement_reddit'] = df['peak_viewers'] * np.random.uniform(0.06, 0.10, len(df))
        df['social_engagement_youtube'] = df['peak_viewers'] * np.random.uniform(0.08, 0.15, len(df))
        
        # 添加其他欄位
        df['duration_minutes'] = np.random.normal(35, 8, len(df)).clip(20, 60)
        df['stream_platform'] = np.random.choice(['Multi-platform', 'YouTube', 'Twitch'], len(df), p=[0.6, 0.3, 0.1])
        df['tournament_tier'] = df['tournament_type'].map({
            'World Championship': 'International',
            'MSI': 'International', 
            'Regional League': 'Regional'
        })
        
        # 添加共同直播數據
        df['co_streaming_hours'] = df['hours_watched'] * np.random.uniform(0.35, 0.50, len(df))  # 基於44%的共同直播佔比
        
        # 添加語言分解數據
        language_patterns = {
            'Global': 'Korean:35%|English:25%|Chinese:20%|Vietnamese:8%|Portuguese:7%|Other:5%',
            'Korea': 'Korean:85%|English:10%|Other:5%',
            'China': 'Chinese:90%|English:7%|Other:3%',
            'Europe': 'English:45%|French:20%|Spanish:15%|German:12%|Other:8%',
            'North America': 'English:80%|Spanish:12%|Other:8%',
            'Brazil': 'Portuguese:85%|English:10%|Other:5%',
            'Vietnam': 'Vietnamese:80%|English:15%|Other:5%'
        }
        df['language_breakdown'] = df['region'].map(language_patterns)
        
        return df
    
    def _prepare_data(self):
        """Prepare and clean viewership data for analysis"""
        # Convert date column if not already datetime
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        # Extract year and month for analysis
        self.df['year'] = self.df['date'].dt.year
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        
        # Calculate viewership metrics
        self.df['hours_watched_millions'] = self.df['hours_watched'] / 1000000
        
        # Handle missing columns
        if 'viewership_average' not in self.df.columns:
            self.df['viewership_average'] = self.df['peak_viewers'] * 0.6  # 估計平均觀看數
        
        if 'peak_viewers' in self.df.columns and 'viewership_peak' not in self.df.columns:
            self.df['viewership_peak'] = self.df['peak_viewers']
        
        self.df['peak_to_avg_ratio'] = self.df['viewership_peak'] / self.df['viewership_average'].replace(0, 1)
        
        # Handle language breakdown
        self.df['primary_language'] = self.df['language_breakdown'].apply(self._extract_primary_language)
        
        # Add match_id for compatibility
        if 'match_id' not in self.df.columns:
            self.df['match_id'] = range(len(self.df))
        
        print(f"數據準備完成: {len(self.df)} 場賽事, 涵蓋 {self.df['year'].min()}-{self.df['year'].max()}")
    
    def _extract_primary_language(self, language_str: str) -> str:
        """Extract primary language from language breakdown string"""
        try:
            if pd.isna(language_str):
                return 'Korean'
            
            # Parse format like "Korean:35%|English:25%"
            languages = language_str.split('|')
            if languages:
                first_lang = languages[0].split(':')[0]
                return first_lang
            return 'Korean'
        except Exception:
            return 'Korean'
    
    def create_viewership_distribution_chart(self) -> go.Figure:
        """
        Create interactive viewership distribution chart across tournaments
        建立跨賽事觀看數據分布的互動式圖表 (最重要的目標)
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
        
        # Create subplots for comprehensive analysis
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '各賽事峰值觀看人數對比 Peak Viewership by Tournament', 
                '觀看人數分布 Viewership Distribution',
                '總觀看時數 (百萬小時) Total Hours Watched', 
                '峰值vs平均觀看比例 Peak vs Average Ratio'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Peak viewership by tournament (Bar Chart)
        fig.add_trace(
            go.Bar(
                x=tournament_stats['league'],
                y=tournament_stats['viewership_peak_mean'],
                name='平均峰值觀看',
                marker_color='#1f77b4',
                text=[f'{x:,.0f}' for x in tournament_stats['viewership_peak_mean']],
                textposition='auto',
                hovertemplate='賽事: %{x}<br>平均峰值: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Plot 2: Viewership distribution (Box plot)
        tournaments = self.df['league'].unique()
        for tournament in tournaments:
            tournament_data = self.df[self.df['league'] == tournament]['viewership_peak']
            fig.add_trace(
                go.Box(
                    y=tournament_data,
                    name=tournament,
                    boxpoints='outliers',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Plot 3: Total hours watched (Horizontal Bar)
        fig.add_trace(
            go.Bar(
                x=tournament_stats['hours_watched_millions_sum'],
                y=tournament_stats['league'],
                orientation='h',
                name='總觀看時數',
                marker_color='#ff7f0e',
                text=[f'{x:.1f}M' for x in tournament_stats['hours_watched_millions_sum']],
                textposition='auto',
                hovertemplate='賽事: %{y}<br>總時數: %{x:.1f}M小時<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Plot 4: Peak to Average ratio (Line + Markers)
        avg_ratios = self.df.groupby('league')['peak_to_avg_ratio'].mean()
        fig.add_trace(
            go.Scatter(
                x=avg_ratios.index,
                y=avg_ratios.values,
                mode='markers+lines+text',
                name='峰值/平均比例',
                marker=dict(size=12, color='#2ca02c'),
                line=dict(width=3),
                text=[f'{x:.1f}x' for x in avg_ratios.values],
                textposition='top center',
                hovertemplate='賽事: %{x}<br>比例: %{y:.1f}x<extra></extra>'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='英雄聯盟電競觀看數據分布分析<br>League of Legends Esports Viewership Distribution Analysis',
            height=800,
            showlegend=False,
            template='plotly_white',
            font=dict(size=12)
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="賽事 Tournament", row=1, col=1)
        fig.update_yaxes(title_text="峰值觀看人數 Peak Viewership", row=1, col=1)
        fig.update_xaxes(title_text="賽事 Tournament", row=1, col=2)
        fig.update_yaxes(title_text="觀看人數 Viewership", row=1, col=2)
        fig.update_xaxes(title_text="觀看時數 (百萬小時) Hours (M)", row=2, col=1)
        fig.update_yaxes(title_text="賽事 Tournament", row=2, col=1)
        fig.update_xaxes(title_text="賽事 Tournament", row=2, col=2)
        fig.update_yaxes(title_text="峰值/平均比例 Peak/Avg Ratio", row=2, col=2)
        
        return fig
    
    def create_social_media_heatmap(self) -> go.Figure:
        """
        Create social media interaction heat map
        建立社交媒體互動熱力圖，展示觀眾偏好與比賽特徵的關聯
        """
        # 準備社交媒體數據
        social_columns = ['social_engagement_twitter', 'social_engagement_reddit', 'social_engagement_youtube']
        
        # 建立賽事 vs 社交平台的熱力圖數據
        heatmap_data = []
        tournaments = sorted(self.df['league'].unique())
        platforms = ['Twitter', 'Reddit', 'YouTube']
        
        for tournament in tournaments:
            tournament_data = self.df[self.df['league'] == tournament]
            row_data = []
            for platform in ['twitter', 'reddit', 'youtube']:
                col_name = f'social_engagement_{platform}'
                avg_engagement = tournament_data[col_name].mean() / 1000000  # 轉換為百萬
                row_data.append(avg_engagement)
            heatmap_data.append(row_data)
        
        # 建立熱力圖
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=platforms,
            y=tournaments,
            colorscale='Viridis',
            text=[[f'{val:.1f}M' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 12, "color": "white"},
            hoverongaps=False,
            colorbar=dict(title="互動數 (百萬) Engagement (M)")
        ))
        
        fig.update_layout(
            title='社交媒體互動熱力圖<br>Social Media Engagement Heat Map by Tournament',
            xaxis_title="社交媒體平台 Social Media Platform",
            yaxis_title="賽事 Tournament",
            template='plotly_white',
            height=600,
            font=dict(size=12)
        )
        
        return fig
    
    def create_market_trend_analysis(self) -> go.Figure:
        """
        Create comprehensive market trend analysis
        建立全面的市場趨勢分析
        """
        # 建立多個子圖
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '觀看數據成長趨勢 Viewership Growth Over Time', 
                '區域市場佔有率 Regional Market Share',
                '共同直播影響分析 Co-streaming Impact', 
                '賽事等級表現 Tournament Tier Performance'
            ),
            specs=[[{"secondary_y": True}, {"type": "pie"}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: 時間序列成長趨勢
        yearly_data = self.df.groupby('year').agg({
            'viewership_peak': 'mean',
            'hours_watched_millions': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['viewership_peak'],
                mode='lines+markers',
                name='峰值觀看人數',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=10),
                hovertemplate='年份: %{x}<br>峰值: %{y:,.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=yearly_data['year'],
                y=yearly_data['hours_watched_millions'],
                mode='lines+markers',
                name='總觀看時數 (M)',
                line=dict(color='#ff7f0e', width=2),
                marker=dict(size=8),
                yaxis='y2',
                hovertemplate='年份: %{x}<br>時數: %{y:.1f}M<extra></extra>'
            ),
            row=1, col=1, secondary_y=True
        )
        
        # Plot 2: 區域市場圓餅圖
        regional_data = self.df.groupby('region')['hours_watched_millions'].sum()
        
        fig.add_trace(
            go.Pie(
                labels=regional_data.index,
                values=regional_data.values,
                name="區域市場",
                hole=0.3,
                textinfo='label+percent',
                hovertemplate='區域: %{label}<br>時數: %{value:.1f}M<br>佔比: %{percent}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Plot 3: 共同直播影響分析
        if 'co_streaming_hours' in self.df.columns:
            self.df['costream_ratio'] = self.df['co_streaming_hours'] / self.df['hours_watched']
            costream_analysis = self.df.groupby('league').agg({
                'costream_ratio': 'mean',
                'viewership_peak': 'mean'
            }).reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=costream_analysis['costream_ratio'] * 100,
                    y=costream_analysis['viewership_peak'],
                    mode='markers+text',
                    text=costream_analysis['league'],
                    textposition='top center',
                    name='共同直播影響',
                    marker=dict(size=15, color='#d62728', opacity=0.7),
                    hovertemplate='賽事: %{text}<br>共同直播比例: %{x:.1f}%<br>峰值: %{y:,.0f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Plot 4: 賽事等級表現
        tier_data = self.df.groupby('tournament_tier').agg({
            'viewership_peak': 'mean',
            'hours_watched_millions': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=tier_data['tournament_tier'],
                y=tier_data['viewership_peak'],
                name='峰值觀看',
                marker_color='#9467bd',
                text=[f'{x:,.0f}' for x in tier_data['viewership_peak']],
                textposition='auto',
                hovertemplate='等級: %{x}<br>峰值: %{y:,.0f}<extra></extra>'
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='英雄聯盟電競市場趨勢分析<br>League of Legends Esports Market Trend Analysis',
            height=900,
            showlegend=True,
            template='plotly_white',
            font=dict(size=12)
        )
        
        # 更新坐標軸標籤
        fig.update_xaxes(title_text="年份 Year", row=1, col=1)
        fig.update_yaxes(title_text="峰值觀看人數 Peak Viewership", row=1, col=1)
        fig.update_yaxes(title_text="觀看時數 (百萬) Hours (M)", row=1, col=1, secondary_y=True)
        
        fig.update_xaxes(title_text="共同直播比例 (%) Co-stream Ratio", row=2, col=1)
        fig.update_yaxes(title_text="峰值觀看人數 Peak Viewership", row=2, col=1)
        
        fig.update_xaxes(title_text="賽事等級 Tournament Tier", row=2, col=2)
        fig.update_yaxes(title_text="峰值觀看人數 Peak Viewership", row=2, col=2)
        
        return fig
    
    def create_audience_preference_analysis(self) -> go.Figure:
        """
        Analyze audience preferences across different dimensions
        分析不同維度的觀眾偏好
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '語言偏好分布 Language Preference', 
                '比賽時長 vs 觀看數 Duration vs Viewership',
                '串流平台熱度 Platform Popularity', 
                '月份觀看模式 Monthly Patterns'
            ),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: 語言偏好圓餅圖
        language_counts = self.df['primary_language'].value_counts()
        
        fig.add_trace(
            go.Pie(
                labels=language_counts.index,
                values=language_counts.values,
                name="語言偏好",
                hole=0.3,
                textinfo='label+percent'
            ),
            row=1, col=1
        )
        
        # Plot 2: 比賽時長 vs 觀看數散點圖
        fig.add_trace(
            go.Scatter(
                x=self.df['duration_minutes'],
                y=self.df['viewership_peak'],
                mode='markers',
                name='時長影響',
                marker=dict(
                    size=10,
                    color=self.df['tournament_tier'].map({'International': '#ff7f0e', 'Regional': '#1f77b4'}),
                    opacity=0.7
                ),
                text=self.df['league'],
                hovertemplate='時長: %{x:.0f} 分鐘<br>峰值: %{y:,}<br>賽事: %{text}<extra></extra>'
            ),
            row=1, col=2
        )
        
        # Plot 3: 串流平台長條圖
        platform_data = self.df.groupby('stream_platform').agg({
            'viewership_peak': 'mean',
            'match_id': 'count'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(
                x=platform_data['stream_platform'],
                y=platform_data['viewership_peak'],
                name='平台熱度',
                marker_color='#2ca02c',
                text=[f'{x:,.0f}' for x in platform_data['viewership_peak']],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # Plot 4: 月份觀看模式
        monthly_pattern = self.df.groupby('month')['viewership_peak'].mean()
        month_names = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
        
        fig.add_trace(
            go.Scatter(
                x=[month_names[i-1] for i in monthly_pattern.index],
                y=monthly_pattern.values,
                mode='lines+markers',
                name='月份模式',
                line=dict(color='#d62728', width=3),
                marker=dict(size=10)
            ),
            row=2, col=2
        )
        
        # 更新布局
        fig.update_layout(
            title='觀眾偏好分析<br>Audience Preference Analysis',
            height=800,
            showlegend=True,
            template='plotly_white',
            font=dict(size=12)
        )
        
        # 更新坐標軸
        fig.update_xaxes(title_text="比賽時長 (分鐘) Duration (min)", row=1, col=2)
        fig.update_yaxes(title_text="峰值觀看人數 Peak Viewership", row=1, col=2)
        fig.update_xaxes(title_text="串流平台 Platform", row=2, col=1)
        fig.update_yaxes(title_text="平均峰值觀看 Avg Peak", row=2, col=1)
        fig.update_xaxes(title_text="月份 Month", row=2, col=2)
        fig.update_yaxes(title_text="平均峰值觀看 Avg Peak", row=2, col=2)
        
        return fig
    
    def generate_viewership_summary(self) -> Dict:
        """
        Generate comprehensive viewership summary statistics
        產生全面的觀看數據摘要統計
        """
        summary = {
            'total_tournaments': len(self.df),
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
            'tournament_tiers': self.df['tournament_tier'].value_counts().to_dict(),
            'years_covered': f"{self.df['year'].min()}-{self.df['year'].max()}",
            'peak_tournament': self.df.loc[self.df['viewership_peak'].idxmax(), 'tournament_name']
        }
        
        return summary
    
    def _calculate_growth_rate(self) -> float:
        """Calculate year-over-year growth rate"""
        yearly_data = self.df.groupby('year')['viewership_peak'].mean()
        if len(yearly_data) >= 2:
            latest_year = yearly_data.index.max()
            previous_years = yearly_data.index[yearly_data.index < latest_year]
            if len(previous_years) > 0:
                previous_year = previous_years.max()
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
        summary_df.to_csv(f'{output_dir}viewership_summary.csv', index=False, encoding='utf-8-sig')
        
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
        
        tournament_analysis.to_csv(f'{output_dir}tournament_analysis.csv', encoding='utf-8-sig')
        
        # Export yearly trends
        yearly_trends = self.df.groupby('year').agg({
            'viewership_peak': ['mean', 'max', 'sum'],
            'hours_watched': 'sum',
            'social_engagement_twitter': 'sum',
            'social_engagement_reddit': 'sum',
            'social_engagement_youtube': 'sum'
        }).round(2)
        
        yearly_trends.to_csv(f'{output_dir}yearly_trends.csv', encoding='utf-8-sig')
        
        print(f"分析結果已匯出至: {output_dir}")

def main():
    """Demo script showing how to use the analyzer"""
    print("=== 英雄聯盟電競觀眾分析器 ===")
    print("正在初始化...")
    
    analyzer = ViewershipAnalyzer()
    
    # Generate summary
    summary = analyzer.generate_viewership_summary()
    print(f"\n📊 觀看數據摘要:")
    print(f"總賽事數量: {summary['total_tournaments']:,}")
    print(f"總觀看時數: {summary['total_hours_watched']:,.0f}")
    print(f"平均峰值觀看人數: {summary['avg_peak_viewership']:,.0f}")
    print(f"最高峰值觀看人數: {summary['max_peak_viewership']:,.0f}")
    print(f"最受歡迎賽事: {summary['top_tournament']}")
    print(f"最受歡迎地區: {summary['most_popular_region']}")
    print(f"年度成長率: {summary['growth_rate']:.1f}%")
    print(f"涵蓋年份: {summary['years_covered']}")
    
    print("\n📈 正在建立視覺化圖表...")
    
    try:
        # 建立觀眾分布圖表
        dist_fig = analyzer.create_viewership_distribution_chart()
        dist_fig.write_html('c:/Users/shiou/Documents/DataVisualization/viewership_distribution.html')
        print("✅ 觀眾分布圖表已儲存")
        
        # 建立社交媒體熱力圖
        social_fig = analyzer.create_social_media_heatmap()
        social_fig.write_html('c:/Users/shiou/Documents/DataVisualization/social_media_heatmap.html')
        print("✅ 社交媒體熱力圖已儲存")
        
        # 建立市場趨勢分析
        market_fig = analyzer.create_market_trend_analysis()
        market_fig.write_html('c:/Users/shiou/Documents/DataVisualization/market_trend_analysis.html')
        print("✅ 市場趨勢分析已儲存")
        
        # 建立觀眾偏好分析
        preference_fig = analyzer.create_audience_preference_analysis()
        preference_fig.write_html('c:/Users/shiou/Documents/DataVisualization/audience_preference_analysis.html')
        print("✅ 觀眾偏好分析已儲存")
        
        # 匯出分析結果
        analyzer.export_analysis_results()
        print("✅ 分析結果已匯出")
        
    except Exception as e:
        print(f"❌ 圖表建立時發生錯誤: {e}")
    
    print("\n🎉 分析完成!")
    return analyzer

if __name__ == "__main__":
    analyzer = main()
