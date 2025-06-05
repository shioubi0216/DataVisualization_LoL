import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from lol_champion_zh_tw import translate_champion
from viewership_analysis import ViewershipAnalyzer
import plotly.graph_objects as go

# filepath: [interactive_dashboard.py](http://_vscodecontentref_/7)
# 載入分析資料
viewership_analyzer = ViewershipAnalyzer('viewership_market_data.csv')

def load_data():
    try:
        champion_data = pd.read_csv("Oracle's_Elixir_data_Processed/champion_analysis_data.csv")
        # 嘗試用 format='mixed' 解析所有格式
        try:
            champion_data['date'] = pd.to_datetime(champion_data['date'], format='mixed', errors='coerce')
        except Exception:
            # 若 pandas 版本不支援 format='mixed'，則用自動推斷
            champion_data['date'] = pd.to_datetime(champion_data['date'], errors='coerce')
        return champion_data
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        # 創建一個空的DataFrame作為備用
        return pd.DataFrame()

# 載入資料
df = load_data()

# 初始化Dash應用
app = dash.Dash(__name__, title="英雄聯盟電競分析")

# 設定背景圖樣式
# 使用正確的 assets 路徑，並移除 opacity，改用 CSS 覆蓋層
background_style = {
    'className': 'bg-image-container',
    'style': {'position': 'relative', 'width': '100%', 'height': '100%'}
}

hero_section_style = {
    'className': 'bg-image-container',
    'style': {'position': 'relative', 'width': '100%', 'height': '100%'}
}

# 圖片對應頁面
background_images = {
    'viewership': '/assets/觀看人數分布_背景圖.jpg',
    'social': '/assets/社交媒體互動熱力圖_背景圖.jpg',
    'market': '/assets/市場趨勢分析_背景圖.jpg',
    'audience': '/assets/觀眾偏好分析_背景圖.jpg',
    'hero': '/assets/英雄分析區域_背景圖.jpg',
}

# 設計版面配置
app.layout = html.Div([
    html.H1("英雄聯盟電競資料視覺化", style={'textAlign': 'center'}),
    html.H2("觀眾與市場分析"),
    dcc.Tabs([
        dcc.Tab(label="觀看人數分布", children=[
            html.Div([
                html.Div(className='bg-image', style={'backgroundImage': f"url('{background_images['viewership']}')"}),
                html.Div([
                    dcc.Graph(id='viewership-distribution-chart')
                ], className='content')
            ], className='bg-image-container', style={'position': 'relative', 'width': '100%', 'height': '100%'})
        ]),
        dcc.Tab(label="社交媒體互動熱力圖", children=[
            html.Div([
                html.Div(className='bg-image', style={'backgroundImage': f"url('{background_images['social']}')"}),
                html.Div([
                    dcc.Graph(id='social-media-heatmap')
                ], className='content')
            ], className='bg-image-container', style={'position': 'relative', 'width': '100%', 'height': '100%'})
        ]),
        dcc.Tab(label="市場趨勢分析", children=[
            html.Div([
                html.Div(className='bg-image', style={'backgroundImage': f"url('{background_images['market']}')"}),
                html.Div([
                    dcc.Graph(id='market-trend-analysis')
                ], className='content')
            ], className='bg-image-container', style={'position': 'relative', 'width': '100%', 'height': '100%'})
        ]),
        dcc.Tab(label="觀眾偏好分析", children=[
            html.Div([
                html.Div(className='bg-image', style={'backgroundImage': f"url('{background_images['audience']}')"}),
                html.Div([
                    dcc.Graph(id='audience-preference-analysis')
                ], className='content')
            ], className='bg-image-container', style={'position': 'relative', 'width': '100%', 'height': '100%'})
        ]),
    ]),
    html.Div([
        # 過濾控制區
        html.Div([
            html.H3("過濾條件"),
            
            html.Label("選擇賽區"),
            dcc.Dropdown(
                id='league-dropdown',
                options=[{'label': league, 'value': league} for league in sorted(df['league'].unique()) if pd.notna(league)] if not df.empty and 'league' in df.columns else [],
                value=[],
                multi=True
            ),
            
            html.Label("選擇年份"),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': year, 'value': year} for year in sorted(df['year'].unique()) if pd.notna(year)] if not df.empty and 'year' in df.columns else [],
                value=df['year'].max() if not df.empty and 'year' in df.columns else None
            ),
            
            html.Label("選擇位置"),
            dcc.Dropdown(
                id='position-dropdown',
                options=[{'label': pos, 'value': pos} for pos in sorted(df['position'].unique()) if pd.notna(pos)] if not df.empty and 'position' in df.columns else [],
                value=[],
                multi=True
                ),
        ], style={'width': '20%', 'display': 'inline-block', 'vertical-align': 'top', 'padding': '20px'}),
        
        # 圖表展示區
        html.Div([
            html.Div(className='bg-image', style={'backgroundImage': f"url('{background_images['hero']}')"}),
            html.Div([
                html.H3("英雄選用率分析"),
                
                dcc.Graph(id='champion-pickrate-graph'),
                
                html.H3("英雄勝率分析"),
                
                dcc.Graph(id='champion-winrate-graph')
                
            ], className='content')
        ], style={'width': '75%', 'display': 'inline-block', 'padding': '20px', 'position': 'relative'}, className='bg-image-container'),
        
    ]),
    
    # 詳細數據表
    html.Div([
        html.H3("英雄詳細數據表"),
        
        html.Div(id='champion-stats-table')
        
    ], style={'padding': '20px', 'position': 'relative'}, className='bg-image-container')
])

@app.callback(
    Output('viewership-distribution-chart', 'figure'),
    Input('league-dropdown', 'value')  # 可根據需要加更多 Input
)
def update_viewership_distribution(selected_leagues):
    # 可根據選擇篩選資料
    return viewership_analyzer.create_viewership_distribution_chart()

@app.callback(
    Output('social-media-heatmap', 'figure'),
    Input('league-dropdown', 'value')
)
def update_social_media_heatmap(selected_leagues):
    return viewership_analyzer.create_social_media_heatmap()

@app.callback(
    Output('market-trend-analysis', 'figure'),
    Input('league-dropdown', 'value')
)
def update_market_trend_analysis(selected_leagues):
    return viewership_analyzer.create_market_trend_analysis()

@app.callback(
    Output('audience-preference-analysis', 'figure'),
    Input('league-dropdown', 'value')
)
def update_audience_preference_analysis(selected_leagues):
    return viewership_analyzer.create_audience_preference_analysis()

# 定義回調函數 - 選用率圖表
@app.callback(
    Output('champion-pickrate-graph', 'figure'),
    [Input('league-dropdown', 'value'),
     Input('year-dropdown', 'value'),
     Input('position-dropdown', 'value')]
)
def update_pickrate_graph(selected_leagues, selected_year, selected_positions):
        filtered_df = df.copy()
        if selected_year:
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_leagues:
            filtered_df = filtered_df[filtered_df['league'].isin(selected_leagues)]
        if selected_positions:
            filtered_df = filtered_df[filtered_df['position'].isin(selected_positions)]
        if filtered_df.empty:
            return px.bar(title="無符合條件的資料")
        champion_counts = filtered_df['champion'].value_counts().reset_index()
        champion_counts.columns = ['champion', 'count']
        total_games = filtered_df['gameid'].nunique() * 2
        champion_counts['pick_rate'] = champion_counts['count'] / total_games * 100
        champion_counts['champion_zh'] = champion_counts['champion'].apply(translate_champion)
        top_champions = champion_counts.head(15)
        fig = px.bar(
            top_champions,
            x='pick_rate',
            y='champion_zh',
            orientation='h',
            labels={'pick_rate': '選用率 (%)', 'champion_zh': '英雄'},
            title='英雄選用率 (前15名)',
            height=500
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        return fig

@app.callback(
        Output('champion-winrate-graph', 'figure'),
        [Input('league-dropdown', 'value'),
        Input('year-dropdown', 'value'),
        Input('position-dropdown', 'value')]
)
def update_winrate_graph(selected_leagues, selected_year, selected_positions):
        filtered_df = df.copy()
        if selected_year:
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_leagues:
            filtered_df = filtered_df[filtered_df['league'].isin(selected_leagues)]
        if selected_positions:
            filtered_df = filtered_df[filtered_df['position'].isin(selected_positions)]
        if filtered_df.empty or 'result' not in filtered_df.columns:
            return px.scatter(title="無符合條件的資料或缺少結果欄位")
        champion_stats = filtered_df.groupby('champion').agg(
            games=('result', 'count'),
            wins=('result', lambda x: sum(x == 1))
        ).reset_index()
        champion_stats['winrate'] = champion_stats['wins'] / champion_stats['games'] * 100
        total_games = filtered_df['gameid'].nunique() * 2
        champion_stats['pick_rate'] = champion_stats['games'] / total_games * 100
        champion_stats['champion_zh'] = champion_stats['champion'].apply(translate_champion)
        champion_stats = champion_stats[champion_stats['games'] >= 5]
        fig = px.scatter(
            champion_stats,
            x='pick_rate',
            y='winrate',
            size='games',
            hover_name='champion_zh',
            labels={'pick_rate': '選用率 (%)', 'winrate': '勝率 (%)', 'games': '出場次數', 'champion_zh': '英雄'},
            title='英雄選用率vs勝率',
            height=500
        )
        fig.add_hline(y=50, line_dash="dash", line_color="red", opacity=0.5)
        for _, row in champion_stats.nlargest(5, 'games').iterrows():
            fig.add_annotation(
                x=row['pick_rate'],
                y=row['winrate'],
                text=row['champion_zh'],
                showarrow=True,
                arrowhead=1
            )
        return fig

@app.callback(
        Output('champion-stats-table', 'children'),
        [Input('league-dropdown', 'value'),
        Input('year-dropdown', 'value'),
        Input('position-dropdown', 'value')]
    )
def update_stats_table(selected_leagues, selected_year, selected_positions):
        filtered_df = df.copy()
        if selected_year:
            filtered_df = filtered_df[filtered_df['year'] == selected_year]
        if selected_leagues:
            filtered_df = filtered_df[filtered_df['league'].isin(selected_leagues)]
        if selected_positions:
            filtered_df = filtered_df[filtered_df['position'].isin(selected_positions)]
        if filtered_df.empty or 'result' not in filtered_df.columns:
            return html.P("無符合條件的資料或缺少必要欄位")
        champion_stats = filtered_df.groupby('champion').agg(
            games=('result', 'count'),
            wins=('result', lambda x: sum(x == 1)),
            kills=('kills', 'mean'),
            deaths=('deaths', 'mean'),
            assists=('assists', 'mean')
        ).reset_index()
        champion_stats['winrate'] = (champion_stats['wins'] / champion_stats['games'] * 100).round(2)
        champion_stats['kda'] = ((champion_stats['kills'] + champion_stats['assists']) / champion_stats['deaths'].replace(0, 1)).round(2)
        champion_stats['champion_zh'] = champion_stats['champion'].apply(translate_champion)
        champion_stats = champion_stats.sort_values('games', ascending=False).head(20)
        formatted_stats = champion_stats[['champion_zh', 'games', 'winrate', 'kills', 'deaths', 'assists', 'kda']]
        formatted_stats.columns = ['英雄', '出場次數', '勝率(%)', '平均擊殺', '平均死亡', '平均助攻', 'KDA']
        for col in ['平均擊殺', '平均死亡', '平均助攻']:
            formatted_stats[col] = formatted_stats[col].round(2)
        table = html.Table([
            html.Thead(
                html.Tr([html.Th(col, style={'text-align': 'center', 'border': '1px solid #ddd', 'padding': '8px'}) for col in formatted_stats.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(formatted_stats.iloc[i][col], style={
                        'text-align': 'center' if col != '英雄' else 'left',
                        'border': '1px solid #ddd',
                        'padding': '8px'
                    }) for col in formatted_stats.columns
                ]) for i in range(len(formatted_stats))
            ])
        ], style={'width': '100%', 'border-collapse': 'collapse', 'font-size': '16px'})
        return table

# 運行應用
if __name__ == '__main__':
    app.run(debug=True)