import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_champion_data():
    """載入英雄分析資料"""
    filepath = "Oracle's_Elixir_data_Processed/champion_analysis_data.csv"
    if not os.path.exists(filepath):
        print(f"找不到檔案: {filepath}")
        return None
    
    df = pd.read_csv(filepath)
    
    # 確保日期格式正確
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def analyze_champion_trends_over_time(df, top_champions=8, save_fig=True):
    """分析英雄選用趨勢隨時間的變化"""
    if 'champion' not in df.columns or 'date' not in df.columns:
        print("資料中缺少需要的欄位")
        return
    
    # 確保日期欄位是datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 按月份聚合資料
    df['month'] = df['date'].dt.to_period('M')
    
    # 計算每個月各英雄的出場次數
    monthly_picks = df.groupby(['month', 'champion']).size().reset_index(name='count')
    
    # 計算每月總比賽場次
    monthly_games = df.groupby('month')['gameid'].nunique() * 2  # 兩隊
    
    # 計算每月選用率
    monthly_picks['total_games'] = monthly_picks['month'].map(monthly_games)
    monthly_picks['pick_rate'] = monthly_picks['count'] / monthly_picks['total_games'] * 100
    
    # 找出總選用率最高的N個英雄
    top_champions_list = df['champion'].value_counts().head(top_champions).index.tolist()
    
    # 篩選這些英雄的月度資料
    top_monthly_data = monthly_picks[monthly_picks['champion'].isin(top_champions_list)]
    
    # 將月份轉換回datetime以便繪圖
    top_monthly_data['month_date'] = top_monthly_data['month'].dt.to_timestamp()
    
    # 視覺化 - 折線圖
    plt.figure(figsize=(14, 8))
    
    # 為每個英雄繪製一條線
    for champion in top_champions_list:
        champion_data = top_monthly_data[top_monthly_data['champion'] == champion]
        plt.plot(
            'month_date', 'pick_rate', 
            data=champion_data, 
            marker='o', 
            markersize=5, 
            linewidth=2, 
            label=champion
        )
    
    # 設置圖表格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.gcf().autofmt_xdate()  # 自動調整日期標籤
    
    plt.title('熱門英雄選用率趨勢', fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('選用率 (%)', fontsize=12)
    plt.legend(title='英雄', loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        output_path = "data/analysis/figures/champion_trends_over_time.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    return top_monthly_data

def analyze_champion_by_league(df, save_fig=True):
    """分析不同賽區的英雄選擇偏好差異"""
    if 'champion' not in df.columns or 'league' not in df.columns:
        print("資料中缺少需要的欄位")
        return
    
    # 只考慮主要賽區
    major_leagues = ['LPL', 'LCK', 'LEC', 'LCS']
    league_df = df[df['league'].isin(major_leagues)]
    
    # 計算每個賽區各英雄的出場次數
    league_picks = league_df.groupby(['league', 'champion']).size().reset_index(name='count')
    
    # 計算每個賽區的總比賽場次
    league_games = league_df.groupby('league')['gameid'].nunique() * 2  # 兩隊
    
    # 計算選用率
    league_picks['total_games'] = league_picks['league'].map(league_games)
    league_picks['pick_rate'] = league_picks['count'] / league_picks['total_games'] * 100
    
    # 對每個賽區找出選用率最高的前5名英雄
    top_by_league = league_picks.groupby('league').apply(
        lambda x: x.nlargest(5, 'pick_rate')
    ).reset_index(drop=True)
    
    # 視覺化 - 群組條形圖
    plt.figure(figsize=(14, 10))
    
    # 為每個賽區創建子圖
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, league in enumerate(major_leagues):
        league_data = top_by_league[top_by_league['league'] == league]
        sns.barplot(
            x='pick_rate', y='champion', 
            data=league_data, 
            palette='viridis', 
            ax=axes[i]
        )
        axes[i].set_title(f'{league} 賽區熱門英雄', fontsize=14)
        axes[i].set_xlabel('選用率 (%)', fontsize=12)
        axes[i].set_ylabel('英雄', fontsize=12)
    
    plt.tight_layout()
    
    if save_fig:
        output_path = "data/analysis/figures/champion_by_league.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    # 計算賽區間英雄選擇的相似度
    pivot_table = league_picks.pivot_table(
        index='champion', 
        columns='league', 
        values='pick_rate', 
        fill_value=0
    )
    
    # 計算相關性
    correlation = pivot_table.corr()
    
    # 視覺化相關性
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
    plt.title('各賽區英雄選擇偏好相關性', fontsize=16)
    
    if save_fig:
        output_path = "Oracle's_Elixir_data_Processed/figures/league_similarity_heatmap.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    return top_by_league, correlation

# 主函數
if __name__ == "__main__":
    # 載入分析資料
    df = load_champion_data()
    
    if df is not None:
        # 1. 分析英雄選用率隨時間的變化趨勢
        time_trends = analyze_champion_trends_over_time(df)
        
        # 2. 分析不同賽區的英雄選擇偏好差異
        league_analysis, league_similarity = analyze_champion_by_league(df)
        
        # 保存分析結果
        time_trends.to_csv("data/analysis/champion_time_trends.csv", index=False)
        league_analysis.to_csv("data/analysis/champion_by_league.csv", index=False)
        league_similarity.to_csv("data/analysis/league_similarity_matrix.csv")