import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from lol_champion_zh_tw import translate_champion

# 設定中文字型 (根據您的環境調整)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_champion_data():
    """載入英雄分析資料"""
    filepath = "Oracle's_Elixir_data_Processed/champion_analysis_data.csv"
    if not os.path.exists(filepath):
        print(f"找不到檔案: {filepath}")
        return None
    
    return pd.read_csv(filepath)

def analyze_champion_pickrate(df, top_n=20, save_fig=True):
    """分析並視覺化英雄選用率"""
    if 'champion' not in df.columns:
        print("資料中缺少champion欄位")
        return
    
    # 計算每個英雄的出場次數
    champion_counts = df['champion'].value_counts()
    total_games = df['gameid'].nunique() * 2  # 每場比賽有兩支隊伍
    
    # 計算選用率
    pick_rates = (champion_counts / total_games * 100).reset_index()
    pick_rates.columns = ['champion', 'pick_rate']
    
    # 取前N個使用率最高的英雄
    top_champions = pick_rates.head(top_n).copy()
    top_champions['champion_zh'] = top_champions['champion'].apply(translate_champion)
    
    # 視覺化
    plt.figure(figsize=(12, 8))
    sns.barplot(x='pick_rate', y='champion_zh', data=top_champions, palette='viridis')
    plt.title(f'英雄聯盟職業賽選用率前{top_n}名英雄', fontsize=16)
    plt.xlabel('選用率 (%)', fontsize=12)
    plt.ylabel('英雄名稱', fontsize=12)
    plt.tight_layout()
    
    if save_fig:
        output_path = "Oracle's_Elixir_data_Processed/figures/champion_pick_rate.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    return pick_rates

def analyze_champion_by_position(df, save_fig=True):
    """按位置分析英雄選擇"""
    if 'champion' not in df.columns or 'position' not in df.columns:
        print("資料中缺少需要的欄位")
        return
    
    # 確保position欄位沒有缺失值
    df = df[df['position'].notna()]
    
    # 按位置分組計算英雄出現次數
    position_champions = df.groupby(['position', 'champion']).size().reset_index(name='count')
    
    # 對每個位置，找出使用率最高的5個英雄
    top_by_position = position_champions.groupby('position').apply(
        lambda x: x.nlargest(5, 'count')
    ).reset_index(drop=True)
    top_by_position['champion_zh'] = top_by_position['champion'].apply(translate_champion)
    
    # 計算每個位置的總出場次數
    position_totals = df.groupby('position').size()
    
    # 計算選用率
    top_by_position['pick_rate'] = top_by_position.apply(
        lambda row: row['count'] / position_totals[row['position']] * 100, axis=1
    )
    
    # 視覺化 - 分面圖
    positions = df['position'].unique()
    n_positions = len(positions)
    
    # 設定分面圖
    fig, axes = plt.subplots(1, n_positions, figsize=(4*n_positions, 6), sharey=False)
    
    for i, position in enumerate(sorted(positions)):
        position_data = top_by_position[top_by_position['position'] == position]
        position_data = position_data.sort_values('count', ascending=True)
        
        ax = axes[i] if n_positions > 1 else axes
        sns.barplot(x='count', y='champion_zh', data=position_data, palette='viridis', ax=ax)
        ax.set_title(f'{position} 位置', fontsize=14)
        ax.set_xlabel('出場次數', fontsize=12)
        
        if i == 0:
            ax.set_ylabel('英雄名稱', fontsize=12)
        else:
            ax.set_ylabel('')
    
    plt.tight_layout()
    
    if save_fig:
        output_path = "Oracle's_Elixir_data_Processed/figures/champion_by_position.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    return top_by_position

def analyze_champion_winrate(df, min_games=10, save_fig=True):
    """分析英雄勝率與選用率的關係"""
    if 'champion' not in df.columns or 'result' not in df.columns:
        print("資料中缺少需要的欄位")
        return
    
    # 確保結果欄位沒有缺失值
    df = df[df['result'].notna()]
    
    # 計算每個英雄的選用次數和勝場
    champion_stats = df.groupby('champion').agg(
        games=('result', 'count'),
        wins=('result', lambda x: sum(x == 1))  # 假設1代表獲勝
    ).reset_index()
    
    # 計算勝率
    champion_stats['winrate'] = champion_stats['wins'] / champion_stats['games'] * 100
    
    # 計算選用率
    total_games = df['gameid'].nunique() * 2  # 每場比賽有兩支隊伍
    champion_stats['pick_rate'] = champion_stats['games'] / total_games * 100
    
    # 篩選出出場次數足夠的英雄
    champion_stats = champion_stats[champion_stats['games'] >= min_games]
    champion_stats['champion_zh'] = champion_stats['champion'].apply(translate_champion)
    
    # 視覺化
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        champion_stats['pick_rate'],
        champion_stats['winrate'],
        s=champion_stats['games'] / 2,  # 點的大小代表出場次數
        alpha=0.6,
        c=champion_stats['games'],
        cmap='viridis',
        label=champion_stats['champion_zh']
    )
    
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.3)  # 50%勝率參考線
    
    # 標記一些特殊英雄
    top_pick_rate = champion_stats.nlargest(5, 'pick_rate')
    top_winrate = champion_stats.nlargest(5, 'winrate')
    
    for _, row in pd.concat([top_pick_rate, top_winrate]).drop_duplicates().iterrows():
        plt.annotate(
            row['champion_zh'],
            xy=(row['pick_rate'], row['winrate']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    plt.colorbar(scatter, label='出場次數')
    plt.title('英雄選用率與勝率關係', fontsize=16)
    plt.xlabel('選用率 (%)', fontsize=12)
    plt.ylabel('勝率 (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_fig:
        output_path = "Oracle's_Elixir_data_Processed/figures/champion_winrate_vs_pickrate.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"圖表已儲存至 {output_path}")
    
    plt.show()
    
    return champion_stats

# 主函數
if __name__ == "__main__":
    # 載入分析資料
    df = load_champion_data()
    
    if df is not None:
        # 1. 分析英雄選用率
        pick_rates = analyze_champion_pickrate(df)
        
        # 2. 按位置分析英雄選擇
        position_analysis = analyze_champion_by_position(df)
        
        # 3. 分析英雄勝率與選用率關係
        winrate_analysis = analyze_champion_winrate(df)
        
        # 保存分析結果
        pick_rates.to_csv("data/analysis/champion_pick_rates.csv", index=False)
        position_analysis.to_csv("data/analysis/champion_by_position.csv", index=False)
        winrate_analysis.to_csv("data/analysis/champion_winrate_analysis.csv", index=False)