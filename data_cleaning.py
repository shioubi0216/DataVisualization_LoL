import pandas as pd
import numpy as np
import os

def clean_esports_data(year="2022"):
    """清洗英雄聯盟電競數據"""
    input_path = f"Oracle's_Elixir_data/{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    output_path = f"Oracle's_Elixir_data_Processed/{year}_LoL_esports_match_data_from_OraclesElixir_cleaned.csv"
    
    if not os.path.exists(input_path):
        print(f"找不到輸入檔案: {input_path}")
        return None
    
    # 載入原始資料
    print(f"正在處理{year}年資料...")
    df = pd.read_csv(input_path)
    original_rows = df.shape[0]
    
    # 1. 篩選出有效比賽記錄 (只保留有英雄名稱的記錄)
    df = df[df['champion'].notna()]
    
    # 2. 處理缺失值 - 針對關鍵欄位
    # 對於數值型欄位，用0填充
    numeric_cols = ['kills', 'deaths', 'assists', 'gold', 'damagetochampions']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    # 3. 標準化欄位名稱並建立新的衍生欄位
    # 建立KDA欄位 (如果不存在)
    if 'kills' in df.columns and 'deaths' in df.columns and 'assists' in df.columns:
        df['kda'] = (df['kills'] + df['assists']) / df['deaths'].replace(0, 1)  # 避免除以零
    
    # 4. 篩選主要賽區
    major_leagues = ['LPL', 'LCK', 'LEC', 'LCS', 'WCS', 'MSI'] # 主要賽區+國際賽
    if 'league' in df.columns:
        df_major = df[df['league'].isin(major_leagues)]
        print(f"主要賽區資料筆數: {df_major.shape[0]}")
    else:
        df_major = df
    
    # 5. 儲存處理後的完整數據和主要賽區數據
    df.to_csv(output_path, index=False)
    
    if 'league' in df.columns:
        major_output_path = f"Oracle's_Elixir_data_Processed/{year}_major_leagues.csv"
        df_major.to_csv(major_output_path, index=False)
    
    print(f"資料清洗完成! 原始資料: {original_rows}筆, 清洗後: {df.shape[0]}筆")
    print(f"清洗後資料已儲存至 {output_path}")
    
    return df

def prepare_champion_data(years=["2022", "2023", "2024"]):
    """準備英雄選用分析用的資料集"""
    combined_data = []
    
    for year in years:
        filepath = f"Oracle's_Elixir_data_Processed/{year}_LoL_esports_match_data_from_OraclesElixir_cleaned.csv"
        if not os.path.exists(filepath):
            print(f"找不到檔案: {filepath}")
            continue
            
        df = pd.read_csv(filepath)
        
        # 只保留需要的欄位
        if 'champion' in df.columns and 'position' in df.columns and 'league' in df.columns:
            keep_cols = ['gameid', 'league', 'date', 'champion', 'position', 'side', 'result', 
                         'kills', 'deaths', 'assists', 'patch']
            keep_cols = [col for col in keep_cols if col in df.columns]
            
            champion_df = df[keep_cols].copy()
            champion_df['year'] = year
            combined_data.append(champion_df)
    
    if not combined_data:
        print("沒有可用的資料!")
        return None
        
    # 合併多年資料
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # 儲存英雄分析用的資料集
    output_path = "Oracle's_Elixir_data_Processed/champion_analysis_data.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"英雄分析資料已儲存至 {output_path}")
    
    return combined_df

# 執行資料清洗
if __name__ == "__main__":
    # 清洗2022-2024年資料
    clean_esports_data("2022")
    clean_esports_data("2023")
    clean_esports_data("2024")  
    
    # 準備英雄分析資料
    champion_data = prepare_champion_data(["2022", "2023", "2024"])