import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_and_explore_data(year="2023"):
    """載入並初步探索資料集"""
    filepath = f"Oracle's_Elixir_data/{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    
    if not os.path.exists(filepath):
        print(f"找不到檔案: {filepath}")
        return None
    
    # 載入資料
    df = pd.read_csv(filepath)
    
    # 基本資訊
    print(f"\n{year}年資料基本資訊:")
    print(f"資料筆數: {df.shape[0]}, 欄位數: {df.shape[1]}")
    
    # 了解資料結構
    print("\n欄位資訊:")
    for col in df.columns:
        non_null = df[col].count()
        dtype = df[col].dtype
        unique = df[col].nunique() if df[col].dtype != 'float64' else None
        print(f"{col}: 類型={dtype}, 非空值數={non_null}, 唯一值數={unique}")
    
    # 檢查關鍵欄位
    print("\n重要分類變數的唯一值:")
    categorical_columns = ['league', 'position', 'side', 'champion', 'teamname', 'gameid']
    for col in categorical_columns:
        if col in df.columns:
            print(f"{col} 唯一值數量: {df[col].nunique()}")
            # 只印出前10個唯一值作為範例
            print(f"範例值: {df[col].dropna().unique()[:10]}")
    
    return df

def identify_data_structure(df):
    """識別資料結構中的規則和代碼意義"""
    # 識別隊伍編號規則
    if 'teamid' in df.columns and 'side' in df.columns:
        team_side_mapping = df[['teamid', 'side']].drop_duplicates()
        print("\n隊伍ID與比賽方的對應關係:")
        print(team_side_mapping.head(10))
    
    # 識別選手位置編號
    if 'position' in df.columns:
        positions = df['position'].dropna().unique()
        print("\n遊戲中的位置:")
        print(positions)
    
    # 檢查勝負記錄方式
    if 'result' in df.columns:
        results = df['result'].dropna().unique()
        print("\n比賽結果記錄方式:")
        print(results)
        
    return None

# 執行數據檢查
if __name__ == "__main__":
    # 載入2022-2024年資料進行探索
    for year in ["2022", "2023", "2024"]:
        df = load_and_explore_data(year)
        
        if df is not None:
            # 進一步識別資料結構中的規則
            identify_data_structure(df)
            
            # 儲存資料摘要
            output_dir = "Oracle's_Elixir_data_Processed"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(f"{output_dir}/data_summary_{year}.txt", "w") as f:
                f.write(f"資料筆數: {df.shape[0]}, 欄位數: {df.shape[1]}\n")
                f.write("\n重要欄位:\n")
                for col in df.columns[:30]:  # 記錄前30個欄位
                    f.write(f"- {col}\n")