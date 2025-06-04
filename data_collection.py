import pandas as pd
import os
import shutil

def create_directory_structure():
    """確保處理後的目錄結構存在"""
    processed_dir = "Oracle's_Elixir_data_Processed"
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
        print(f"創建目錄: {processed_dir}")
    
    # 創建圖表目錄
    figures_dir = os.path.join(processed_dir, "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"創建目錄: {figures_dir}")
    
    print("目錄結構已確認")

def copy_check_data(year):
    """檢查原始資料是否可讀取並顯示基本資訊"""
    source_path = f"Oracle's_Elixir_data/{year}_LoL_esports_match_data_from_OraclesElixir.csv"
    
    if not os.path.exists(source_path):
        print(f"找不到源檔案: {source_path}")
        return None
    
    try:
        print(f"正在載入{year}年賽事資料...")
        df = pd.read_csv(source_path)
        print(f"成功載入資料: {df.shape[0]}行 x {df.shape[1]}列")
        return df
    except Exception as e:
        print(f"載入資料時發生錯誤: {e}")
        return None

# 主函數
if __name__ == "__main__":
    create_directory_structure()
    # 檢查2022-2024年資料是否可讀取
    data_2022 = copy_check_data("2022")
    data_2023 = copy_check_data("2023")
    data_2024 = copy_check_data("2024")