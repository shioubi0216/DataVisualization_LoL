import os
import time
import subprocess
import sys

def run_command(command, description):
    """執行指定命令並顯示說明"""
    print(f"\n{'='*80}")
    print(f"執行: {description}")
    print(f"{'='*80}")
    
    # 使用虛擬環境中的 Python
    if command.startswith('python'):
        command = f"lol_venv\\Scripts\\python.exe {command[7:]}"
    
    start_time = time.time()
    process = subprocess.run(command, shell=True)
    end_time = time.time()
    
    if process.returncode == 0:
        print(f"\n✓ {description} 執行成功! (用時: {end_time - start_time:.2f}秒)")
    else:
        print(f"\n✗ {description} 執行失敗! 錯誤碼: {process.returncode}")
    
    return process.returncode == 0

def main():
    """執行完整分析流程"""
    # 確保所有必要目錄存在
    processed_dir = "Oracle's_Elixir_data_Processed"
    figures_dir = os.path.join(processed_dir, "figures")
    
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # 1. 資料收集 (確認資料可讀取)
    success = run_command("python data_collection.py", "資料收集與檢查")
    if not success:
        print("資料檢查失敗，無法繼續後續步驟!")
        return
    
    # 2. 資料探索
    run_command("python data_exploration.py", "資料探索")
    
    # 3. 資料清洗與準備
    success = run_command("python data_cleaning.py", "資料清洗與準備")
    if not success:
        print("資料清洗失敗，無法繼續後續步驟!")
        return
    
    # 4. 英雄分析
    run_command("python champion_analysis.py", "英雄選用分析")
    
    # 5. 英雄趨勢分析
    run_command("python champion_trends.py", "英雄趨勢分析")
    
    # 6. 啟動互動式儀表板
    print("\n資料分析與視覺化已完成!")
    print("\n要啟動互動式儀表板，請執行以下命令:")
    print("lol_venv\\Scripts\\python.exe interactive_dashboard.py")
    
    launch_dashboard = input("\n是否現在啟動儀表板? (y/n): ")
    if launch_dashboard.lower() == 'y':
        run_command("python interactive_dashboard.py", "啟動互動式儀表板")

if __name__ == "__main__":
    main()