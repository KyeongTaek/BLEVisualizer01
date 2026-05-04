# D
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기 (data_provider가 만든 파일을 쓴다고 가정)
try:
    df = pd.read_csv('all_sensor_data_merge.csv')
    print("데이터 로드 성공!")
    
    # 2. RSSI 신호 강도 분포 그리기 (발표 자료용)
    plt.figure(figsize=(10, 5))
    sns.histplot(df['rssi'], kde=True, color='blue')
    plt.title('RSSI Signal Distribution')
    plt.xlabel('RSSI (dBm)')
    plt.ylabel('Frequency')
    plt.show()

except FileNotFoundError:
    print("파일을 찾을 수 없습니다. 경로를 확인해주세요.")


# 팀원들이 파일을 주면 이 부분을 활성화할 겁니다!
def plot_comparison(knn_errors, gpr_errors):
    plt.figure(figsize=(10, 6))
    
    # KNN 오차 분포 (팀원 B의 결과)
    sns.kdeplot(knn_errors, label='KNN Error', fill=True)
    
    # GPR 오차 분포 (팀원 C의 결과)
    sns.kdeplot(gpr_errors, label='GPR Error', fill=True)
    
    plt.title('Performance Comparison: KNN vs GPR')
    plt.xlabel('Error Distance (m)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# 예시: 나중에 이런 식으로 호출할 거예요.
# plot_comparison(df_knn['error'], df_gpr['error'])