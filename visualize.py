import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 시각화 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Malgun Gothic')

def load_full_data(file_path):
    if not file_path or not os.path.exists(file_path): return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict) and 'test_results' in data:
                res = data['test_results']
                return {
                    "errors": [float(i['error_m']) for i in res if 'error_m' in i],
                    "actual": np.array([i['actual_coords'] for i in res if 'actual_coords' in i]),
                    "pred": np.array([i['pred_coords'] for i in res if 'pred_coords' in i]),
                    "unc": [float(i['unc_m']) for i in res if 'unc_m' in i]
                }
            return {"errors": [float(x) for x in (data.get('errors', []) if isinstance(data, dict) else data)]}
    except: return None

def run_analysis(prefix=""):
    curr_dir = os.getcwd()
    knn_f = os.path.join(curr_dir, f"{prefix}knn_result.json")
    gpr_f = os.path.join(curr_dir, f"{prefix}gpr_result.json")
    ens_f = os.path.join(curr_dir, f"{prefix}ensemble_result.json")

    knn_d, gpr_d, ens_d = load_full_data(knn_f), load_full_data(gpr_f), load_full_data(ens_f)

    if not (knn_d and gpr_d and ens_d):
        print(f"⚠️ {prefix} 관련 파일을 찾을 수 없습니다.")
        return

    # plt.figure()를 삭제하고 subplots만 사용합니다. (중복 창 방지)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    
    title_type = "Merge 데이터" if prefix else "원본 데이터"
    fig.suptitle(f'RSSI Fingerprinting 성능 분석 ({title_type})', fontsize=18, fontweight='bold')

    # (1) Box Plot
    axes[0].boxplot([knn_d['errors'], gpr_d['errors'], ens_d['errors']], labels=['KNN', 'GPR', 'Ensemble'], patch_artist=True)
    axes[0].set_title('거리 오차 분포')

    # (2) CDF Plot
    colors = ['#3b82f6', '#10b981', '#ef4444']
    for d, label, color in zip([knn_d, gpr_d, ens_d], ['KNN', 'GPR', 'Ensemble'], colors):
        sorted_err = np.sort(d['errors'])
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        axes[1].plot(sorted_err, cdf, label=f'{label} (Avg: {np.mean(sorted_err):.1f}m)', color=color, lw=3)
    axes[1].set_title('누적 오차 분포 (CDF)')
    axes[1].legend()

    # (3) GPR Scatter Analysis
    act, prd, unc = gpr_d['actual'], gpr_d['pred'], gpr_d['unc']
    for i in range(len(act)):
        axes[2].plot([act[i,1], prd[i,1]], [act[i,0], prd[i,0]], 'gray', alpha=0.3, lw=0.8)
    
    sc = axes[2].scatter(prd[:,1], prd[:,0], c=unc, cmap='YlOrRd', s=100, marker='^', edgecolors='white', label='GPR 예측')
    axes[2].scatter(act[:,1], act[:,0], c='#3b82f6', s=60, edgecolors='white', label='실제 위치')
    
    plt.colorbar(sc, ax=axes[2], label='불확실도 (m)')
    axes[2].set_title('GPR 공간 예측 및 불확실도')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    
    # 저장 파일명 구분
    save_path = "model_result_analysis.png" if not prefix else "merge_model_result_analysis.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 분석 완료 및 저장됨: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # 순차적으로 실행 (첫 번째 창을 닫으면 두 번째 창이 뜹니다)
    run_analysis(prefix="")
    run_analysis(prefix="merge_")