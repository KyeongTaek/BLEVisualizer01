import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd

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

def run_visualize(prefix=""):
    curr_dir = os.getcwd()
    knn_f = os.path.join(curr_dir, f"{prefix}knn_result.json")
    gpr_f = os.path.join(curr_dir, f"{prefix}gpr_result.json")
    ens_f = os.path.join(curr_dir, f"{prefix}ensemble_result.json")

    knn_d, gpr_d, ens_d = load_full_data(knn_f), load_full_data(gpr_f), load_full_data(ens_f)

    if not (knn_d and gpr_d and ens_d):
        print(f"⚠️ {prefix} 관련 파일을 찾을 수 없습니다.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('RSSI 기반 KNN Fingerprinting 위치 측위', fontsize=14, fontweight='bold')

    # ── 그래프 1: K값별 오차 비교 ──
    df = pd.read_csv(f"{prefix}knn_error_analysis.csv")

    idx = df.groupby('k')['mean_error_m'].idxmin()
    df_best = df.loc[idx]

    results = df_best.sort_values(by='k').to_dict(orient='records')
    
    best = min(results, key=lambda x: x['mean_error_m'])
    best_k = best['k']
    best_mean_error = best['mean_error_m']

    ax1 = axes[0]

    ks     = [r['k'] for r in results]
    dist_ms = [r['mean_error_m'] for r in results]
    bars = ax1.bar(ks, dist_ms, color=['#ef4444' if r['k'] == best_k else '#3b82f6' for r in results],
                alpha=0.8, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('K 값')
    ax1.set_ylabel('평균 거리 오차 (m)')
    ax1.set_title('K값별 위치 추정 오차')
    ax1.set_xticks(ks)
    for bar, val in zip(bars, dist_ms):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}m', ha='center', va='bottom', fontsize=9)
    ax1.axhline(y=best_mean_error, color='#ef4444', linestyle='--', alpha=0.5, linewidth=1)


    # ── 그래프 2: 라디오맵 RSSI 히트맵 ──
    ax2 = axes[1]

    # 가장 데이터 많은 센서 선택
    radio_map = pd.read_csv(f"{prefix}radio_map.csv", index_col=['lat_r', 'lon_r'])
    
    # 위치 좌표 추출 (정답 레이블)
    coords = radio_map.index.to_frame(index=False)
    
    y_lat = coords['lat_r'].values
    y_lon = coords['lon_r'].values

    best_mac = radio_map.count().idxmax()
    heatmap_data = radio_map[best_mac].dropna().reset_index()
    sc = ax2.scatter(
        heatmap_data['lon_r'],
        heatmap_data['lat_r'],
        c=heatmap_data[best_mac],
        cmap='RdYlGn',
        vmin=-105, vmax=-40,
        s=80, edgecolors='white', linewidths=0.3
    )
    plt.colorbar(sc, ax=ax2, label='RSSI (dBm)')
    ax2.set_xlabel('경도')
    ax2.set_ylabel('위도')
    ax2.set_title(f'라디오맵 — {best_mac[:8]}... RSSI 분포')
    ax2.tick_params(labelsize=7)

    # ── 그래프 3: 실제 위치 vs 예측 위치 ──
    ax3 = axes[2]

    # 라디오맵 전체 위치 (회색 배경)
    y_lon_test = knn_d['actual'][:,1]
    pred_lon_all = knn_d['pred'][:,1]
    y_lat_test = knn_d['actual'][:,0]
    pred_lat_all = knn_d['pred'][:,0]

    ax3.scatter(y_lon, y_lat, c='#374151', s=30, alpha=0.4,
                zorder=1, label='라디오맵 위치')

    # 실제 vs 예측 연결선
    for i in range(len(knn_d['errors'])):
        ax3.plot(
            [y_lon_test[i], pred_lon_all[i]],
            [y_lat_test[i], pred_lat_all[i]],
            'gray', alpha=0.3, linewidth=0.8, zorder=2
        )

    # 실제 위치 (파랑)
    ax3.scatter(y_lon_test, y_lat_test, c='#3b82f6', s=60,
                zorder=3, label='실제 위치', edgecolors='white', linewidths=0.5)

    # 예측 위치 (빨강)
    ax3.scatter(pred_lon_all, pred_lat_all, c='#ef4444', s=60, marker='^',
                zorder=4, label='예측 위치', edgecolors='white', linewidths=0.5)

    ax3.set_xlabel('경도')
    ax3.set_ylabel('위도')
    ax3.set_title(f'실제 위치 vs 예측 위치 (K={best_k})')
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=7)

    plt.tight_layout()

    # 저장 파일명 구분
    save_path = "fingerprinting_result.png" if not prefix else "merge_fingerprinting_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"✅ 결과 저장됨: {save_path}")

if __name__ == "__main__":
    # 순차적으로 실행 (첫 번째 창을 닫으면 두 번째 창이 뜹니다)
    run_analysis(prefix="")
    run_analysis(prefix="merge_")
    run_visualize(prefix="")
    run_visualize(prefix="merge_")
