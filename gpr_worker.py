# ──────────────────────────────────────────────
# STEP 6. Gaussian Process Regression (GPR)
# ──────────────────────────────────────────────
print("\n" + "=" * 50)
print("STEP 6. Gaussian Process Regression (GPR)")
print("=" * 50)

"""
GPR이란?
  KNN은 "가장 가까운 이웃"을 찾는 방식이지만
  GPR은 데이터 분포 전체를 학습해서 확률적으로 위치를 추정함.

  핵심 차이:
  - KNN → "가장 비슷한 K개 위치의 평균"
  - GPR → "학습 데이터 기반 확률 분포에서 가장 가능성 높은 위치"
            + 예측 불확실도(신뢰구간)도 함께 출력 ← GPR만의 장점!

  커널(Kernel) 종류:
  ┌─────────────────────────────────────────────────────┐
  │ RBF(Radial Basis Function)                          │
  │   - 데이터가 무한히 매끄럽다고 가정                 │
  │   - 노이즈 적은 환경에서 유리                       │
  │                                                     │
  │ Matern(nu=2.5)                                      │
  │   - 데이터가 어느 정도 거칠어도 OK                  │
  │   - RSSI처럼 노이즈 많은 데이터에 더 적합           │
  │                                                     │
  │ ConstantKernel * 커널                               │
  │   - 커널의 전체 진폭(크기)을 조정하는 스케일러      │
  │                                                     │
  │ WhiteKernel                                         │
  │   - 센서 노이즈, 측정 오차를 모델링                 │
  └─────────────────────────────────────────────────────┘
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    Matern, WhiteKernel, RBF, ConstantKernel as C
)
from sklearn.preprocessing import StandardScaler
import numpy as np

X_train = np.loadtxt('x_train_1833.csv', delimiter=',', skiprows=1)
X_test = np.loadtxt('x_test_1833.csv', delimiter=',', skiprows=1)
y_lat_train = np.loadtxt('y_lat_train_1833.csv', delimiter=',', skiprows=1)
y_lat_test = np.loadtxt('y_lat_test_1833.csv', delimiter=',', skiprows=1)
y_lon_train = np.loadtxt('y_lon_train_1833.csv', delimiter=',', skiprows=1)
y_lon_test = np.loadtxt('y_lon_test_1833.csv', delimiter=',', skiprows=1)

# ── 전처리: 스케일링 ──────────────────────────
# GPR은 입력 스케일에 매우 민감 → StandardScaler 필수
scaler_X = StandardScaler() # 범위를 0~1로
X_train_s = scaler_X.fit_transform(X_train) # 학습하면서 mu와 sigma 계산하고, 이를 통해 정규화
X_test_s  = scaler_X.transform(X_test) # x_train 기반으로 학습한 mu와 sigma를 단순히 적용하여, 정규화함

# y도 정규화 (커널 최적화 안정성을 위해)
lat_mean, lat_std = y_lat_train.mean(), y_lat_train.std()
lon_mean, lon_std = y_lon_train.mean(), y_lon_train.std()
y_lat_train_n = (y_lat_train - lat_mean) / lat_std
y_lon_train_n = (y_lon_train - lon_mean) / lon_std


# ── 커널 설정: 두 가지 버전 비교 ──────────────
"""
어떤 커널이 더 나을지, 파라미터는 어떻게 조정할지는 직접 실험해보는 것이 최선!
"""

import json

kernel_A = C(1.0, (1e-3, 1e3))*(Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5)
            + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 10)))

kernel_B = (C(1.0, (1e-3, 1e3))
            * RBF(length_scale=10, length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 10)))

gpr_results_m = {}
gpr_results_r = {}

for kernel_name, kernel in [('Matern', kernel_A), ('C*RBF', kernel_B)]:
    print(f"\n[{kernel_name} 커널] 학습 중...")

    gpr_lat = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        n_restarts_optimizer=5, random_state=42
    )
    gpr_lon = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6,
        n_restarts_optimizer=5, random_state=42
    )
    gpr_lat.fit(X_train_s, y_lat_train_n) # lat 학습
    gpr_lon.fit(X_train_s, y_lon_train_n) # lon 학습

    # 예측 + 불확실도(std) 함께 반환
    pred_lat_n, std_lat = gpr_lat.predict(X_test_s, return_std=True)
    pred_lon_n, std_lon = gpr_lon.predict(X_test_s, return_std=True)

    # 역정규화
    pred_lat_gpr = pred_lat_n * lat_std + lat_mean
    pred_lon_gpr = pred_lon_n * lon_std + lon_mean

    # 오차 계산
    # 1. 거리 오차
    dists = np.sqrt(
        ((y_lat_test - pred_lat_gpr) * 111000)**2 +
        ((y_lon_test - pred_lon_gpr) * 88000)**2
    )
    # 2. 불확실성(std_lat * lat_std: 상대적 오차 * 실제 눈금의 크기)
    unc_m = np.sqrt(
        (std_lat * lat_std * 111000)**2 +
        (std_lon * lon_std * 88000)**2
    )

    print(f"  평균 오차:   {dists.mean():.1f}m")
    print(f"  중앙값 오차: {np.median(dists):.1f}m")
    print(f"  최소/최대:   {dists.min():.1f}m / {dists.max():.1f}m")
    print(f"  불확실도:    평균 {unc_m.mean():.1f}m")
    # 코드의 장점: 최적화된 커널 파라미터 출력
    # → 커널이 데이터에서 어떤 패턴을 학습했는지 확인 가능
    print(f"  최적 커널:   {gpr_lat.kernel_}") # kernel_: 최적화된 하이퍼파라미터 가짐
    print(f"  최적 커널:   {gpr_lon.kernel_}")


    if kernel_name == "Matern":
      gpr_results_m["model_type"]= "GPR"
      gpr_results_m["parameters"] = {
        "kernel": kernel_name,
        "lat_nu": gpr_lat.kernel_.get_params()['k2__k1__nu'],
        "lon_nu": gpr_lon.kernel_.get_params()['k2__k1__nu'],
        "lat_length_scale": gpr_lat.kernel_.get_params()['k2__k1__length_scale'],
        "lon_length_scale": gpr_lon.kernel_.get_params()['k2__k1__length_scale'],
        "lat_noise_level": gpr_lat.kernel_.get_params()['k2__k2__noise_level'],
        "lon_noise_level": gpr_lon.kernel_.get_params()['k2__k2__noise_level'],
        "scaler": "StandardScaler"
      }
      gpr_results_m["summary_metrics"] = {
        "mean_error_m": dists.mean(),
        "median_error_m": np.median(dists),
        "mean_uncertainty_m": unc_m.mean(),
        "min_error_m": dists.min(),
        "max_error_m": dists.max()
      }
      gpr_results_m["gpr_results"] = []

      for i in range(len(dists)):
        gpr_results_m['gpr_results'].append(
          {
            "point_id": i,
            "actual_coords": [y_lat_test[i], y_lon_test[i]],
            "pred_coords": [pred_lat_gpr[i], pred_lon_gpr[i]],
            "error_m": dists[i],
            "unc_m": unc_m[i],
            "std_lat": std_lat[i],
            "std_lon": std_lon[i]
          }
        )
      with open('gpr_matern_07.json', 'w', encoding='utf-8') as f:
        json.dump(gpr_results_m, f, ensure_ascii=False, indent=4)
    else:
      gpr_results_r["model_type"] = "GPR"
      gpr_results_r["parameters"] = {
        "kernel": kernel_name,
        "lat_length_scale": gpr_lat.kernel_.get_params()['k1__k2__length_scale'],
        "lon_length_scale": gpr_lon.kernel_.get_params()['k1__k2__length_scale'],
        "lat_noise_level": gpr_lat.kernel_.get_params()['k2__noise_level'],
        "lon_noise_level": gpr_lon.kernel_.get_params()['k2__noise_level'],
        "scaler": "StandardScaler"
      }
      gpr_results_r["summary_metrics"] = {
        "mean_error_m": dists.mean(),
        "median_error_m": np.median(dists),
        "mean_uncertainty_m": unc_m.mean(),
        "min_error_m": dists.min(),
        "max_error_m": dists.max()
      }
      gpr_results_r["gpr_results"] = []


      for i in range(len(dists)):
        gpr_results_r['gpr_results'].append(
          {
            "point_id": i,
            "actual_coords": [y_lat_test[i], y_lon_test[i]],
            "pred_coords": [pred_lat_gpr[i], pred_lon_gpr[i]],
            "error_m": dists[i],
            "unc_m": unc_m[i],
            "std_lat": std_lat[i],
            "std_lon": std_lon[i]
          }
        )
      with open('gpr_crbf_07.json', 'w', encoding='utf-8') as f:
        json.dump(gpr_results_r, f, ensure_ascii=False, indent=4)


# ── KNN vs GPR 전체 비교 ──────────────────────
print(f"\n{'='*50}")
print("===== GPR 최종 비교 =====")
print(f"{'방법':>10} | {'평균 오차':>10} | {'중앙값 오차':>12} | {'최소 오차':>10}")
print("-" * 50)
# dists_knn = np.sqrt( # 리스트 단위 계산
#     ((y_lat_test - pred_lat_all) * 111000)**2 +
#     ((y_lon_test - pred_lon_all) * 88000)**2
# )
# print(f"{'KNN':>10} | {dists_knn.mean():>8.1f}m | {np.median(dists_knn):>10.1f}m | {dists_knn.min():>8.1f}m  (K={K})")

print(f"{'GPR('+gpr_results_m['parameters']['kernel']+')':>10}", end='|')
print(f"{gpr_results_m['summary_metrics']['mean_error_m']:>8.1f}m", end='|')
print(f"{gpr_results_m['summary_metrics']['median_error_m']:>10.1f}m", end='|')
print(f"{gpr_results_m['summary_metrics']['min_error_m']:>8.1f}m")

print(f"{'GPR('+gpr_results_r['parameters']['kernel']+')':>10}", end='|')
print(f"{gpr_results_r['summary_metrics']['mean_error_m']:>8.1f}m", end='|')
print(f"{gpr_results_r['summary_metrics']['median_error_m']:>10.1f}m", end='|')
print(f"{gpr_results_r['summary_metrics']['min_error_m']:>8.1f}m")


# 시각화용으로 Matern 결과 사용
# best_gpr = gpr_results['Matern']
pred_lat_gpr = [res['pred_coords'][0] for res in gpr_results_m['gpr_results']]
pred_lon_gpr = [res['pred_coords'][1] for res in gpr_results_m['gpr_results']]
dists_gpr    = [res['error_m'] for res in gpr_results_m['gpr_results']]
dist_gpr     = gpr_results_m['summary_metrics']['mean_error_m']
unc_m        = [res['unc_m'] for res in gpr_results_m['gpr_results']]