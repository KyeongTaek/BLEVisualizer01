import json
import numpy as np

with open('gpr_crbf_07.json', 'r') as f:
  gpr_data = json.load(f)

# knn의 결과와 gpr의 결과를 6:4로 섞음
w_knn = 0.6
w_gpr = 0.4

ensemble_results = []
mean_error = 0 # 평균 오차 담을 변수
for i in range(len(gpr_data['gpr_results'])): # 각 데이터에 대해서
  knn_pred = np.array([pred_lat_all[i], pred_lon_all[i]]) # knn의 위도경도 예측값
  gpr_pred = np.array(gpr_data['gpr_results'][i]['pred_coords']) # gpr의 위도경도 예측값

  ensemble_pred = (w_knn * knn_pred) + (w_gpr * gpr_pred) # knn의 결과와 gpr의 결과를 섞음

  actual = np.array([y_lat_test[i], y_lon_test[i]]) # 실제 위도경도 값

  # 각 데이터의 실제 위도경도 값과 앙상블의 예측값 간 오차
  error = np.sqrt(
      ((actual[0] - ensemble_pred[0])*111000)**2
      + ((actual[1] - ensemble_pred[1])*88000)**2
  )
  mean_error += float(error) # 각 데이터의 오차 더함

  ensemble_results.append({
      "point_id": i,
      "actual_coords": actual.tolist(),
      "pred_coords": ensemble_pred.tolist(),
      "error_m": float(error)
  })

mean_error /= len(gpr_data['gpr_results']) # 평균 구함
print(f"ensemble mean error: {mean_error}m")
print(f"c*gpr mean error: {gpr_data['summary_metrics']['mean_error_m']}m")
print(f"knn mean error: {dists_knn.mean()}m")

ensemble_data = {
    "model_type": "ensemble",
    "ensemble_results": ensemble_results
}
with open('ensemble_00.json', 'w') as f:
  json.dump(ensemble_data, f, indent=4)