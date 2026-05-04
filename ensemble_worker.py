import json
import numpy as np

with open('gpr_crbf_07.json', 'r') as f:
  gpr_data = json.load(f)

w_knn = 0.6
w_gpr = 0.4

ensemble_results = []
mean_error = 0
for i in range(len(gpr_data['gpr_results'])):
  knn_pred = np.array([pred_lat_all[i], pred_lon_all[i]])
  gpr_pred = np.array(gpr_data['gpr_results'][i]['pred_coords'])

  ensemble_pred = (w_knn * knn_pred) + (w_gpr * gpr_pred)

  actual = np.array([y_lat_test[i], y_lon_test[i]])

  error = np.sqrt(
      ((actual[0] - ensemble_pred[0])*111000)**2
      + ((actual[1] - ensemble_pred[1])*88000)**2
  )
  mean_error += float(error)

  ensemble_results.append({
      "point_id": i,
      "actual_coords": actual.tolist(),
      "pred_coords": ensemble_pred.tolist(),
      "error_m": float(error)
  })

mean_error /= len(gpr_data['gpr_results'])
print(f"ensemble mean error: {mean_error}m")
print(f"c*gpr mean error: {gpr_data['summary_metrics']['mean_error_m']}m")
print(f"knn mean error: {dists_knn.mean()}m")

ensemble_data = {
    "model_type": "ensemble",
    "ensemble_results": ensemble_results
}
with open('ensemble_00.json', 'w') as f:
  json.dump(ensemble_data, f, indent=4)