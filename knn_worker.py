import json
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

# 실행 모드 설정
# python knn_worker.py original  -> 원본 데이터 실험
# python knn_worker.py merge     -> merge 데이터 실험
mode = sys.argv[1] if len(sys.argv) > 1 else "original"

if mode == "original":
    X_train_path = "X_train.csv"
    X_test_path = "X_test.csv"
    y_train_path = "y_train.csv"
    y_test_path = "y_test.csv"

    error_analysis_output = "knn_error_analysis.csv"
    json_output = "knn_result.json"

elif mode == "merge":
    X_train_path = "data provider/merge_X_train.csv"
    X_test_path = "data provider/merge_X_test.csv"
    y_train_path = "data provider/merge_y_train.csv"
    y_test_path = "data provider/merge_y_test.csv"

    error_analysis_output = "merge_knn_error_analysis.csv"
    json_output = "merge_knn_result.json"


print("=" * 60)
print(f"KNN 실험 모드: {mode}")
print("=" * 60)
print("X_train:", X_train_path)
print("X_test :", X_test_path)
print("y_train:", y_train_path)
print("y_test :", y_test_path)
print()

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

def remove_unnamed_columns(df):
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


X_train = remove_unnamed_columns(X_train)
X_test = remove_unnamed_columns(X_test)
y_train = remove_unnamed_columns(y_train)
y_test = remove_unnamed_columns(y_test)

X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# train/test 컬럼 순서 맞추기
X_test = X_test[X_train.columns]

def extract_coordinates(y_df):
    columns = y_df.columns.tolist()

    if "lat" in columns and "lon" in columns:
        return y_df[["lat", "lon"]].values

    if "lat_r" in columns and "lon_r" in columns:
        return y_df[["lat_r", "lon_r"]].values

    if "latitude" in columns and "longitude" in columns:
        return y_df[["latitude", "longitude"]].values

    numeric_y = y_df.select_dtypes(include=[np.number])

    if numeric_y.shape[1] < 2:
        raise ValueError("y_train/y_test에서 위도와 경도 컬럼을 찾을 수 없습니다.")

    return numeric_y.iloc[:, :2].values

y_train_values = extract_coordinates(y_train)
y_test_values = extract_coordinates(y_test)


print("=" * 60)
print("데이터 로드 완료")
print("=" * 60)
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train_values.shape)
print("y_test :", y_test_values.shape)
print()

def calculate_distance_m(actual, predicted):
    """
    실제 좌표와 예측 좌표의 거리 오차를 m 단위로 계산한다.
    강의자료/예시 코드 기준 근사식:
    위도 1도 ≈ 111,000m
    경도 1도 ≈ 88,000m, 북위 36도 기준
    """
    lat_error_m = (actual[:, 0] - predicted[:, 0]) * 111000
    lon_error_m = (actual[:, 1] - predicted[:, 1]) * 88000

    distance_m = np.sqrt(lat_error_m ** 2 + lon_error_m ** 2)

    return distance_m


k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
metrics = ["euclidean", "manhattan"]
weights_options = ["uniform", "distance"]


experiment_results = []

best_result = None
best_pred = None
best_errors = None

print("=" * 60)
print("KNN 실험 시작")
print("=" * 60)

for k in k_values:
    if k > len(X_train):
        continue

    for metric in metrics:
        for weights in weights_options:

            # KNN 회귀 모델 생성
            model = KNeighborsRegressor(
                n_neighbors=k,
                metric=metric,
                weights=weights
            )

            # 모델 학습
            model.fit(X_train, y_train_values)

            # 테스트 RSSI 데이터 넣어 위치 좌표 예측
            pred = model.predict(X_test)

            # 예측 좌표와 실제 좌표 간 거리 오차 계산
            errors = calculate_distance_m(y_test_values, pred)

            # 실험 결과 정리
            result = {
                "k": k,
                "metric": metric,
                "weights": weights,
                "mean_error_m": float(np.mean(errors)),
                "median_error_m": float(np.median(errors)),
                "min_error_m": float(np.min(errors)),
                "max_error_m": float(np.max(errors))
            }

            # 전체 실험 결과 목록에 추가
            experiment_results.append(result)

            print(
                f"K={k:>2}, "
                f"metric={metric:<9}, "
                f"weights={weights:<8} "
                f"-> mean={result['mean_error_m']:.2f}m, "
                f"median={result['median_error_m']:.2f}m"
            )

            if best_result is None or result["mean_error_m"] < best_result["mean_error_m"]:
                best_result = result
                best_pred = pred
                best_errors = errors


print()

# K값별 오차 분석표 저장
error_table = pd.DataFrame(experiment_results)
error_table = error_table.sort_values(by="mean_error_m")

error_table.to_csv(
    error_analysis_output,
    index=False,
    encoding="utf-8-sig"
)

print("=" * 60)
print("최적 KNN 결과")
print("=" * 60)
print(f"K값        : {best_result['k']}")
print(f"거리 방식  : {best_result['metric']}")
print(f"가중치 방식: {best_result['weights']}")
print(f"평균 오차  : {best_result['mean_error_m']:.2f}m")
print(f"중앙값 오차: {best_result['median_error_m']:.2f}m")
print(f"최소 오차  : {best_result['min_error_m']:.2f}m")
print(f"최대 오차  : {best_result['max_error_m']:.2f}m")
print()


#knn_result.json 생성
test_results = []

for i in range(len(y_test_values)):
    test_results.append({
        "point_id": int(i),
        "actual_coords": [
            float(y_test_values[i][0]),
            float(y_test_values[i][1])
        ],
        "pred_coords": [
            float(best_pred[i][0]),
            float(best_pred[i][1])
        ],
        "error_m": float(best_errors[i])
    })


final_result = {
    "model_type": "KNN",
    "dataset" : mode,
    "parameters": {
        "k": int(best_result["k"]),
        "metric": best_result["metric"],
        "weights": best_result["weights"],
        "imputer_strategy": "mean", # 결측치 처리 방식
    },
    "summary_metrics": {
        "mean_error_m": float(best_result["mean_error_m"]),
        "median_error_m": float(best_result["median_error_m"]),
        "min_error_m": float(best_result["min_error_m"]),
        "max_error_m": float(best_result["max_error_m"])
    },
    "test_results": test_results
}


with open(json_output, "w", encoding="utf-8") as f:
    json.dump(final_result, f, indent=2, ensure_ascii=False)


print("=" * 60)
print("파일 저장 완료")
print("=" * 60)
print(f"{error_analysis_output} 저장 완료")
print(f"{json_output} 저장 완료")