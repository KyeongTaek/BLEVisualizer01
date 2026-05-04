import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
 
def get_processed_data():
    # 데이터 로드 
    df = pd.read_csv('all_sensor_data.csv', sep='\t')
    print(f"전체 데이터: {len(df)}행")
 
    # GPS 필터링
    df = df[
        (df['lat'] > 36.625) & (df['lat'] < 36.636) &
        (df['lon'] > 127.450) & (df['lon'] < 127.463)
    ].copy()
    print(f"GPS 필터 후: {len(df)}행")
 
    # mac 주소 통일, 유효 mac 사용
    df['mac'] = df['mac'].str.upper()
    VALID_MACS = [
        '88:A2:9E:9B:5E:6A',
        'B8:27:EB:D3:40:06',
        'D8:3A:DD:79:8E:BF',
        'D8:3A:DD:79:8F:80',
        'D8:3A:DD:C1:88:BD',
    ]
    df = df[df['mac'].isin(VALID_MACS)]
    print(f"유효 MAC 필터 후: {len(df)}행")
 
    # GPS 소수점 4자리로 반올림
    df['lat_r'] = df['lat'].round(4)
    df['lon_r'] = df['lon'].round(4)

    # 위치별, mac 별 평균 rssi 계산, 피벗 테이블 변환
    pivot = df.groupby(['lat_r', 'lon_r', 'mac'])['rssi'].mean().reset_index()
    radio_map = pivot.pivot_table(
        index=['lat_r', 'lon_r'],
        columns='mac',
        values='rssi'
    )
    radio_map.columns.name = None  # 컬럼명 정리
 
    # 최소 3개 센서 RSSI 위치만 사용
    radio_map = radio_map.dropna(thresh=3)
    print(f"\n라디오맵 위치(RP) 수: {len(radio_map)}개")
    print(f"사용 센서 수: {radio_map.shape[1]}개")
    print("\n라디오맵 미리보기:")
    print(radio_map.head(5).to_string())
 
    # 결측값 처리
    imputer = SimpleImputer(strategy='mean')
    X_raw = radio_map.values
    X_imputed = imputer.fit_transform(X_raw)
 
    # 위치 좌표 추출 
    coords = radio_map.index.to_frame(index=False)
    y_lat = coords['lat_r'].values
    y_lon = coords['lon_r'].values
    y = np.column_stack([y_lat, y_lon])
 
    # 학습 테스트 데이터 분리 80:20
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )
    print(f"\n학습 데이터: {len(X_train)}개 위치")
    print(f"테스트 데이터: {len(X_test)}개 위치")
 
    # CSV 저장
    mac_cols = list(radio_map.columns)
 
    pd.DataFrame(X_train, columns=mac_cols).to_csv('X_train.csv', index=False)
    pd.DataFrame(X_test,  columns=mac_cols).to_csv('X_test.csv',  index=False)
    pd.DataFrame(y_train, columns=['lat_r', 'lon_r']).to_csv('y_train.csv', index=False)
    pd.DataFrame(y_test,  columns=['lat_r', 'lon_r']).to_csv('y_test.csv',  index=False)
 
    print("\nCSV 저장 완료: X_train / X_test / y_train / y_test")
 
if __name__ == '__main__':
    get_processed_data()
