import os
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt    

# 데이터 파일 경로 판단
cur_dir = os.getcwd()
dir_name = os.path.basename(cur_dir)
if dir_name == 'tutorial01_ML':
    data_file = os.path.join(cur_dir, 'sales_data.csv') 
elif dir_name.startswith('lecture-tukorea-20260109'):
    data_file = os.path.join(cur_dir, 'tutorial01_ML', 'sales_data.csv')
else: 
    raise ValueError('Please run this script from the proper directory.')

if data_file is None or not os.path.exists(data_file):
    raise FileNotFoundError(f'Data file not found: {data_file}')

# 데이터 불러오기
df = pd.read_csv(data_file)

# 날짜 형식 변환
df['Date'] = pd.to_datetime(df['Date'])

# 특정 매장 및 상품 선택
target_store = 'S001'
target_product = 'P0001'
df = df[(df["Store ID"] == target_store) \
               & (df["Product ID"] == target_product)].copy()
# 데이터를 날짜 오름차순 정렬
df = df.sort_values("Date")

# Prophet 모델에 맞게 컬럼명 변경
df = df.rename(columns={"Date": "ds", "Demand": "y"})

# 결측치 처리
df["y"] = df["y"].fillna(0)
df["Discount"] = df["Discount"].fillna(0)
df["Promotion"] = df["Promotion"].fillna(0)
df["Competitor Pricing"] = df["Competitor Pricing"].fillna(df["Competitor Pricing"].mean())

# 학습/검증 데이터 분리
testset_days = 60
train_df = df.iloc[:-testset_days]
test_df = df.iloc[-testset_days:]

# Prophet 모델 생성
model = Prophet(
    growth='linear',
    yearly_seasonality=True, 
    weekly_seasonality=True,
    daily_seasonality=False, 
    changepoint_prior_scale=0.05, 
    seasonality_prior_scale=10.0
)
# 회귀 변수 추가
model.add_regressor('Discount')
model.add_regressor('Promotion')  
model.add_regressor('Competitor Pricing')

# 모델 학습
print("모델 학습 중...")
model.fit(train_df)
print("모델 학습 완료")

# 미래 데이터 초기화
future = model.make_future_dataframe(periods=testset_days)
# 미래 시점의 회귀 변수 추가
future = future.merge(
    df[['ds', 'Discount', 'Promotion', 'Competitor Pricing']], 
    on='ds', how='left')

# 미래 수요 예측
forecast = model.predict(future)

# 예측 성능 평가 (MAPE)
y_true = test_df['y'].values
y_pred = forecast['yhat'].iloc[-testset_days:].values
mape = (abs(y_true - y_pred) / y_true).mean() * 100
print(f'예측 정확도 (MAPE) = {mape:.2f}%')

# 예측 결과 차트 출력
fig = model.plot(forecast)
ax = fig.gca()
ax.axvline(train_df['ds'].max(), color='red', linestyle='--')
ax.scatter(test_df['ds'], test_df['y'], color='red', s=10)
plt.show()