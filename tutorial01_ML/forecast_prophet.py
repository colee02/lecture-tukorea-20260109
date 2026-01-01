import os
import pandas as pd
from prophet import Prophet
from matplotlib import pyplot as plt    

cur_dir = os.getcwd()
data_file = os.path.join(cur_dir, 'tutorial01_ML', 'sales_data.csv')
df = pd.read_csv(data_file)

df['Date'] = pd.to_datetime(df['Date'])

target_store = 'S001'
target_product = 'P0001'
df = df[(df["Store ID"] == target_store) \
               & (df["Product ID"] == target_product)].copy()
df = df.sort_values("Date")

df = df.rename(columns={"Date": "ds", "Demand": "y"})
df["y"] = df["y"].fillna(0)
df["Discount"] = df["Discount"].fillna(0)
df["Promotion"] = df["Promotion"].fillna(0)
df["Competitor Pricing"] = df["Competitor Pricing"].fillna(df["Competitor Pricing"].mean())

testset_days = 60
train_df = df.iloc[:-testset_days]
test_df = df.iloc[-testset_days:]

model = Prophet(
    growth='linear',
    yearly_seasonality=True, 
    weekly_seasonality=True,
    daily_seasonality=False, 
    changepoint_prior_scale=0.05, 
    seasonality_prior_scale=10.0
)
model.add_regressor('Discount')
model.add_regressor('Promotion')  
model.add_regressor('Competitor Pricing')

model.fit(train_df)

future = model.make_future_dataframe(periods=testset_days)
future = future.merge(
    df[['ds', 'Discount', 'Promotion', 'Competitor Pricing']], 
    on='ds', how='left')

forecast = model.predict(future)

y_true = test_df['y'].values
y_pred = forecast['yhat'].iloc[-testset_days:].values
mape = (abs(y_true - y_pred) / y_true).mean() * 100
print(f'MAPE: {mape:.2f}%')

fig = model.plot(forecast)
ax = fig.gca()
ax.axvline(train_df['ds'].max(), color='red', linestyle='--')
ax.scatter(test_df['ds'], test_df['y'], color='red', s=10)

plt.show()