from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT

index_multi = 'date'
target_multi = 'QV2M'
data = read_csv('data/drought_forecasting_dataset.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True, dayfirst =True)
'''figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data_multi[target_multi], x_label=index_multi, y_label='consumption', title=target_multi)
plot_series(data_multi['class'])
xticks(rotation = 45)
savefig(f'images/drought_data_dim.png')'''

WIN_SIZE = 10
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_smooth_10.png')
smooth_df.to_csv(f'data/transform/smooth_10.csv', index=True)


WIN_SIZE = 100
rolling = data.rolling(window=WIN_SIZE)
smooth_df = rolling.mean()
figure(figsize=(3*HEIGHT, HEIGHT/2))
plot_series(smooth_df, title=f'Smoothing (win_size={WIN_SIZE})', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_smooth_100.png')
smooth_df.to_csv(f'data/transform/smooth_100.csv', index=True)



def aggregate_by(data: Series, index_var: str, period: str):
    index = data.index.to_period(period)
    agg_df = data.copy().groupby(index).mean()
    agg_df[index_var] = index.drop_duplicates().to_timestamp()
    agg_df.set_index(index_var, drop=True, inplace=True)
    return agg_df

figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'date', 'W')
plot_series(agg_df, title='Weekly', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_agg_weekly.png')


figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'date', 'M')
plot_series(agg_df, title='Monthly', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_agg_monthly.png')
agg_df.to_csv(f'data/transform/agg_m.csv', index=True)



figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'date', 'Q')
plot_series(agg_df, title='Quarterly', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_agg_quarterly.png')
agg_df.to_csv(f'data/transform/agg_q.csv', index=True)



figure(figsize=(3*HEIGHT, HEIGHT))
agg_df = aggregate_by(data, 'date', 'Y')
plot_series(agg_df, title='Yearly', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_agg_quarterly.png')
agg_df.to_csv(f'data/transform/agg_y.csv', index=True)



diff_df = data.diff()
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(diff_df, title='Differentiation', x_label='timestamp', y_label='QV2M')
xticks(rotation = 45)
savefig(f'images/drought_diff.png')
diff_df.to_csv(f'data/transform/diff.csv', index=True)

