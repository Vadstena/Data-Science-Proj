from pandas import read_csv, Series
from matplotlib.pyplot import figure, xticks, show, savefig
from ts_functions import plot_series, HEIGHT
from matplotlib.pyplot import subplots

#data = read_csv('data/drought_feat_selected.csv', index_col='date', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True)
#data_multi = read_csv('data/drought_feat_selected.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)


index_multi = 'date'
target_multi = 'QV2M'
data = read_csv('data/drought_forecasting_dataset.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True, dayfirst =True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(data, x_label='timestamp', y_label='QV2M',)
xticks(rotation = 45)
savefig(f'images/drought_data_dim.png')

index = data.index.to_period('W')
week_df = data.copy().groupby(index).mean()
week_df['timestamp'] = index.drop_duplicates().to_timestamp()
week_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(week_df, title='Weekly consumptions', x_label='timestamp', y_label='consumption')
xticks(rotation = 45)
savefig(f'images/drought_weekly.png')


index = data.index.to_period('M')
month_df = data.copy().groupby(index).mean()
month_df['timestamp'] = index.drop_duplicates().to_timestamp()
month_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(month_df, title='Monthly', x_label='timestamp', y_label='QVSM')
savefig(f'images/drought_monthly.png')

index = data.index.to_period('Q')
quarter_df = data.copy().groupby(index).mean()
quarter_df['timestamp'] = index.drop_duplicates().to_timestamp()
quarter_df.set_index('timestamp', drop=True, inplace=True)
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(quarter_df, title='Quarterly', x_label='timestamp', y_label='consumption')
savefig(f'images/drought_quarterly.png')



index = data.index.to_period('Q')
quart_df = data.copy().groupby(index).sum()
quart_df['date'] = index.drop_duplicates().to_timestamp()
quart_df.set_index('date', drop=True, inplace=True)
index = data.index.to_period('Y')
yearly_df = data.copy().groupby(index).sum()
yearly_df['date'] = index.drop_duplicates().to_timestamp()
yearly_df.set_index('date', drop=True, inplace=True)
_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT/2))
axs[0].grid(False)
axs[0].set_axis_off()
axs[0].set_title('Quarterly', fontweight="bold")
axs[0].text(0, 0, str(quart_df.describe()))
axs[1].grid(False)
axs[1].set_axis_off()
axs[1].set_title('Yearly', fontweight="bold")
axs[1].text(0, 0, str(yearly_df.describe()))
show()

_, axs = subplots(1, 2, figsize=(2*HEIGHT, HEIGHT))
quart_df.boxplot(ax=axs[0])
yearly_df.boxplot(ax=axs[1])
savefig(f'images/data_distr.png')




bins = (10, 25, 50)
_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for quarterly QV2M %d bins'%bins[j])
    axs[j].set_xlabel('QV2M')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(quart_df.values, bins=bins[j])
savefig(f'images/drought_quart_hist.png')

_, axs = subplots(1, len(bins), figsize=(len(bins)*HEIGHT, HEIGHT))
for j in range(len(bins)):
    axs[j].set_title('Histogram for yearly QV2M %d bins'%bins[j])
    axs[j].set_xlabel('QV2M')
    axs[j].set_ylabel('Nr records')
    axs[j].hist(week_df.values, bins=bins[j])
savefig(f'images/drought_yearly_hist.png')




from numpy import ones
from pandas import Series

dt_series = Series(data['QV2M'])

mean_line = Series(ones(len(dt_series.values)) * dt_series.mean(), index=dt_series.index)
series = {'QV2M': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='timestamp', y_label='QV2M', title='Stationary study', show_std=True)
savefig(f'images/drought_stationarity_study.png')

BINS = 10
line = []
n = len(dt_series)
for i in range(BINS):
    b = dt_series[i*n//BINS:(i+1)*n//BINS]
    mean = [b.mean()] * (n//BINS)
    line += mean
line += [line[-1]] * (n - len(line))
mean_line = Series(line, index=dt_series.index)
series = {'QV2M': dt_series, 'mean': mean_line}
figure(figsize=(3*HEIGHT, HEIGHT))
plot_series(series, x_label='time', y_label='QV2M', title='Stationary study', show_std=True)
savefig(f'images/drought_stationarity_study_2.png')
