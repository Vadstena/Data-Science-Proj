from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import DataFrame, concat, read_csv
from pandas.plotting import register_matplotlib_converters
from ds_charts import get_variable_types
from matplotlib.pyplot import subplots, show, savefig

register_matplotlib_converters()

def get_variable_types(df: DataFrame) -> dict:
    variable_types: dict = {
        'Numeric': [],
        'Binary': [],
        'Date': [],
        'Symbolic': []
    }
    for c in df.columns:
        uniques = df[c].dropna(inplace=False).unique()
        if c == 'date':
            print("\nMONEY")
            print(df[c])
        if len(uniques) == 2:
            variable_types['Binary'].append(c)
            df[c].astype('bool')
            print(df[c])
            print("\n # # # # # #")
        elif df[c].dtype == 'datetime64[ns]':
            variable_types['Date'].append(c)
        elif df[c].dtype == 'int':
            variable_types['Numeric'].append(c)
        elif df[c].dtype == 'float':
            variable_types['Numeric'].append(c)
        else:
            df[c].astype('category')
            variable_types['Symbolic'].append(c)

    return variable_types


file = 'drought'
#filename = 'data/diabetic_data.csv'
filename = 'data/drought.csv'
data = read_csv(filename, index_col='date', na_values='', parse_dates=True, infer_datetime_format=True)
data['date'] = to_datetime(data['date'], format='%d/%m/%Y')
data.shape


variable_types = get_variable_types(data)
numeric_vars = variable_types['Numeric']
symbolic_vars = variable_types['Symbolic']
boolean_vars = variable_types['Binary']
date_vars = variable_types['Date']

df_nr = data[numeric_vars]
df_sb = data[symbolic_vars]
df_bool = data[boolean_vars]
df_date = data[date_vars]

transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_zscore = concat([df_date, tmp, df_sb,  df_bool], axis=1)
norm_data_zscore.to_csv(f'data/{file}_scaled_zscore.csv', index=False)

transf = MinMaxScaler(feature_range=(0, 1), copy=True).fit(df_nr)
tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
norm_data_minmax = concat([df_date, tmp, df_sb,  df_bool], axis=1)
norm_data_minmax.to_csv(f'data/{file}_scaled_minmax.csv', index=False)
print(norm_data_minmax.describe())


fig, axs = subplots(1, 3, figsize=(18,5),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
norm_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
norm_data_minmax.boxplot(ax=axs[0, 2])
axs[0, 0].set_xticklabels([])
axs[0, 1].set_xticklabels([])
axs[0, 2].set_xticklabels([])
savefig('images/scaling.png')
