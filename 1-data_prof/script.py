from pandas import DataFrame, read_csv, to_datetime
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, show
from ds_charts import bar_chart


register_matplotlib_converters()

#filename = 'data/diabetic_data.csv'
filename = 'data/drought.csv'
data = read_csv(filename, na_values=0, parse_dates=True, infer_datetime_format=True)
#data = read_csv(filename, na_values='na')
data['date'] = to_datetime(data['date'], format='%d/%m/%Y')
data.shape

figure(figsize=(4,2))
values = {'nr records': data.shape[0], 'nr variables': data.shape[1]}
bar_chart(list(values.keys()), list(values.values()), title='Nr of records vs nr variables')
savefig('images/records_variables.png')

print(data.dtypes)

cat_vars = data.select_dtypes(include='object')
data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
data.dtypes

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


variable_types = get_variable_types(data)
print(variable_types)
counts = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])
figure(figsize=(4,2))
bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
savefig('images/variable_types.png')


mv = {}
'''for var in data:
	mv[var] = 0
	for r in data[var]:
		if r == 0: # r == "?"
			mv[var] = mv[var] + 1'''

for var in data:
    nr = data[var].isna().sum()
    if nr > 0:
        mv[var] = nr

figure(figsize=(13,10))
bar_chart(list(mv.keys()), list(mv.values()), title='Nr of missing values per variable',
            xlabel='variables', ylabel='nr missing values', rotation=True)
savefig('images/mv.png')
