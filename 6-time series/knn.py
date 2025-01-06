from pandas import read_csv, DataFrame, Series
from matplotlib.pyplot import savefig, show, subplots
from ds_charts import multiple_line_chart, HEIGHT
from ts_functions import split_temporal_data, PREDICTION_MEASURES, plot_evaluation_results
from sklearn.neighbors import KNeighborsRegressor
from matplotlib.pyplot import figure, xticks
from ts_functions import plot_series, HEIGHT
from ts_functions import plot_evaluation_results
from ds_charts import plot_overfitting_study


file_tag = 'drought_temporal'
#filename = 'data/drought_feat_selected.csv'
index_multi = 'date'

#data: DataFrame = read_csv(filename, index_col='date')
data = read_csv('data/drought_feat_selected.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

target = 'class'

trnX, tstX, trnY, tstY = split_temporal_data(data, target, trn_pct=0.7)

weights = ['uniform', 'distance']
dist = ['manhattan', 'euclidean', 'chebyshev']
kvalues = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]

measure = 'R2'
flag_pct = False
best = ('',  0, 0.0)
last_best = -100000
best_model = None
ncols = len(weights)

fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)
for wei in range(len(weights)):
    w = weights[wei]
    values = {}
    for d in dist:
        yvalues = []
        for k in kvalues:
            pred = KNeighborsRegressor(n_neighbors=k, metric=d, weights=w)
            pred.fit(trnX, trnY)
            prdY = pred.predict(tstX)
            yvalues.append(PREDICTION_MEASURES[measure](tstY,prdY))
            if yvalues[-1] > last_best:
                best = (w, d, k)
                last_best = yvalues[-1]
                best_model = pred

        values[d] = yvalues
    multiple_line_chart(kvalues, values, ax=axs[0, wei], title=f'KNN with {w} weight', xlabel='K', ylabel=measure, percentage=flag_pct)
savefig(f'images/original_{file_tag}_ts_knn_study.png')
print(f'Best results achieved with {best[0]} weight, dist={best[1]} and K={best[2]} ==> measure={last_best:.2f}')




transf = ['agg_m', 'agg_y', 'agg_q', 'diff', 'smooth_10', 'smooth_100']

for t in transf:

    data = read_csv(f'data/transform/{t}.csv', index_col=index_multi, parse_dates=True, infer_datetime_format=True)

    trnX, tstX, trnY, tstY = split_temporal_data(data, target, trn_pct=0.7)

    measure = 'R2'
    flag_pct = False
    best = ('',  0, 0.0)
    last_best = -100000
    best_model = None
    ncols = len(weights)

    fig, axs = subplots(1, ncols, figsize=(ncols*HEIGHT, HEIGHT), squeeze=False)
    for wei in range(len(weights)):
        w = weights[wei]
        values = {}
        for d in dist:
            yvalues = []
            for k in kvalues:
                pred = KNeighborsRegressor(n_neighbors=k, metric=d, weights=w)
                pred.fit(trnX, trnY)
                prdY = pred.predict(tstX)
                yvalues.append(PREDICTION_MEASURES[measure](tstY,prdY))
                if yvalues[-1] > last_best:
                    best = (w, d, k)
                    last_best = yvalues[-1]
                    best_model = pred

            values[d] = yvalues
        multiple_line_chart(kvalues, values, ax=axs[0, wei], title=f'KNN with {w} weight', xlabel='K', ylabel=measure, percentage=flag_pct)
    savefig(f'images/{t}_ts_knn_study.png')
    print(f'Best results achieved with {best[0]} weight, dist={best[1]} and K={best[2]} ==> measure={last_best:.2f}')


    prd_trn = best_model.predict(trnX)
    prd_tst = best_model.predict(tstX)

    plot_evaluation_results(trnY, prd_trn, tstY, prd_tst, f'images/{t}_ts_knn_best.png')
    savefig(f'images/{t}_ts_knn_best.png')

    y_tst_values = []
    y_trn_values = []
    for k in kvalues:
        pred = KNeighborsRegressor(n_neighbors=k, metric=best[1], weights=best[0])
        pred.fit(trnX, trnY)
        prd_tst_Y = pred.predict(tstX)
        prd_trn_Y = pred.predict(trnX)
        y_tst_values.append(PREDICTION_MEASURES[measure](tstY, prd_tst_Y))
        y_trn_values.append(PREDICTION_MEASURES[measure](trnY, prd_trn_Y))
    plot_overfitting_study(kvalues, y_trn_values, y_tst_values, name=f'{t}ts_knn_{best[0]}_{best[1]}', xlabel='K', ylabel=measure, pct=flag_pct)