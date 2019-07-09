import pandas as pd
import numpy as np
from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from tqdm import tqdm

def change_row(row):
    row_wo_nan = row.drop('nan')
    if row['nan']:
        return pd.Series(np.array([np.nan for i in range(len(row_wo_nan))]), index=row_wo_nan.index)
    else:
        return row_wo_nan


wals = pd.read_csv('data/wals_language_withstatus.csv')

# Filter columns
wals = wals.drop(['iso_code', 'glottocode', 'Name', 'latitude', 'longitude', 'genus', 'family', 'macroarea', 'countrycodes'], axis=1)

# Filter features < 10
count_features = wals.shape[0] - wals.isnull().sum()
count_features = count_features[count_features > 10]

wals = wals[count_features.index]

# Filter registers < 10
count_langs = wals.shape[1] - wals.isnull().sum(axis=1)
count_langs = count_langs[count_langs > 10]

wals = wals.iloc[count_langs.index].reset_index(drop=True)

wals_target = wals['Status from Glotto']
wals = wals.drop('Status from Glotto', axis=1)

cols_null = {}
for col in tqdm(wals.drop('wals_code', axis=1).columns):
    cols_null[col] = len(wals[col].value_counts())
    if cols_null[col] > 2:
        wals[col] = wals[col].astype(str)
        wals[col] = wals[col].fillna('nan')
        wals_dummies = pd.get_dummies(wals[col])
        wals_dummies = wals_dummies.apply(lambda row: change_row(row), axis=1)
        wals_dummies.columns = col + wals_dummies.columns

        wals = wals.drop(col, axis=1)
        wals = pd.concat((wals, wals_dummies), axis=1)
    else:
        wals[col] = wals[col].replace({c:ix for ix, c in enumerate(wals[col].value_counts().index)})
        wals[col] = wals[col].replace('nan', np.nan)
        wals.fillna(-1)
        
wals.fillna(-1, inplace=True)
wals = wals.loc[wals.wals_code != -1]

wals_melt = pd.melt(wals, id_vars=['wals_code'], var_name='col_name', value_name='feature')
wals_ratings = wals_melt[wals_melt.feature != -1]
wals_ratings_null = wals_melt[wals_melt.feature == -1]

reader = Reader(rating_scale=(0, 1))

trainset = Dataset.load_from_df(wals_ratings, reader).build_full_trainset()

algo = SVD()
algo.fit(trainset)

predictions = []
for ix, row in wals_ratings_null.iterrows():
    predictions.append(algo.predict(row.wals_code, row.col_name).est)
    
wals_ratings_null['feature'] = predictions

temp = pd.pivot_table(pd.concat((wals_ratings_null, wals_ratings), axis=0), values='feature', index=['wals_code'],
                         columns=['col_name'], aggfunc=np.sum)
temp = temp.reset_index()

wals_final = pd.DataFrame()
wals_final['wals_code'] = temp['wals_code']
for col in list(cols_null.keys()):
    group_col = [temp_col for temp_col in temp.columns if temp_col.startswith(col)]
    wals_final[col] = temp[group_col].idxmax(axis=1).str.replace(col, '')
    
wals_final = pd.concat((wals_final, wals_target), axis=1)
wals_final.to_csv(output_file, index=False)