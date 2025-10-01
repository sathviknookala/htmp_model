import pandas as pd

train_df = pd.read_csv('/home/sathvik/hull-tactical-market-prediction/train.csv', delimiter=',')

print(train_df['E10'].value_counts())
print(train_df['E20'].value_counts())

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(train_df.isnull().sum())