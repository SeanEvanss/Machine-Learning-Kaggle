import pandas as pd
import self as self


class FeatureEncoder:
    def freq_encode(self, cols, df):
        for col in cols:
            temp = df[col].value_counts().to_dict()
            df[col + '_counts'] = df[col].map(temp)
            print("feature encoded: " + col + "_counts" + "\n")

    def aggr_encode(self, cols, target, df, aggregations=["mean"]):
        for col in cols:
            for aggrType in aggregations:
                temp = df.groupby(target)[col].agg([aggrType]).rename({aggrType: col + "_" + target + "_" + aggrType},
                                                                       axis=1)
                df = pd.merge(df, temp, on=target, how='left')
                print("feature encoded: " + col + "_" + target + "_" + aggrType + "\n")

    def aggr_encode2(self, cols, ids, train_df, test_df):
        for col in cols:
            for id in ids:
                comb = pd.concat([train_df[[id]+[col]],test_df[[id]+[col]]],axis=0)
                mp = comb.groupby(id)[col].agg(['nunique'])['nunique'].to_dict()
                train_df[id+'_'+col+'_ct'] = train_df[id].map(mp).astype('float32')
                test_df[id+'_'+col+'_ct'] = test_df[id].map(mp).astype('float32')
                print(id+'_'+col+'_ct, ',end='')
    def label_encode(self, cols, train_df):
        for col in cols:
            train_df[col], _ = train_df[col].factorize(sort=True)
            if train_df[col].max() > 32000:
                train_df[col] = train_df[col].astype('int32')
            else:
                train_df[col] = train_df[col].astype('int16')
