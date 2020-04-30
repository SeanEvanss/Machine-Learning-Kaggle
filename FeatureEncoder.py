import pandas as pd



class FeatureEncoder:

    def concat_encode(self, leftcol, rightcols, df):
        for col in rightcols:
            df[leftcol+'_'+col] = df[leftcol].astype(str) +df[col].astype(str)
            print("feature encoded: "+leftcol+'_'+col + "\n")

    def freq_encode(self, cols, train,test):
        for col in cols:
            train[col + '_counts'] = train[col].map(
                pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
            test[col + '_counts'] = test[col] .map(
                pd.concat([train[col], test[col]], ignore_index=True).value_counts(dropna=False))
            print("feature encoded: " + col + "_counts" + "\n")



    def aggr_encode(self, cols, target, train,test, aggregations=["mean"]):

        for col in cols:
            for aggrType in aggregations:
                temp = pd.concat([train[[target,col]], test[[target,col]]])
                temp = temp.groupby([target])[col].agg([aggrType]).reset_index().rename(
                    columns={aggrType: col + "_" + target + "_" + aggrType},)

                temp.index = list(temp[target])
                temp = temp[col + "_" + target + "_" + aggrType].to_dict()

                train[col + "_" + target + "_" + aggrType] = train[target].map(temp).astype('float32')
                test[col + "_" + target + "_" + aggrType] = test[target].map(temp).astype('float32')
                print("feature encoded: " + col + "_" + target + "_" + aggrType + "\n")



    def aggr_encode_nunique(self, cols, ids, train_df, test_df):
        for col in cols:
            for id in ids:
                comb = pd.concat([train_df[[id]+[col]],test_df[[id]+[col]]],axis=0)
                mp = comb.groupby(id)[col].agg(['nunique'])['nunique'].to_dict()
                train_df[id+'_'+col+'_ct'] = train_df[id].map(mp).astype('float32')
                test_df[id+'_'+col+'_ct'] = test_df[id].map(mp).astype('float32')
                print('feature encoded: '+ id+'_'+col+'_ct',end='\n')

    def label_encode(self, cols, df):

        for col in cols:
            df[col], _ = df[col].factorize(sort=True)
            if df[col].max() > 32000:
                df[col] = df[col].astype('int32')
            else:
                df[col] = df[col].astype('int16')
