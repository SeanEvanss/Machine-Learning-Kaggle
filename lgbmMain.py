import lightgbm as lgb
import numpy as np
import pandas as pd
#import multiprocessing
import warnings
import seaborn as sns
from FeatureEncoder import*
#from tqdm import tqdm_notebook
#from sklearn.preprocessing import LabelEncoder
#from sklearn.metrics import roc_auc_score
#%matplotlib inline
import matplotlib.pyplot as plt
import gc
from time import time
import datetime
from BayesianOptimizer import*
from sklearn.model_selection import KFold, TimeSeriesSplit
warnings.simplefilter('ignore')
sns.set()


"""------------------------------------------read in df----------------------------------------------------"""
train_tr= pd.read_csv("csv/train_transaction.csv")
test_tr= pd.read_csv("csv/test_transaction.csv")
sub=pd.read_csv("csv/sample_submission.csv")
train_id=pd.read_csv("csv/train_identity.csv")
test_id=pd.read_csv("csv/test_identity.csv")
featimpt=pd.read_csv("csv/feature_importances.csv")
train = pd.merge(train_tr, train_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

del test_id, test_tr, train_id, train_tr

x=gc.collect()


"""---------------------------------------------preprocessing-----------------------------------------"""
#drop mostly empty columns
for col in ['TransactionID','card4','dist2','D6','D7','D8','D9','D12','D13','D14','V138','V139','V140','V141','V142','V146',
'V147','V148','V149','V153','V154','V155','V156','V157','V158','V161','V162','V163','V143','V144','V145','V150','V151','V152','V159',
'V160','V164','V165','V166','V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336','V337',
'V338','V339','R_emaildomain']:
    train= train.drop([col], axis=1)
    test = test.drop([col], axis=1)



print(train.shape,test.shape)


##fill NaN with -999

for col in test.columns.values.tolist():
    train[col].fillna(-999, inplace=True)
    test[col].fillna(-999, inplace=True)


#normalization:
#normalize transactionAmt against itself
train['TransactionAmt'] = ( train['TransactionAmt']-train['TransactionAmt'].mean() ) / train['TransactionAmt'].std()
test['TransactionAmt'] = ( test['TransactionAmt']-test['TransactionAmt'].mean() ) / test['TransactionAmt'].std()

"""#normalize Ds
for dcol in ['D1','D2','D3','D4','D5','D15']:
    train[dcol]=np.floor(train.TransactionDT/86400-train[dcol])
    test[dcol]=np.floor(train.TransactionDT/86400-train[dcol])"""


"""-----------------------------------feature engineering--------------------------------------------"""
###IssueDate
train['IssueDate']=np.floor(train.TransactionDT/86400-train.D1)
test['IssueDate']=np.floor(test.TransactionDT/86400-test.D1)
###TransactitonHour
train['TransactionHour']=np.floor((np.mod((train.TransactionDT-86400)/3600, 24))).astype('category')
test['TransactionHour']=np.floor((np.mod((train.TransactionDT-86400)/3600, 24))).astype('category')

##Transaction Day of week
train['Transaction_dow'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_dow'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

#log transaction amount.
train['TransactionAmt_log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_log'] = np.log(test['TransactionAmt'])

# New feature - decimal part of the transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)


####ClientID consisting of card1+add1+Issue Date(TransactionDT/86400-D1)
train['ClientID']=train['card1'].astype(str)+train['addr1'].astype(str)+train['IssueDate'].astype(str)
test['ClientID']=test['card1'].astype(str)+test['addr1'].astype(str)+test['IssueDate'].astype(str)


fe = FeatureEncoder()

#concat encode


#card1+card2,carrd5,dist1
fe.concat_encode('card1', ['card2','card5','dist1'],train)
fe.concat_encode('card1', ['card2','card5','dist1'],test)

#card2+dist1
fe.concat_encode('card2',['dist1'],train)
fe.concat_encode('card2',['dist1'],test)


#emaildomain+C1,C2,card1,card2,card5
fe.concat_encode('P_emaildomain', ['C1','C2','card1','card2','card5','card1_dist1'],train)
fe.concat_encode('P_emaildomain', ['C1','C2','card1','card2','card5','card1_dist1'],test)

#devicetype+deviceinfo
fe.concat_encode('DeviceType',['DeviceInfo'],train)
fe.concat_encode('DeviceType',['DeviceInfo'],test)

#split some of the columns with complex info
train['OS'] = train['id_30'].str.split(' ', expand=True)[0]
train['OSversion'] = train['id_30'].str.split(' ', expand=True)[1]
test['OS'] = test['id_30'].str.split(' ', expand=True)[0]
test['OSversion'] = test['id_30'].str.split(' ', expand=True)[1]

train['browser'] = train['id_31'].str.split(' ', expand=True)[0]
train['browser_version'] = train['id_31'].str.split(' ', expand=True)[1]
test['browser'] = test['id_31'].str.split(' ', expand=True)[0]
test['browser_version'] = test['id_31'].str.split(' ', expand=True)[1]

train['id_34'] = train['id_34'].str.split(':', expand=True)[1]
test['id_34'] = test['id_34'].str.split(':', expand=True)[1]

train['device_name'] = train['DeviceInfo'].str.split('/', expand=True)[0]
train['device_version'] = train['DeviceInfo'].str.split('/', expand=True)[1]
test['device_name'] = test['DeviceInfo'].str.split('/', expand=True)[0]
test['device_version'] = test['DeviceInfo'].str.split('/', expand=True)[1]

#frequency encode
fe.freq_encode(['card1', 'card2', 'addr1', 'P_emaildomain','D1', 'OSversion', 'browser_version','Transaction_dow', 'TransactionHour'], train, test)


#aggregate encode(std)
fe.aggr_encode(['C1','C7','D10','D11'],
               'card1_dist1', train, test ,['std'])


fe.aggr_encode(['TransactionAmt','TransactionAmt_decimal','TransactionDT','C2','C13','D1'],
               'P_emaildomain_card2', train, test,['std'])


#aggregate encode(mean)
fe.aggr_encode(['TransactionAmt','TransactionDT','TransactionAmt_decimal','D15','D4','C13'], 'ClientID', train, test,['mean'])


#aggregate encode(nunique)
fe.aggr_encode_nunique(['P_emaildomain', 'C1','C2','C6','C13','C14',
                        'TransactionHour', 'D1', 'D2','D3','D4','D5',
                        'D10','D15','V310','V307','V303','Transaction_dow','TransactionHour','id_19','id_20','id_02','browser'],
                       ['ClientID','card1_card5', 'card1_dist1','card1_card2'], train, test)

##label encode

fe.label_encode(['card6','P_emaildomain','ClientID', 'P_emaildomain_C2','P_emaildomain_C1', 'P_emaildomain_card1', 'P_emaildomain_card2'
                    ,'P_emaildomain_card5','OS','OSversion','browser','browser_version','device_name','device_version',
                 'card1_card2','card1_card5', 'card1_dist1','card2_dist1','DeviceType_DeviceInfo'
                , 'P_emaildomain_card1_dist1','M1',
                 'M2','M3','M4','M5','M6','M7','M8','M9','id_02',
                 'id_03','id_05','id_06','id_09','id_11','id_12','id_15','id_16','id_28','id_29'
                    ,'id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38','DeviceType','DeviceInfo','ProductCD'],train)
fe.label_encode(['card6','P_emaildomain', 'ClientID', 'P_emaildomain_C2','P_emaildomain_C1', 'P_emaildomain_card1', 'P_emaildomain_card2'
                    ,'P_emaildomain_card5','OS','OSversion','browser','browser_version','device_name','device_version',
                 'card1_card2','card1_card5', 'card1_dist1','card2_dist1'
                 , 'P_emaildomain_card1_dist1','M1','M2',
                 'M3','M4','M5','M6','M7','M8','M9','id_02',
                 'id_03','id_05','id_06','id_09','id_11','id_12','id_15','id_16','id_28','id_29'.
                    ,'id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38','DeviceType','DeviceInfo','DeviceType_DeviceInfo',
                 'ProductCD'],test)


"""----------------------------------post processing train&test----------------------------------"""
#drop features with low importance in the previous training
for index, row in featimpt.iterrows():
    if row['average']<=200:
        train = train.drop([row['feature']], axis=1)
        test = test.drop([row['feature']], axis=1)

x=gc.collect()

##reduce memory
for col in train.columns:
    if train[col].dtype == 'float64':
        train[col] = train[col].astype('float32')
    if train[col].dtype == 'int64':
        train[col] = train[col].astype('int32')
for col in test.columns:
    if test[col].dtype == 'float64':
        test[col] = test[col].astype('float32')
    if test[col].dtype == 'int64':
        test[col] = test[col].astype('int32')


"""-------------------------------------prepare model------------------------------------------------------"""
#prepare trainset&target
train_df=train
train = train.drop(['isFraud'], axis=1)
X=train
Y= train_df['isFraud']
print(X.shape)
print(test.shape)
print("features:  ")
Features=[]
for col in X.columns:
    Features.append(col)
    print(col+",")

'''
#bounding params to test
bounds={
            'num_leaves': (30, 500),
            'min_data_in_leaf': (20, 200),
            'bagging_fraction': (0.1, 0.9),
            'feature_fraction': (0.1, 0.9),
            'learning_rate': (0.01, 0.3),
            'min_child_weight': (0.00001, 0.01),
            'reg_alpha': (0, 2),
            'reg_lambda': (0, 2),
            'max_depth': (-1, 50),
        }
BO=BayesianOptimizer(train_df, X, test, Y, 'isFraud')
BO.optimize(bounds)


'''
#prepare model params
params = {'bagging_fraction': 0.5113875507308893,
         'feature_fraction': 0.573931655089634,
         'learning_rate': 0.006347061968879934,
         'max_depth': 30,
         'min_child_weight': 0.0017135359956360425,
         'min_data_in_leaf': 32,
         'num_leaves': 500,
         'reg_alpha': 1.9312640661491187,
         'reg_lambda': 1.6167946962329223,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "max_bin": 255,
          "verbosity": -1,
          'random_state': 47,
          'objective': 'binary',
         }


#split 5 folds
folds = TimeSeriesSplit(n_splits=5)

aucs = list()

feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
training_start_time = time()
clf=lgb.Booster

"""-------------------------------------train & predict--------------------------------------------------"""
#train in five folds
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, Y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=Y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=Y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 5000, valid_sets=[trn_data, val_data], verbose_eval=500,
                    early_stopping_rounds=100, categorical_feature=['TransactionHour'])

    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
meanAuc=np.mean(aucs)
print('Mean AUC:', meanAuc)

#append mean AUC with params used to trainedResults.txt
f = open("trainedResults.txt",'a+',encoding = 'utf-8')
f.write('Mean AUC: %f' %meanAuc+"\n")
f.write('paramsused: '+"\n")
for p, v in params.items():
    f.write(str(p) + " : "+ str(v)+"\n")
f.write("-" *30 + "\n")
f.close()

print('-' * 30)
#plot top 50 feature importance 
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('csv/feature_importances.csv')

plt.figure(figsize=(32, 32))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('feature importance over {} folds average'.format(folds.n_splits))
plt.show()

#predict and label with best iteration
best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, Y)
sub['isFraud'] = clf.predict_proba(test)[:, 1]
sub.to_csv('lgbmsubmission.csv', index=False)
