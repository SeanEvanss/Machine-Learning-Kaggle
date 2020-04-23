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


train= pd.read_csv("csv/new_train_transaction.csv")
test= pd.read_csv("csv/new_test_transaction.csv")
sub=pd.read_csv("csv/sample_submission.csv")




for col in ['TransactionID','card3','card4','card6','addr2','dist2','C3','C4','V339','ProductCD','R_emaildomain']:
    train= train.drop([col], axis=1)
    test = test.drop([col], axis=1)

print(train.shape,test.shape)
#new features
###IssueDate
train['IssueDate']=np.floor(train.TransactionDT/86400-train.D1)
test['IssueDate']=np.floor(test.TransactionDT/86400-test.D1)
###TransactitonHour
train['TransactionHour']=np.floor((np.mod((train.TransactionDT-86400)/3600, 24))).astype('category')
test['TransactionHour']=np.floor((np.mod((train.TransactionDT-86400)/3600, 24))).astype('category')
####ClientID consisting of card1+add1+Issue Date(TransactionDT/86400-D1)
train['ClientID']=train['card1'].astype(str)+train['addr1'].astype(str)+train['IssueDate'].astype(str)
test['ClientID']=test['card1'].astype(str)+test['addr1'].astype(str)+test['IssueDate'].astype(str)
#email_domain+C2
train['P_emaildomain_C2']=train['P_emaildomain'].astype(str)+train['C2'].astype(str)
test['P_emaildomain_C2']=test['P_emaildomain'].astype(str)+test['C2'].astype(str)
#email_domain+C1
train['P_emaildomain_C1']=train['P_emaildomain'].astype(str)+train['C1'].astype(str)
test['P_emaildomain_C1']=test['P_emaildomain'].astype(str)+test['C1'].astype(str)
#email_domain+card1
train['P_emaildomain_card1']=train['P_emaildomain'].astype(str)+train['card1'].astype(str)
test['P_emaildomain_card1']=test['P_emaildomain'].astype(str)+test['card1'].astype(str)
#email_domain+card2
train['P_emaildomain_card2']=train['P_emaildomain'].astype(str)+train['card2'].astype(str)
test['P_emaildomain_card2']=test['P_emaildomain'].astype(str)+test['card2'].astype(str)
#email_domain+card5
train['P_emaildomain_card5']=train['P_emaildomain'].astype(str)+train['card5'].astype(str)
test['P_emaildomain_card5']=test['P_emaildomain'].astype(str)+test['card5'].astype(str)
#card1+card2
train['card1_card2']=train['card1'].astype(str)+train['card2'].astype(str)
test['card1_card2']=test['card1'].astype(str)+test['card2'].astype(str)
#card1+card5
train['card1_card5']=train['card1'].astype(str)+train['card5'].astype(str)
test['card1_card5']=test['card1'].astype(str)+test['card5'].astype(str)
#card1+dist1
train['card1_dist1']=train['card1'].astype(str)+train['dist1'].astype(str)
test['card1_dist1']=test['card1'].astype(str)+test['dist1'].astype(str)
#card2+dist1
train['card2_dist1']=train['card2'].astype(str)+train['dist1'].astype(str)
test['card2_dist1']=test['card2'].astype(str)+test['dist1'].astype(str)
#feature engineering

##fill NaN with -999
#train['P_emaildomain'].fillna(-999, inplace=True)
#train['card3'].fillna(-999, inplace=True)
#test['P_emaildomain'].fillna(-999, inplace=True)
#test['card3'].fillna(-999, inplace=True)
##encode
fe= FeatureEncoder()
fe.freq_encode(['ClientID', 'card1', 'card2', 'addr1', 'P_emaildomain', 'D1', 'TransactionHour'], train)
fe.aggr_encode(['TransactionAmt','TransactionDT'], 'ClientID', train)
fe.freq_encode(['ClientID','card1', 'card2','addr1', 'P_emaildomain', 'D1', 'TransactionHour'], test)
fe.aggr_encode(['TransactionAmt', 'TransactionDT'], 'ClientID', test)
fe.aggr_encode2(['P_emaildomain', 'C1','C2', 'TransactionHour', 'D1', 'TransactionAmt'], ['ClientID',
                                                                                        'card1_card5', 'card2_dist1'], train, test)

#fe.aggr_encode(['ProductCD', 'P_emaildomain', 'C1', 'TransactionHour', 'card2', 'card3', 'D1'], 'ClientID', test, ['nunique'])

##categorization
#train['P_emaildomain']=train['P_emaildomain'].astype('category')
#test['P_emaildomain']=test['P_emaildomain'].astype('category')

#concat ClientID of train and test

##label encode

fe.label_encode(['ClientID', 'P_emaildomain_C2','P_emaildomain_C1', 'P_emaildomain_card1', 'P_emaildomain_card2'
                    ,'P_emaildomain_card5','card1_card2','card1_card5', 'card1_dist1','card2_dist1'],train)
fe.label_encode(['ClientID', 'P_emaildomain_C2', 'P_emaildomain_C1','P_emaildomain_card1', 'P_emaildomain_card2'
                    ,'P_emaildomain_card5','card1_card2','card1_card5', 'card1_dist1','card2_dist1'],test)

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


#prepare trainset&target
train_df=train
Y= train_df['isFraud']
train = train.drop(['isFraud'], axis=1)
train = train.drop(['P_emaildomain'],axis=1)
X=train
test=test.drop(['P_emaildomain'],axis=1)
print(X.shape)
print(test.shape)
print("features:  ")
Features=[]
for col in X.columns:
    Features.append(col)
    print(col+",")
'''#bounding params to test
bounds={
            'num_leaves': (31, 500),
            'min_data_in_leaf': (20, 200),
            'bagging_fraction': (0.1, 0.9),
            'feature_fraction': (0.1, 0.9),
            'learning_rate': (0.01, 0.3),
            'min_child_weight': (0.00001, 0.01),
            'reg_alpha': (0, 2),
            'reg_lambda': (0, 2),
            'max_depth': (-1, 50),
        }
BO=BayesianOptimizer(train_df, train, test, Y, 'isFraud')
BO.optimize(bounds)


'''
#prepare model params
params = {'num_leaves': 500,
          'min_child_weight': 0.008023947837732857,
          'feature_fraction': 0.32474760774990463,
          'bagging_fraction': 0.38540266135487145,
          'min_data_in_leaf': 150,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate' : 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "max_bin": 255,
          "verbosity": -1,
          'reg_alpha' : 0.3899927210061127,
          'reg_lambda' : 0.6485237330340494,
          'random_state': 47,

         }


folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, Y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))

    trn_data = lgb.Dataset(X.iloc[trn_idx], label=Y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=Y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                    early_stopping_rounds=100, categorical_feature=['TransactionHour'])

    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])

    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
meanAuc=np.mean(aucs)
print('Mean AUC:', meanAuc)

f = open("trainedResults.txt",'a+',encoding = 'utf-8')
f.write('Mean AUC: %f' %meanAuc+"\n")
f.write('paramsused: '+"\n")
for p, v in params.items():
    f.write(str(p) + " : "+ str(v)+"\n")

f.write("-" *30 + "\n")
f.close()

print('-' * 30)

feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('C:\\Users\\User\\Desktop\\MLout\\feature_importances.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('feature importance over {} folds average'.format(folds.n_splits))
plt.show()

best_iter = clf.best_iteration
clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, Y)
sub['isFraud'] = clf.predict_proba(test)[:, 1]
sub.to_csv('ieee_cis_fraud_detection_v2.csv', index=False)