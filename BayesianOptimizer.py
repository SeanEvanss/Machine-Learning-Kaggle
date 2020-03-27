import lightgbm as lgb
import numpy as np
from FeatureEncoder import*
import warnings
import seaborn as sns
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()


class BayesianOptimizer():

    def __init__(self, df, train, test, targetCol, targetlb):
        self.train_df=df
        self.train=train
        self.test=test
        self.targetCol=targetCol
        self.target=targetlb
    def LGB_bayesian(self, learning_rate,
    num_leaves,
    bagging_fraction,
    feature_fraction,
    min_child_weight,
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda):

        # LightGBM expects next three parameters need to be integer.
        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)

        assert type(num_leaves) == int
        assert type(min_data_in_leaf) == int
        assert type(max_depth) == int

        BayesianParams = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'min_child_weight': min_child_weight,
            'bagging_fraction': bagging_fraction,
            'feature_fraction': feature_fraction,
            'learning_rate' : learning_rate,
            'max_depth': max_depth,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'objective': 'binary',
            'save_binary': True,
            'seed': 1337,
            'feature_fraction_seed': 1337,
            'bagging_seed': 1337,
            'drop_seed': 1337,
            'data_random_seed': 1337,
            'boosting_type': 'gbdt',
            'verbose': 1,
            'is_unbalance': False,
            'boost_from_average': True,
            'metric': 'auc'}



        folds = TimeSeriesSplit(n_splits=5)

        for fold, (bayesian_tr_idx, bayesian_val_idx) in enumerate(folds.split(self.train, self.targetCol)):
            print('Training on fold {}'.format(fold + 1))

            trn_data = lgb.Dataset(self.train.iloc[bayesian_tr_idx], label=self.targetCol.iloc[bayesian_tr_idx])
            val_data = lgb.Dataset(self.train.iloc[bayesian_val_idx], label=self.targetCol.iloc[bayesian_val_idx])
            clf = lgb.train(BayesianParams, trn_data, 10000, valid_sets=[trn_data, val_data], verbose_eval=1000,
                            early_stopping_rounds=10, categorical_feature=['TransactionHour'])
            oof = np.zeros(len(self.train))
            features=list(self.train)
            oof[bayesian_val_idx] = clf.predict(self.train_df.iloc[bayesian_val_idx][features].values,
                                                num_iteration=clf.best_iteration)
            score = roc_auc_score(self.train_df.iloc[bayesian_val_idx][self.target].values, oof[bayesian_val_idx])

            return score


    # Bounded region of parameter space

    def optimize(self, bounds_LGB):
        LGB_BO = BayesianOptimization(self.LGB_bayesian, bounds_LGB, random_state=42)
        print(LGB_BO.space.keys)
        init_points = 10
        n_iter = 15
        print('-' * 130)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

        # max target
        print(LGB_BO.max['target'])
        #best params
        print(LGB_BO.max['params'])


'''
#black box LGBM 
def LGB_bayesian(
    #learning_rate,
    num_leaves, 
    bagging_fraction,
    feature_fraction,
    min_child_weight, 
    min_data_in_leaf,
    max_depth,
    reg_alpha,
    reg_lambda
     ):

    oof = np.zeros(len(train))
    trn_data= lgb.Dataset(train_df.iloc[bayesian_tr_idx][features].values, label=train_df.iloc[bayesian_tr_idx][target].values)
    val_data= lgb.Dataset(train_df.iloc[bayesian_val_idx][features].values, label=train_df.iloc[bayesian_val_idx][target].values)

    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=0, early_stopping_rounds = 50)

    oof[bayesian_val_idx]  = clf.predict(train_df.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  



'''

'''
# Bounded region of parameter space
bounds_LGB = {
    'num_leaves': (31, 500), 
    'min_data_in_leaf': (20, 200),
    'bagging_fraction' : (0.1, 0.9),
    'feature_fraction' : (0.1, 0.9),
    #'learning_rate': (0.01, 0.3),
    'min_child_weight': (0.00001, 0.01),   
    'reg_alpha': (1, 2), 
    'reg_lambda': (1, 2),
    'max_depth':(-1,50),
}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
print(LGB_BO.space.keys)
init_points = 10
n_iter = 15
print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)

# max target
print(LGB_BO.max['target'])
#best params
print(LGB_BO.max['params'])'''