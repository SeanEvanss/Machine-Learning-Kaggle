import lightgbm as lgb
import numpy as np
import pandas as pd
from FeatureEncoder import *
from pandas.api.types import CategoricalDtype
import random
from datetime import datetime

df_tr= pd.read_csv("csv/train_transaction.csv")
df_id=pd.read_csv("csv/train_identity.csv")
test_tr= pd.read_csv("csv/test_transaction.csv")
test_id=pd.read_csv("csv/test_identity.csv")
df = pd.merge(df_tr, df_id, on='TransactionID', how='left')
test = pd.merge(test_tr, test_id, on='TransactionID', how='left')
del df_id, df_tr,test_id, test_tr



"""---------------------------------------------preprocessing-----------------------------------------"""
#drop mostly empty columns
for col in ['TransactionID','card4','dist2','D6','D7','D8','D9','D12','D13','D14','V138','V139','V140','V141','V142','V146',
'V147','V148','V149','V153','V154','V155','V156','V157','V158','V161','V162','V163','V143','V144','V145','V150','V151','V152','V159',
'V160','V164','V165','V166','V322','V323','V324','V325','V326','V327','V328','V329','V330','V331','V332','V333','V334','V335','V336','V337',
'V338','V339','R_emaildomain']:
    df= df.drop([col], axis=1)
    test = test.drop([col], axis=1)


##fill NaN with -999

for col in test.columns.values.tolist():
    df[col].fillna(-999, inplace=True)
    test[col].fillna(-999, inplace=True)


"""--------------------------------------feature engineering------------------------------------------------"""
#normalization:
#normalize transactionAmt against itself
df['TransactionAmt'] = (df['TransactionAmt']-df['TransactionAmt'].mean() ) / df['TransactionAmt'].std()
test['TransactionAmt'] = ( test['TransactionAmt']-test['TransactionAmt'].mean() ) / test['TransactionAmt'].std()
###IssueDate
df['IssueDate']=np.floor(df.TransactionDT/86400-df.D1)
test['IssueDate']=np.floor(test.TransactionDT/86400-test.D1)
###TransactitonHour
df['TransactionHour']=np.floor((np.mod((df.TransactionDT-86400)/3600, 24))).astype('category')
test['TransactionHour']=np.floor((np.mod((test.TransactionDT-86400)/3600, 24))).astype('category')
##Transaction Day of week
df['Transaction_dow'] = np.floor((df['TransactionDT'] / (3600 * 24) - 1) % 7)
test['Transaction_dow'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)
#log transaction amount.
df['TransactionAmt_log'] = np.log(df['TransactionAmt'])
test['TransactionAmt_log'] = np.log(test['TransactionAmt'])
# New feature - decimal part of the transaction amount.
df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)
####ClientID consisting of card1+add1+Issue Date(TransactionDT/86400-D1)
df['ClientID']=df['card1'].astype(str)+df['addr1'].astype(str)+df['IssueDate'].astype(str)
test['ClientID']=test['card1'].astype(str)+test['addr1'].astype(str)+test['IssueDate'].astype(str)

fe = FeatureEncoder()
#concat encode
#card1+card2,carrd5,dist1
fe.concat_encode('card1', ['card2','card5','dist1'],df)
fe.concat_encode('card1', ['card2','card5','dist1'],test)
#card2+dist1
fe.concat_encode('card2',['dist1'],df)
fe.concat_encode('card2',['dist1'],test)
#emaildomain+C1,C2,card1,card2,card5
fe.concat_encode('P_emaildomain', ['C1','C2','card1','card2','card5','card1_dist1'],df)
fe.concat_encode('P_emaildomain', ['C1','C2','card1','card2','card5','card1_dist1'],test)
#devicetype+deviceinfo
fe.concat_encode('DeviceType',['DeviceInfo'],df)
fe.concat_encode('DeviceType',['DeviceInfo'],test)

#split some of the columns with complex info
df['OS'] = df['id_30'].str.split(' ', expand=True)[0]
df['OSversion'] = df['id_30'].str.split(' ', expand=True)[1]
test['OS'] = test['id_30'].str.split(' ', expand=True)[0]
test['OSversion'] = test['id_30'].str.split(' ', expand=True)[1]

df['browser'] = df['id_31'].str.split(' ', expand=True)[0]
df['browser_version'] = df['id_31'].str.split(' ', expand=True)[1]
test['browser'] = test['id_31'].str.split(' ', expand=True)[0]
test['browser_version'] = test['id_31'].str.split(' ', expand=True)[1]

df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
test['id_34'] = test['id_34'].str.split(':', expand=True)[1]

df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]
test['device_name'] = test['DeviceInfo'].str.split('/', expand=True)[0]
test['device_version'] = test['DeviceInfo'].str.split('/', expand=True)[1]

#frequency encode
fe.freq_encode(['card1', 'card2', 'addr1', 'P_emaildomain','D1', 'OSversion', 'browser_version','Transaction_dow', 'TransactionHour'], df, test)
#aggregate encode(std)
fe.aggr_encode(['C1','C7','D10','D11'],
               'card1_dist1', df, test ,['std'])
fe.aggr_encode(['TransactionAmt','TransactionAmt_decimal','TransactionDT','C2','C13','D1'],
               'P_emaildomain_card2', df, test,['std'])
#aggregate encode(mean)
fe.aggr_encode(['TransactionAmt','TransactionDT','TransactionAmt_decimal','D15','D4','C13'], 'ClientID', df, test,['mean'])
#aggregate encode(nunique)
fe.aggr_encode_nunique(['P_emaildomain', 'C1','C2','C6','C13','C14',
                        'TransactionHour', 'D1', 'D2','D3','D4','D5',
                        'D10','D15','V310','V307','V303','Transaction_dow','TransactionHour','id_19','id_20','id_02','browser'],
                       ['ClientID','card1_card5', 'card1_dist1','card1_card2'], df, test)
##label encode
fe.label_encode(['card6','P_emaildomain','ClientID', 'P_emaildomain_C2','P_emaildomain_C1', 'P_emaildomain_card1', 'P_emaildomain_card2'
                    ,'P_emaildomain_card5','OS','OSversion','browser','browser_version','device_name','device_version',
                 'card1_card2','card1_card5', 'card1_dist1','card2_dist1','DeviceType_DeviceInfo'
                , 'P_emaildomain_card1_dist1','M1',
                 'M2','M3','M4','M5','M6','M7','M8','M9','id_02',
                 'id_03','id_05','id_06','id_09','id_11','id_12','id_15','id_16','id_28','id_29'
                    ,'id_30','id_31','id_32','id_33','id_34','id_35','id_36','id_37','id_38','DeviceType','DeviceInfo','ProductCD'],df)


#drop features with low importance in the previous dfing
for index, row in featimpt.iterrows():
    if row['average']<=200:
        df = df.drop([row['feature']], axis=1)

##reduce memory
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col] = df[col].astype('float32')
    if df[col].dtype == 'int64':
        df[col] = df[col].astype('int32')

df=df.drop(['isFraud'], axis=1)
"""----------------------------------prepare mock dataset----------------------------------"""
#index 240 fraud, index 241 non fraud
mockclient_1=df.loc[240,].copy()
mockclient_2=df.loc[240,].copy()
mockclient_3=df.loc[241,].copy()
mockclient_4=df.loc[241,].copy()


##Scenario 1: Fraudulent client makes another transaction using the same card

card1 = mockclient_1['card1']
Tdt = mockclient_1['TransactionDT']
D_1 = mockclient_1['D1']
addr1 = mockclient_1['addr1']
index = 0
while index < mockclient_1.size:
    mockclient_1[index] = mockclient_2[index]
    index+=1

mockclient_1['card1'] = card1
mockclient_1['TransactionDT'] = Tdt
mockclient_1['D1'] = D_1
mockclient_1['addr1'] = addr1

##Scenario 2: Fraudulent client change a card and makes another transaction
mockclient_2['card1']=123456
mockclient_2['TransactionDT']+=10000
mockclient_2['addr1']=123

##Scenario 3: Non-fraudulent client makes another transaction using the same card
card1=mockclient_3['card1']
Tdt=mockclient_3['TransactionDT']
D_1=mockclient_3['D1']
addr1=mockclient_3['addr1']
index=0

while index < mockclient_3.size:
    mockclient_3[index]=mockclient_4[index]
    index+=1

mockclient_3['card1']=card1
mockclient_3['TransactionDT']=Tdt
mockclient_3['D1']=D_1
mockclient_3['addr1']=addr1

##Scenario 4: Non-fraudulent client change a card and makes another transaction

mockclient_4['card1']=67890
mockclient_4['TransactionDT']+=10000
mockclient_4['addr1']=456


##prepare dataset for prediction
mocklist=[mockclient_1,mockclient_2,mockclient_3,mockclient_4]
mockdf=pd.DataFrame(mocklist)
mockdf['TransactionHour']=mockdf['TransactionHour'].astype('category')

#load model
model = lgb.Booster(model_file='lgb-model.txt')
pred= model.predict(mockdf)

print('client1: ', pred[0])
print('client2: ', pred[1])
print('client3: ', pred[2])
print('client4: ', pred[3])




