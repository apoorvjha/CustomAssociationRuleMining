from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from joblib import dump, load
from numpy import array, sort, where
from warnings import filterwarnings
from pandas import read_csv, read_excel, DataFrame

filterwarnings("ignore")

def getRecs(x,products):
    filter = x==1
    return " , ".join(list(array(products)[filter]))

def custom_split(train_df,features,target,test_size=0.1,normalize=True, random_state=42):
    uniques=train_df[target].unique()
    scaler=StandardScaler()
    X_train=[]
    Y_train=[]
    X_val=[]
    Y_val=[]
    for i in uniques:
        train_space=train_df[train_df[target]==i].sample(frac=1-test_size,replace=False,random_state=random_state)
        val_space=train_df[train_df[target]==i].sample(frac=1-test_size,replace=False,random_state=random_state)
        X_train.extend(train_space[features].values)
        Y_train.extend(train_space[target].values)
        X_val.extend(val_space[features].values)
        Y_val.extend(val_space[target].values)
    X_train=array(X_train)
    X_val=array(X_val)
    if normalize==True:
        X_train=scaler.fit_transform(array(X_train))
        X_val=scaler.transform(array(X_val))
    return X_train, X_val, array(Y_train), array(Y_val)

def read_data(path):
    if path.endswith('.csv'):
        return read_csv(path)
    elif path.endswith('.xslx'):
        return read_excel(path)
    else:
        return Exception()

def criterion(actual,predicted):
    return average_precision_score(actual,predicted)

def train(selected_features_path='./workspace/selected_features.csv',preprocessed_train_path='./workspace/preprocessed_train.csv'):
    features_df=read_data(selected_features_path)
    models={}
    for col in features_df.columns:
        target=col
        features=features_df[col].dropna().values
        train_df=read_data(preprocessed_train_path)
        X_train, X_val, Y_train, Y_val = custom_split(train_df,features=features,target=target,test_size=0.1,normalize=True)
        X=train_df[features]
        Y=train_df[target]
        model=LogisticRegression(n_jobs=-1)
        model.fit(X_train,Y_train)
        predictions=model.predict(X_val)
        print(f"{target} Average Precision : {criterion(Y_val, predictions)}")
        dump(model, f"./models/{col}_model.pkl")
        models[col]=f"./models/{col}_model.pkl"
    return models

def generate_predictions(models,selected_features_path='./workspace/selected_features.csv',preprocessed_test_path='./workspace/preprocessed_test.csv'):
    features_df=read_data(selected_features_path)
    result=[]
    result_df={}
    test_df=read_data(preprocessed_test_path)
    for col in features_df.columns:
        features=features_df[col].dropna().values
        X=test_df[features]
        scaler=StandardScaler()
        X=scaler.fit_transform(X)
        model=load(models[col])
        predictions=model.predict(X)
        result.append(predictions)
    result=array(result).T
    recommended_products=[]
    for i in range(len(result)):
        t=getRecs(result[i],list(features_df.keys()))
        recommended_products.append(t)
    result_df['Customer_ID']=test_df['Customer_ID'].values
    result_df['Recommended_Products']=recommended_products
    DataFrame(result_df).to_csv('./output/prediction.csv',index=False)

def test():
    models_df=train()
    generate_predictions(models_df)
    
if __name__=='__main__':
    test()

        

