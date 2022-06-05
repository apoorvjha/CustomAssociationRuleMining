from pandas import read_csv,read_excel,DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import average_precision_score,make_scorer
from numpy import linspace,abs,mean,where,array
from warnings import filterwarnings

filterwarnings("ignore")

def read_data(path):
    if path.endswith('.csv'):
        return read_csv(path)
    elif path.endswith('.xslx'):
        return read_excel(path)
    else:
        return Exception()

def prepareLists(data,exclusion_set=['Customer_ID','Product_Holding_B1','Product_Holding_B2','Customer_Category']):
    target_cols=[col for col in data.columns if 'Product_Holding_B2_' in col]
    features_under_consideration=list(set(data.columns) - set(exclusion_set) - set(target_cols))
    selected_features={i : [] for i in target_cols}
    return features_under_consideration, target_cols, selected_features

def getThreshold(feature_importances):
    # Create a non Naive threshold strategy.
    return mean(feature_importances)

def featureImportanceBasedSelection(data_path='./workspace/preprocessed_train.csv'):
    try:
        data=read_data(data_path)
    except:
        print("Only csv and excel data is supported.")
        exit(-1)
    features_under_consideration, target_cols, selected_features = prepareLists(data)
    scaler=StandardScaler()
    scorer=make_scorer(average_precision_score)
    X=scaler.fit_transform(data[features_under_consideration])
    for i in range(len(target_cols)):
        Y=data[target_cols[i]]
        if (Y.nunique() < 2):
            print(f"{target_cols[i]} : {Y.nunique()}")
            continue
        HPT=GridSearchCV(RandomForestClassifier(),{"n_estimators" : range(1,500,10)},n_jobs=-1,scoring=scorer,cv=5,verbose=1)
        HPT.fit(X,Y)
        print(f"Best parameter found for {target_cols[i]} is {HPT.best_params_}")
        feature_importances=abs(array(HPT.best_estimator_.feature_importances_))
        threshold=getThreshold(feature_importances)
        #print(f"{target_cols[i]} Feature Importance stats => MIN : {min(feature_importances)}, MAX : {max(feature_importances)}, MEAN : {mean(feature_importances)}")
        if max(feature_importances) > threshold:
            features=array(features_under_consideration)[feature_importances > threshold]
        else:
            features=array(features_under_consideration)
        selected_features[target_cols[i]].extend(list(features))
    return selected_features

def test():
    selected=featureImportanceBasedSelection()
    max_length=max([len(selected[i]) for i in selected.keys()])
    for i in selected.keys():
        if len(selected[i]) < max_length:
            selected[i].extend(list(['' for j in range(max_length - len(selected[i]))]))
    DataFrame(selected).to_csv('./workspace/selected_features.csv',index=False)

if __name__=='__main__':
    test()
