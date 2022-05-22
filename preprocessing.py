from pandas import read_csv,read_excel,get_dummies,concat

def read_data(path):
    if path.endswith('.csv'):
        return read_csv(path)
    elif path.endswith('.xslx'):
        return read_csv(path)
    else:
        return Exception()

class Preprocessing:
    def __init__(self,train_data_path='Dataset/train.csv',test_data_path='Dataset/test.csv'):
        try:
            self.train_data=read_data(train_data_path)
            self.test_data=read_data(test_data_path)
        except:
            print("Only csv and excel data is supported.")
            exit(-1)
    def handleMissingData(self):
        print("\n\n**** Performing missing data imputation in Train dataset ****\n\n")
        for col in list(set(self.train_data.columns) - set(['Customer_ID','Product_Holding_B1','Product_Holding_B2'])):
            if self.train_data[col].isnull().sum()!=0:
                if self.train_data[col].dtype=='object':
                    self.train_data[col].fillna("MISSING",inplace=True)
                else:
                    if self.train_data[col].dtype == 'int64': 
                        self.train_data[col].fillna(self.train_data[col].mode()[col][0],inplace=True)
                        print("Handelled int64!")
                    else:
                        self.train_data[col].fillna(self.train_data[col].mean(),inplace=True)
                
            else:
                print(f"'{col}' have no missing data record.")
        print("\n\n**** Performing missing data imputation in Test dataset ****\n\n")
        for col in list(set(self.test_data.columns) - set(['Customer_ID','Product_Holding_B1','Product_Holding_B2'])):
            if self.test_data[col].isnull().sum()!=0:
                if self.test_data[col].dtype=='object':
                    self.test_data[col].fillna("MISSING",inplace=True)
                else:
                    if self.test_data[col].dtype == 'int64': 
                        self.test_data[col].fillna(self.test_data[col].mode()[col][0],inplace=True)
                        print("Handelled int64!")
                    else:
                        self.test_data[col].fillna(self.test_data[col].mean(),inplace=True)
                
            else:
                print(f"'{col}' have no missing data record.")
    def strip_spaces(self,x):
        y=[]
        for i in x:
            y.append(i.strip())
        return y
    def mapProductHolding(self):
        product_holding_cols=['Product_Holding_B1','Product_Holding_B2']
        assert self.train_data[product_holding_cols[0]].isnull().sum()==0, f"{product_holding_cols[0]} of train set contains null."
        assert self.train_data[product_holding_cols[1]].isnull().sum()==0, f"{product_holding_cols[1]} of train set contains null."
        assert self.test_data[product_holding_cols[0]].isnull().sum()==0, f"{product_holding_cols[0]} of test set contains null."
        for col in product_holding_cols:
            print(f"\n\n**** Preprocessing {col} column in training set ****\n\n")
            vals=self.train_data[col].values
            distinct=[]
            for i in range(len(vals)):
                temp=vals[i]
                temp=temp.replace('[','')
                temp=temp.replace(']','')
                temp=temp.replace('\'','')
                temp=self.strip_spaces(temp.split(','))
                vals[i]=list(temp) 
                distinct=list(set(distinct + list(temp)))
            # One Hot Encoding
            for i in distinct:
                new_col_name=col + '_' + i
                new_col_values=[]
                for j in vals:
                    if i in j:
                        new_col_values.append(1)
                    else:
                        new_col_values.append(0)
                self.train_data[new_col_name]=new_col_values
                self.train_data[new_col_name]=self.train_data[new_col_name].astype('int')
        print(f"\n\n**** Preprocessing Product_Holding_B1 column in test set ****\n\n")
        vals=self.test_data['Product_Holding_B1'].values
        distinct=[]
        for i in range(len(vals)):
            temp=vals[i]
            temp=temp.replace('[','')
            temp=temp.replace(']','')
            temp=temp.replace('\'','')
            temp=self.strip_spaces(temp.split(','))
            vals[i]=list(temp)
            distinct=list(set(distinct + list(temp)))
        for i in distinct:
            new_col_name='Product_Holding_B1_'+i
            new_col_values=[]
            for j in vals:
                if i in j:
                    new_col_values.append(1)
                else:
                    new_col_values.append(0)
            self.test_data[new_col_name]=new_col_values
            self.test_data[new_col_name]=self.test_data[new_col_name].astype('int')
    def numericEncoding(self):
        print("\n\n**** Performing numeric/onehot encoding of object type columns in train set ****\n\n")
        for col in list(set(self.train_data.columns) - set(['Customer_ID','Product_Holding_B1','Product_Holding_B2'])):
            if self.train_data[col].dtype == 'object' and self.train_data[col].nunique() == 2:
                distinct=self.train_data[col].unique()
                mapping={distinct[i] : i for i in range(len(distinct))}
                self.train_data[col].replace(mapping,inplace=True)
            elif self.train_data[col].dtype == 'object' and self.train_data[col].nunique() > 2:
                dummies=get_dummies(self.train_data[col],prefix=col,drop_first=False)
                self.train_data=concat([self.train_data,dummies],axis=1)
        print("\n\n**** Performing numeric/onehot encoding of object type columns in test set ****\n\n")
        for col in list(set(self.test_data.columns) - set(['Customer_ID','Product_Holding_B1','Product_Holding_B2'])):
            if self.test_data[col].dtype == 'object' and self.test_data[col].nunique() == 2:
                distinct=self.test_data[col].unique()
                mapping={distinct[i] : i for i in range(len(distinct))}
                self.test_data[col].replace(mapping,inplace=True)
            elif self.test_data[col].dtype == 'object' and self.test_data[col].nunique() > 2:
                dummies=get_dummies(self.test_data[col],prefix=col,drop_first=False)
                self.test_data=concat([self.test_data,dummies],axis=1)
    def savePreprocesedData(self,train_file_path='Dataset/preprocessed_train.csv',test_file_path='Dataset/preprocessed_test.csv'):
        self.train_data.to_csv(train_file_path,index=False)
        self.test_data.to_csv(test_file_path,index=False)
def test():
    preprocessor=Preprocessing()
    preprocessor.handleMissingData()
    preprocessor.mapProductHolding()
    preprocessor.numericEncoding()
    preprocessor.savePreprocesedData()
if __name__=='__main__':
    test()
