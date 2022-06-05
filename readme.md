# Association Rule Mining

The agenda of this codebase is to learn the inherent association rules between the feature space that consists of demographic data, historical buying pattern, age, gender, etc., data about the customers of a fictitious consumer market. With the learned pattern we have to come up with a machine learning model that can predict the products that have maximum liklihood of being bought by the customers in the test dataset. 

# Approach

The pipeline of this project is no different to a general machine learning solution which kicks off with preprocessing, then feature engineering , model training and generating predictions for the test set. 

## Preprocessing
As part of preprocessing, Following steps are followed:
	1. Missing data imputation, The string type columns except the historical buying pattern columns and 'Customer_ID' is mapped to 'MISSING' value for occurances of NaN in respective column. The Integer type columns are mapped with mode of the non NaN values present in that column for all NaN occurances. For float type columns, the NaN values are replaced with mean of the non NaN values present in the column.
	2. Product Holding columns are multivalued in nature and thus are onehot vector encoded after removing the leading and trailing brackets and commas. For Example: row("[P1,P17]") => row_P1() and row_P17() Binary columns that have 1 if corresponding product is there in row and 0 if not.  
	3. Numeric Encoding, All string type columns are converted to either binary columns (if the unique values present in that column is <=2) else are one hot vector encoded.
	4. All the preprocessed dataframes are saved in the workspace to be used in further steps.

## Feature Selection
For feature selection, I am using the RandomForestClassifier to assess the feature importances of each column for every target product column and select those features that have feature importance value greater than mean. With this we get selected feature list for each future product holding which is used to compute the liklihood of it being bought. This is saved as a csv file to be used in the modelling step.

## Model training
The preprocessed data is read from the file in workspace and is used to train the logistic regression model to predict the liklihood of each future product holdings seperately and saved as a pickle file(serialized). The evaluation is performed on average presicion metric. To make the train-test split of data fair, I am using a custom split function that divides the data in such a manner that each class gets it's representation in both train and test splits.

## Prediction
The prediction is generated on the test dataset using the saved model for each target product holding and then the results are merged in a way that all the product holding that have value 1 as prediction gets added on the future holding list. The prediction gets saved in the output folder.

# Execution

For the convenience, I have created an orchestration python script which combines the function calls for all the steps in chronological fashion. You can execute the file using below command.
	$ python orchestration.py

# TODO:
	1. I am looking forward to perform indepth EDA of the preprocessed dataset.
	2. Try out Non linear classification models.
	3. Try out deep learning models.

