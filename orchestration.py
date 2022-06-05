import preprocessing
import feature_selection
import model

preprocessor=preprocessing.Preprocessing()
preprocessor.handleMissingData()
preprocessor.mapProductHolding()
preprocessor.numericEncoding()
preprocessor.savePreprocesedData()

selected=feature_selection.featureImportanceBasedSelection()
max_length=max([len(selected[i]) for i in selected.keys()])
for i in selected.keys():
    if len(selected[i]) < max_length:
        selected[i].extend(list(['' for j in range(max_length - len(selected[i]))]))
DataFrame(selected).to_csv('./workspace/selected_features.csv',index=False)

models_df=model.train()
model.generate_predictions(models_df)