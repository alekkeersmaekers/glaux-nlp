import pandas as pd
from data import TabularDatasets
from sklearn import model_selection
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

class TabularClassifier:
    
    def __init__(self, features=None, td_file=None, td_format='tabular', test_file=None, test_format='tabular', normalize_columns=None, normalization='NFC', class_column=None, train_gpu=False):
        if td_file is not None:
            if td_format == 'tabular':
                self.training_data = pd.read_csv(td_file, sep="\t", header=0, encoding="utf-8", quoting=3)
            if normalize_columns is not None:
                for column in normalize_columns:
                    self.training_data = TabularDatasets.normalize_unicode(self.training_data, column, normalization)
            if features is not None:
                self.training_data = TabularDatasets.add_features(self.training_data, features)
            if class_column is not None:
                self.class_name = class_column
                self.training_data = self.training_data.astype({class_column: "category"})
        if test_file is not None:
            if test_format == 'tabular':
                self.test_data = pd.read_csv(test_file, sep="\t", header=0, encoding="utf-8", quoting=3)
            if normalize_columns is not None:
                for column in normalize_columns:
                    self.test_data = TabularDatasets.normalize_unicode(self.test_data, column, normalization)
            if features is not None:
                self.test_data = TabularDatasets.add_features(self.test_data, features)
            if class_column is not None:
                self.class_name = class_column
                self.test_data = self.test_data.astype({class_column: "category"})
        self.train_gpu = train_gpu
    
    def train(self,ignore_columns=None,shuffle_data=True,random_state=None,model_type='xgboost',model_params=None,xgboost_trees=10):
        if model_type == 'xgboost':
            if model_params is None:
                model_params = {}
                model_params['objective'] = 'multi:softmax'
                model_params['tree_method'] = 'hist'
            if random_state is not None:
                model_params['random_state'] = random_state
            if self.train_gpu:
                model_params['device'] = 'cuda'
        if shuffle_data:
            if random_state is None:
                self.training_data = self.training_data.sample(frac=1)
            else:
                self.training_data = self.training_data.sample(frac=1,random_state=random_state)
        print(f'Training classifier')
        train = self.training_data.copy()
        label = train[self.class_name].cat.codes
        label_encoder = train[self.class_name].cat.categories
        if ignore_columns is not None:
            train.drop(columns=ignore_columns,inplace=True)
        train.drop(columns=[self.class_name],inplace=True)
        if model_type == 'xgboost':
            if model_params['objective'] in ['multi:softmax','multi:softprob']:
                model_params['num_class'] = len(label_encoder)
            train_matrix = xgb.DMatrix(data=train,label=label,enable_categorical=True)
            model = xgb.train(model_params,train_matrix,num_boost_round=xgboost_trees)
            if self.test_data is not None:
                test = self.test_data.copy()
                if ignore_columns is not None:
                    test.drop(columns=ignore_columns,inplace=True)
                test.drop(columns=[self.class_name],inplace=True)
                test_matrix = xgb.DMatrix(data=test,enable_categorical=True)
                predictions = model.predict(test_matrix)
        elif model_type == 'mlp':
            model = MLPClassifier(random_state=random_state,**model_params).fit(train,label)
            if self.test_data is not None:
                test = self.test_data.copy()
                if ignore_columns is not None:
                    test.drop(columns=ignore_columns,inplace=True)
                test.drop(columns=[self.class_name],inplace=True)
                predictions = model.predict(test)
        if self.test_data is not None:
            self.set_predictions(label_encoder[predictions.astype(int)])
            print(f'Accuracy: {self.get_accuracy(self.test_data)}')
        self.models = [model]
        self.label_encoders = [label_encoder]

    def train_and_test_nfold(self,ignore_columns=None,n=10,stratified=True,shuffle_data=True,random_state=None,model_type='xgboost',model_params=None,xgboost_trees=10):
        if model_type == 'xgboost':
            if model_params is None:
                model_params = {}
                model_params['objective'] = 'multi:softmax'
                model_params['tree_method'] = 'hist'
            if random_state is not None:
                model_params['random_state'] = random_state
            if self.train_gpu:
                model_params['device'] = 'cuda'
        if shuffle_data:
            if random_state is None:
                self.training_data = self.training_data.sample(frac=1)
            else:
                self.training_data = self.training_data.sample(frac=1,random_state=random_state)
        if stratified:
            kf = model_selection.StratifiedKFold(n_splits=n)
        else:
            kf = model_selection.KFold(n_splits=n)
        split = kf.split(self.training_data,self.training_data[self.class_name])
        all_predictions = []
        models = []
        test_folds = []
        label_encoders = []
        for fold, indices in enumerate(split):
            print(f'Training fold {fold}')
            train = self.training_data.iloc[indices[0]].copy()
            test = self.training_data.iloc[indices[1]].copy()
            label = train[self.class_name].cat.codes
            label_encoder = train[self.class_name].cat.categories
            if ignore_columns is not None:
                train.drop(columns=ignore_columns,inplace=True)
                test.drop(columns=ignore_columns,inplace=True)
            train.drop(columns=[self.class_name],inplace=True)
            test.drop(columns=[self.class_name],inplace=True)
            if model_type == 'xgboost':
                if model_params['objective'] in ['multi:softmax','multi:softprob']:
                    model_params['num_class'] = len(label_encoder)
                train_matrix = xgb.DMatrix(data=train,label=label,enable_categorical=True)
                model = xgb.train(model_params,train_matrix,num_boost_round=xgboost_trees)
                test_matrix = xgb.DMatrix(data=test,enable_categorical=True)
                predictions = model.predict(test_matrix)
            elif model_type == 'mlp':
                model = MLPClassifier(random_state=random_state,**model_params).fit(train,label)
                predictions = model.predict(test)
            all_predictions.extend(label_encoder[predictions.astype(int)])
            models.append(model)
            test_folds.append(test)
            label_encoders.append(label_encoder)
        self.models = models
        self.test_folds = test_folds
        self.label_encoders = label_encoders
        self.set_fold_info()
        self.set_predictions_nfold(all_predictions)
        print(f'Accuracy: {self.get_accuracy(self.training_data)}')
    
    def set_fold_info(self):
        self.row_fold = {}
        self.row_index = {}
        for fold_no, fold in enumerate(self.test_folds):
            fold_index = fold.index
            for token_no, token in enumerate(fold_index):
                self.row_fold[token] = fold_no
                self.row_index[token] = token_no
    
    def set_predictions(self,predictions):
        self.predictions = {}
        index = -1
        for token in self.test_data.index:
            index +=1
            self.predictions[token] = predictions[index]
    
    def set_predictions_nfold(self,predictions):
        self.predictions = {}
        index = -1
        for fold in self.test_folds:
            fold_index = fold.index
            for token in fold_index:
                index +=1
                self.predictions[token] = predictions[index]
    
    def get_accuracy(self,data=None):
        if data is None:
            data = self.test_data
        predictions = data.index.map(self.predictions)
        correct = (predictions == data[self.class_name])
        accuracy = sum(correct) / len(correct)
        return accuracy
    
    def get_shap_values_nfold(self):
        all_shap_values = []
        for fold_no, test_fold in enumerate(self.test_folds):
            explainer = shap.TreeExplainer(self.models[fold_no])
            shap_values = explainer(xgb.DMatrix(test_fold,enable_categorical=True))
            if shap_values.feature_names is None:
                shap_values.feature_names = list(test_fold.columns)
                shap_values.data = np.array(test_fold)
            all_shap_values.append(shap_values)
        return all_shap_values
    
    def merge_shap_values(self,shap_values,feature):
        feature_names = shap_values.feature_names
        start_index = -1
        end_index = -1
        for i, feature_name in enumerate(feature_names):
            if feature_name.startswith(feature):
                if start_index == -1:
                    start_index = i
                end_index = i
        if start_index == -1:
            print('Error: feature not found '+feature)
        else:
            shap_values.feature_names[start_index:(end_index+1)] = [feature]
            values = shap_values.values
            merged_feature_values = np.sum(values[:, start_index:(end_index+1), :], axis=1)
            shap_values.values = np.concatenate([values[:, 0:start_index, :],merged_feature_values[:, np.newaxis, :],values[:,end_index+1:,:]],axis=1)
            data = shap_values.data
            merged_data = np.sum(data[:, start_index:(end_index+1)], axis=1)
            shap_values.data = np.concatenate([data[:, 0:start_index],merged_data[:, np.newaxis],data[:,end_index+1:]],axis=1)
    
    def explain_prediction_nfold(self,all_shap_values,index,data=None,class_label='predicted',form_column=None):
        if data is None:
            data = self.test_data
        fold = self.row_fold[index]
        token_index = self.row_index[index]
        title = 'SHAP values for '
        if form_column is not None:
            title += f'token \'{data.loc[index][form_column]}\', '
        if class_label == 'predicted':
            class_name = self.predictions[index]
            title += f'predicted class \'{class_name}\''
        elif class_label == 'gold':
            class_name = data.loc[index][self.class_name]
            title += f'gold class \'{class_name}\''
        else:
            class_name = class_label
            title += f'class \'{class_name}\''
        class_index = list(self.label_encoders[fold]).index(class_name)
        fig = shap.plots.waterfall(all_shap_values[fold][token_index,:,class_index],show=False)
        plt.title(title)
        plt.show()