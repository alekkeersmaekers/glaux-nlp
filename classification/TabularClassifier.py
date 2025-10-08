import pandas as pd
from data import TabularDatasets
from sklearn import model_selection
import xgboost as xgb
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn import preprocessing
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(100,), output_dim=2, device=torch.device("cpu")):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        self.device = device
        self.to(device)
    
    def forward(self, x):
        return self.net(x)

    def train_mlp(self, X, y, epochs=200, batch_size=200, lr=0.001, weight_decay=0.0001):
        
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),torch.tensor(y, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)
        self.train()
        for epoch in tqdm(range(epochs)):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    def predict(self, X, pred='prob'):
        logits = self(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().detach()
        if pred == 'logit':
            return logits.numpy()
        probs = torch.softmax(logits,dim=1)
        if pred == 'prob':
            return probs.numpy()
        classes = torch.argmax(probs,axis=1)
        if pred == 'class':
            return classes.numpy()

class TabularClassifier:
    
    def __init__(self, features=None, td_file=None, td_format='tabular', test_file=None, test_format='tabular', normalize_columns=None, normalization='NFC', class_column=None, train_gpu=False, model_type='mlp', ignore_columns=None):
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
        self.model_type = model_type
        if self.model_type == 'mlp':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ignore_columns = ignore_columns
    
    def train(self,shuffle_data=True,random_state=None,model_params=None,xgboost_trees=10,mlp_layers=(100,)):
        if self.model_type == 'xgboost':
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
        if self.ignore_columns is not None:
            train.drop(columns=self.ignore_columns,inplace=True)
        train.drop(columns=[self.class_name],inplace=True)
        if self.model_type == 'xgboost':
            if model_params['objective'] in ['multi:softmax','multi:softprob']:
                model_params['num_class'] = len(label_encoder)
            train_matrix = xgb.DMatrix(data=train,label=label,enable_categorical=True)
            model = xgb.train(model_params,train_matrix,num_boost_round=xgboost_trees)
            if self.test_data is not None:
                test = self.test_data.copy()
                if self.ignore_columns is not None:
                    test.drop(columns=self.ignore_columns,inplace=True)
                test.drop(columns=[self.class_name],inplace=True)
                test_matrix = xgb.DMatrix(data=test,enable_categorical=True)
                predictions = model.predict(test_matrix)
        elif self.model_type == 'mlp':
            if random_state is not None:
                torch.manual_seed(random_state)
            model = MLPClassifier(input_dim=len(train.columns),hidden_dims=mlp_layers,output_dim=len(label_encoder),device=self.device)
            model.train_mlp(X=np.array(train.values, dtype=np.float32),y=np.array(label, dtype=np.int64),**model_params)
            if self.test_data is not None:
                test = self.test_data.copy()
                if self.ignore_columns is not None:
                    test.drop(columns=self.ignore_columns,inplace=True)
                test.drop(columns=[self.class_name],inplace=True)
                predictions = model.predict(X=np.array(test.values, dtype=np.float32),pred='class')
        if self.test_data is not None:
            self.set_predictions(label_encoder[predictions.astype(int)])
            print(f'Accuracy: {self.get_accuracy(self.test_data)}')
        self.models = [model]
        self.label_encoders = [label_encoder]

    def train_and_test_nfold(self,n=10,stratified=True,shuffle_data=True,random_state=None,model_type='xgboost',model_params=None,xgboost_trees=10):
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
            if self.ignore_columns is not None:
                train.drop(columns=self.ignore_columns,inplace=True)
                test.drop(columns=self.ignore_columns,inplace=True)
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
        
    def get_indices(self,wid,extractor,tokens,wids,numeric_as_string):
        for sent_no, sent in enumerate(wids):
            if wid in sent or (numeric_as_string==True and str(wid) in sent):
                break
        if numeric_as_string and str(wid) in sent:
            i = sent.index(str(wid))
        else:
            i = sent.index(wid)
        target_token = tokens[sent_no][i]
        tokens_wids = {}
        for token_no, token in enumerate(tokens[sent_no]):
            tok_wids = tokens_wids.get(token,[])
            tok_wids.append(wids[sent_no][token_no])
            tokens_wids[token] = tok_wids
        if numeric_as_string and str(wid) in tokens_wids[target_token]:
            target_no = tokens_wids[target_token].index(str(wid))
        else:
            target_no = tokens_wids[target_token].index(wid)
        sent_str = ' '.join(tokens[sent_no])
        enc = extractor.tokenizer(sent_str, return_offsets_mapping=True, return_tensors="pt")
        word_ids = enc.word_ids()
        offsets = enc["offset_mapping"][0].tolist()
        new_tokens_wids = {}
        for wid in sorted(set(w for w in word_ids if w is not None)):
            idxs = [i for i, w in enumerate(word_ids) if w == wid]
            start = offsets[idxs[0]][0]
            end = offsets[idxs[-1]][1]
            form = sent_str[start:end]
            tok_wids = new_tokens_wids.get(form,[])
            tok_wids.append(wid)
            new_tokens_wids[form] = tok_wids
        new_word_no = new_tokens_wids[target_token][target_no]
        return sent_no, new_word_no, enc
    
    def explain_prediction(self,wid=None,wid_column='ID',shap_sample=None):
        # Not implemented for xgboost yet, but is very straightforward
        if self.model_type == 'mlp':
            train = self.training_data.copy()
            if self.ignore_columns is not None:
                train.drop(columns=self.ignore_columns,inplace=True)
            train.drop(columns=[self.class_name],inplace=True)
            if shap_sample is None:
                explainer = shap.DeepExplainer(self.models[0],torch.tensor(np.array(train.values, dtype=np.float32), dtype=torch.float32).to(self.device))
            else:
                explainer = shap.DeepExplainer(self.models[0],torch.tensor(shap.sample(np.array(train.values, dtype=np.float32),shap_sample), dtype=torch.float32).to(self.device))
            with torch.no_grad():
                base_preds = self.models[0](torch.tensor(np.array(train.values, dtype=np.float32), dtype=torch.float32).to(self.device)).cpu().numpy()
                base_values = base_preds.mean(axis=0)
            if wid is None:
                # If no wid specified, return shap values for everything
                test = self.test_data.copy()
            else:
                test = self.test_data[self.test_data[wid_column]==wid].copy()
            if self.ignore_columns is not None:
                test.drop(columns=self.ignore_columns,inplace=True)
            test.drop(columns=[self.class_name],inplace=True)
            shap_values = explainer(torch.tensor(np.array(test.values,dtype=np.float32), dtype=torch.float32).to(self.device))
            new_shap_values = shap.Explanation(
                values=shap_values.values,
                base_values= np.tile(base_values.reshape(1, -1), (shap_values.values.shape[0], 1)),
                data=np.array(shap_values.data.cpu(),dtype=np.float32),
                feature_names=list(train.columns),
                output_names=shap_values.output_names
            )
            return new_shap_values
    
    def explain_prediction_context(self,wid,extractor,tokens,wids,wid_column='ID',transformer_column='Token',numeric_as_string=False,normalize_embeddings=True):
        sent_no, new_word_no, enc = self.get_indices(wid,extractor,tokens,wids,numeric_as_string)
        test_row = self.test_data[self.test_data[wid_column]==wid]
        ignore_columns = [] if self.ignore_columns is None else self.ignore_columns
        test_row = test_row[[x for x in test_row if x != self.class_name and not x in ignore_columns]]
        column_names = list(test_row.columns)
        test_row = test_row[[x for x in test_row if not transformer_column in x]]
        test_row.index = [0]
        sent_str = ' '.join(tokens[sent_no])
        predict_fn = self.make_predict_fn(token_index=new_word_no,extractor=extractor,test_data=test_row,enc=enc,column_names=column_names,transformer_column=transformer_column,normalize_embeddings=normalize_embeddings)
        explainer = shap.Explainer(predict_fn, extractor.tokenizer,batch_size=64)
        shap_values = explainer([sent_str])
        return shap_values
    
    def make_predict_fn(self,token_index,extractor,test_data,enc,column_names,transformer_column,normalize_embeddings):
        word_ids = enc.word_ids(0)
        offsets = enc["offset_mapping"][0]   
        word_to_subtokens = {}
        for i, wid in enumerate(word_ids):
            if wid is not None:
                word_to_subtokens.setdefault(wid, []).append(i)    
        transformer_cols = [c for c in column_names if transformer_column in c]
        
        def predict(sentences):
            batch_embeddings = []
            sentences = sentences.tolist()
            rows = []
            enc = extractor.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(extractor.model.device)
            for sent_no, sent in enumerate(sentences):
                input_ids = enc['input_ids'][sent_no].unsqueeze(0)
                attention_mask = enc['attention_mask'][sent_no].unsqueeze(0)
                with torch.no_grad():
                    outputs = extractor.model(input_ids=input_ids,attention_mask=attention_mask,output_hidden_states=True)
                hidden_states = outputs.hidden_states
                kept_states = torch.stack([hidden_states[i] for i in extractor.layers])
                embeddings = kept_states[:, 0, :, :].permute(1, 0, 2)
                subtoken_ids = word_to_subtokens[token_index]
                if extractor.subwords_combination_method == 'mean':
                    word_embeddings = embeddings[subtoken_ids, :, :].mean(dim=0)
                elif extractor.subwords_combination_method == 'first':
                    word_embeddings = embeddings[subtoken_ids[0], :, :]
                elif extractor.subwords_combination_method == 'last':
                    word_embeddings = embeddings[subtoken_ids[-1], :, :]
                if extractor.layer_combination_method == 'concatenate':
                    target_embedding = word_embeddings.flatten().cpu().numpy()
                else:
                    target_embedding = word_embeddings.sum(dim=0).cpu().numpy()
                if normalize_embeddings:
                    target_embedding = preprocessing.normalize([target_embedding])[0]
                row = test_data.copy()
                transformer_df = pd.DataFrame([target_embedding], columns=transformer_cols)
                row = pd.concat([row, transformer_df], axis=1)
                rows.append(row)
            test = pd.concat(rows, ignore_index=True)
            test = test[column_names]
            # Should give different options for the type of model, right now this only works for MLPClassifier
            logits = self.models[0].predict(X=np.array(test.values, dtype=np.float32),pred='logit')
            return logits
        return predict