import pandas as pd
import unicodedata as ud
from sklearn import preprocessing

def add_features(dataset,features,transformer_embeddings=None):
    for feature in features:
            if feature[0] == 'static_embedding':
                dataset = add_static_embedding(dataset,**feature[1])
            elif feature[0] == 'transformer_embedding':
                if transformer_embeddings is not None:
                    dataset = add_transformer_embedding(dataset,transformer_embeddings,**feature[1])
            elif feature[0] == 'capital':
                if len(feature) == 2:
                    dataset = add_capital_feature(dataset,**feature[1])
                else:
                    dataset = add_capital_feature(dataset)
            elif feature[0] == 'gazetteer':
                dataset = add_gazetteer_feature(dataset,**feature[1])
    return dataset

def add_static_embedding(dataset,vector_file,index_name="LEMMA",feature_name=None,fill_nas=None,header=0,normalize=False):
    if index_name in dataset:
        vectors = pd.read_csv(vector_file, sep="\t", index_col=0, header=header, quoting=3)
        if normalize:
            vectors = pd.DataFrame(preprocessing.normalize(vectors),index=vectors.index,columns=vectors.columns)
        if feature_name is not None:
            columns = [f'{feature_name}{i}' for i in range(1,vectors.shape[1]+1)]
            vectors.columns = columns
        vectors[index_name] = vectors.index.astype(str)
        data_vectors = dataset.merge(vectors, on=index_name, how='left')
        if fill_nas is not None:
            data_vectors.fillna(fill_nas, inplace=True)
        return data_vectors
    else:
        print('Column name '+index_name+' does not exist in your dataset. Please make sure it is present.')
        return dataset

def add_transformer_embedding(dataset,transformer_embeddings,index_name="ID",feature_name=None,normalize=False):
    transformer_dataframe = pd.DataFrame.from_dict(transformer_embeddings,orient='index')
    if normalize:
        transformer_dataframe = pd.DataFrame(preprocessing.normalize(transformer_dataframe),index=transformer_dataframe.index,columns=transformer_dataframe.columns)
    if index_name in dataset:
        if feature_name is not None:
            columns = [f'{feature_name}{i}' for i in range(1,transformer_dataframe.shape[1]+1)]
            transformer_dataframe.columns = columns
        transformer_dataframe[index_name] = transformer_dataframe.index.astype(int)
        data_vectors = dataset.merge(transformer_dataframe, on=index_name, how='left')
        return data_vectors
    else:
        print('Column name '+index_name+' does not exist in your dataset. Please make sure it is present.')
        return dataset

def add_capital_feature(dataset,index_name="FORM"):
    dataset['CAPITAL'] = dataset[index_name].str[0].str.isupper()
    return dataset

def add_gazetteer_feature(dataset,gazetteer_file,gazetteer_name=None,index_name="LEMMA",class_name=None,fill_nas=None):
    if class_name is None:
        # gazetteer is a simple list
        with open(gazetteer_file) as infile:
            gazetteer = infile.read().strip().split('\n')
            dataset[gazetteer_name] = dataset[index_name].isin(gazetteer)
    else:
        gazetteer = pd.read_csv(gazetteer_file, sep="\t", header=0, quoting=3)
        dataset = dataset.merge(gazetteer, on=index_name, how='left')
        if fill_nas is not None:
            dataset[[class_name]] = dataset[[class_name]].fillna("none")
        dataset = dataset.astype({class_name: "category"})
    return dataset

def normalize_unicode(dataset,column_name,normalization='NFC'):
    dataset[column_name] = dataset[column_name].map(lambda x: ud.normalize(normalization, x))
    return dataset 