import pandas as pd
from data import TabularDatasets

class TabularClassifier:
    
    def __init__(self, features=None, td_file=None, td_format='tabular', test_file=None, test_format='tabular', transformer_embeddings_training=None, transformer_embeddings_test=None, normalize_columns=None, normalization='NFC', class_column=None):
        if td_file is not None:
            if td_format == 'tabular':
                if class_column is not None:
                    self.training_data = pd.read_csv(td_file, sep="\t", header=0, encoding="utf-8", quoting=3, dtype={class_column:'category'})
                else:
                    self.training_data = pd.read_csv(td_file, sep="\t", header=0, encoding="utf-8", quoting=3)
            if normalize_columns is not None:
                for column in normalize_columns:
                    self.training_data = TabularDatasets.normalize_unicode(self.training_data, column, normalization)
            if features is not None:
                self.training_data = TabularDatasets.add_features(self.training_data, features, transformer_embeddings_training)
        if test_file is not None:
            if test_format == 'tabular':
                if class_column is not None:
                    self.test_data = pd.read_csv(test_file, sep="\t", header=0, encoding="utf-8", quoting=3)
                else:
                    self.test_data = pd.read_csv(test_file, sep="\t", header=0, encoding="utf-8", quoting=3, dtype={class_column:'category'})
            if normalize_columns is not None:
                for column in normalize_columns:
                    self.test_data = TabularDatasets.normalize_unicode(self.test_data, column, normalization)
            if features is not None:
                self.test_data = TabularDatasets.add_features(self.test_data, features, transformer_embeddings_test)