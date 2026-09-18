import csv
import os
import pandas as pd
import pickle
from data import TabularDatasets
from classification.TabularClassifier import TabularClassifier
from collections import Counter

class WordSenseDisambiguator:


    def __init__(self,auto_wsd_dir='auto_training_data_wsd',text_vector_file='wsd/glaux_text_vectors.tsv',glauxid_text='wsd/glauxid_text.tsv'):
        self.auto_wsd_dir = auto_wsd_dir
        self.lemma_id = self.get_lemma_id_mappings()
        if text_vector_file is not None:
            self.text_vector_file = text_vector_file
        if glauxid_text is not None:
            self.glauxid_text = {}
            with open(glauxid_text,encoding='utf8') as infile:
                lines = csv.reader(infile,delimiter='\t')
                for line in lines:
                    self.glauxid_text[int(line[0])] = f'l{line[1]}'
        
    def read_gold_data(self,gold_data_file='wsd/wsd_gold_data_greek.tsv'):
        gold_synset = {}
        gold_certainty = {}
        gold_lemmas = {}
        with open(gold_data_file,encoding='utf8') as infile:
            lines = csv.reader(infile,delimiter='\t')
            for line in lines:
                gold_synset[int(line[0])] = line[1].split('|')
                if line[2] == 'in doubt':
                    gold_certainty[int(line[0])] = 'doubt'
                else:
                    gold_certainty[int(line[0])] = 'certain'
                gold_lemmas[int(line[0])] = line[3]
        return gold_synset, gold_certainty, gold_lemmas
    
    def get_lemma_id_mappings(self):
        lemma_id = {}
        for file in os.listdir(self.auto_wsd_dir):
            file = file.replace('.tsv','')
            split = file.split('_')
            lemma_id[split[1]] = split[0]
        return lemma_id
    
    def get_training_instances(self,lemmas,gold_synset):
        csv.field_size_limit(1000000)
        training_data = {}
        lemma_synset_cluster = {}
        for lemma in lemmas:
            if lemma in self.lemma_id:
                with open(f'{self.auto_wsd_dir}/{self.lemma_id.get(lemma)}_{lemma}.tsv',encoding='utf8') as infile:
                    lines = csv.reader(infile,delimiter='\t')
                    synset_cluster = {}
                    lemma_tdids = {}
                    for line in lines:
                        for synset in line[1].split('|'):
                            synset_cluster[synset] = line[0]
                        for tdid in line[2].split('|'):
                            if not int(tdid) in gold_synset:
                                # We exclude instances that are in the gold data from the training data
                                lemma_tdids[int(tdid)] = line[0]
                    lemma_synset_cluster[lemma] = synset_cluster
                    training_data[lemma] = lemma_tdids
        return training_data, lemma_synset_cluster
    
    def get_test_instances(self,gold_synset,gold_lemmas,lemma_synset_cluster):
        no_clusters = []
        unclustered_synsets = []
        onecluster_synsets = []
        test_data_norm = {}
        for wid, synsets in gold_synset.items():
            lemma = gold_lemmas[wid]
            if not lemma in lemma_synset_cluster:
                no_clusters.append(f'{wid} ({lemma})')
            elif len(set(lemma_synset_cluster[lemma].values())) == 1:
                onecluster_synsets.append(f'{wid} ({lemma})')
                new_synsets = set()
                for synset in synsets:
                    if synset in lemma_synset_cluster[lemma]:
                        new_synsets.add(lemma_synset_cluster[lemma][synset])
                if len(new_synsets)==0:
                    unclustered_synsets.append(f'{wid} ({lemma}, {synset})')
            else:
                lemma_norm = test_data_norm.get(lemma,{})
                new_synsets = set()
                for synset in synsets:
                    if synset in lemma_synset_cluster[lemma]:
                        new_synsets.add(lemma_synset_cluster[lemma][synset])
                if len(new_synsets)==0:
                    unclustered_synsets.append(f'{wid} ({lemma}, {synset})')
                else:
                    lemma_norm[wid] = '|'.join(list(new_synsets))
                    test_data_norm[lemma] = lemma_norm
        print(f'No clusters established for lemma ({len(no_clusters)}): {", ".join(no_clusters)}')
        print(f'Only one cluster established for lemma ({len(onecluster_synsets)}): {", ".join(onecluster_synsets)}')
        print(f'Cluster not found for synset ({len(unclustered_synsets)}): {", ".join(unclustered_synsets)}')
        return test_data_norm
    
    def build_datasets(self,lemma,training_data,test_data,train_vectors_dir,test_vectors_file,use_text_vectors=False):
        train = pd.DataFrame.from_dict(training_data[lemma],orient='index',columns=['SYNSET'])
        train['ID'] = train.index
        with open(f'{train_vectors_dir}/{lemma}.pickle','rb') as infile:
            vectors = pickle.load(infile)
        train = TabularDatasets.add_transformer_embedding(train,vectors,normalize=True,feature_name='Token')
        if use_text_vectors:
            train['TEXT'] = train['ID'].map(self.glauxid_text)
            train = TabularDatasets.add_static_embedding(train,self.text_vector_file,feature_name='Text',index_name='TEXT',normalize=True)
        test = pd.DataFrame.from_dict(test_data[lemma],orient='index',columns=['SYNSET'])
        test['ID'] = test.index
        with open(test_vectors_file,'rb') as infile:
            vectors_test = pickle.load(infile)
        test = TabularDatasets.add_transformer_embedding(test,vectors_test,normalize=True,feature_name='Token')
        if use_text_vectors:
            test['TEXT'] = test['ID'].map(self.glauxid_text)
            test = TabularDatasets.add_static_embedding(test,self.text_vector_file,feature_name='Text',index_name='TEXT',normalize=True)
        return train,test
        
    def train_lemma_model(self,lemma,training_data,test_data,train_vectors_dir,test_vectors_file,use_text_vectors=False):
        train, test = self.build_datasets(lemma,training_data,test_data,train_vectors_dir,test_vectors_file,use_text_vectors=use_text_vectors)
        ignore_columns = ['ID']
        if use_text_vectors:
            ignore_columns.append('TEXT')
        classifier = TabularClassifier(model_type='mlp',ignore_columns=ignore_columns)
        classifier.training_data = train
        classifier.test_data = test
        classifier.test_data.index = test['ID']
        classifier.class_name = 'SYNSET'
        classifier.training_data = classifier.training_data.astype({('SYNSET'): "category"})
        classifier.train(random_state=12345,model_params={"epochs":100},show_progress=False,print_messages=False)
        return classifier
    
    def get_accuracy(self,all_predictions,test_data):
        count_total = 0
        count_correct = 0
        for wid_synset in test_data.values():
            for wid, synset in wid_synset.items():
                if wid in all_predictions:
                    prediction = all_predictions[wid]
                    count_total += 1
                    if prediction in synset.split('|'):
                        count_correct += 1
        accuracy = count_correct / count_total
        print(accuracy,count_correct,count_total)
        return accuracy
    
    def get_baseline(self,training_data):
        lemma_baseline = {}
        for lemma, tokens in training_data.items():
            lemma_baseline[lemma] = Counter(tokens.values()).most_common()[0][0]
        return lemma_baseline
    
    def get_baseline_predictions(self,training_data,test_data):
        lemma_baseline = self.get_baseline(training_data)
        all_predictions = {}
        for lemma, tokens in test_data.items():
            for wid in tokens.keys():
                all_predictions[wid] = lemma_baseline[lemma]
        return all_predictions
    
    def get_accuracy_by_majority_sense(self,training_data,test_data,all_predictions):
        lemma_baseline = self.get_baseline(training_data)
        majority_sense_predictions = {}
        minority_sense_predictions = {}
        for lemma, tokens in test_data.items():
            baseline = lemma_baseline[lemma]
            for token, synset in tokens.items():
                if baseline in synset.split('|'):
                    majority_sense_predictions[token] = all_predictions[token]
                else:
                    minority_sense_predictions[token] = all_predictions[token]
        print('Majority sense accuracy:')
        self.get_accuracy(majority_sense_predictions,test_data)
        print('Minority sense accuracy:')
        self.get_accuracy(minority_sense_predictions,test_data)
    
    def get_accuracy_by_num_senses(self,training_data,test_data,all_predictions):
        lemma_senses = {}
        for lemma, tokens in training_data.items():
            lemma_senses[lemma] = len(set(tokens.values()))
        grouped_predictions = {'2-5':{},'6-10':{},'11-20':{},'>20':{}}
        for lemma, tokens in test_data.items():
            no_senses = lemma_senses[lemma]
            for token in tokens.keys():
                if no_senses <= 5:
                    grouped_predictions['2-5'][token] = all_predictions[token]
                elif no_senses <= 10:
                    grouped_predictions['6-10'][token] = all_predictions[token]
                elif no_senses <= 20:
                    grouped_predictions['11-20'][token] = all_predictions[token]
                else:
                    grouped_predictions['>20'][token] = all_predictions[token]
        for group, predictions in grouped_predictions.items():
            print(group)
            self.get_accuracy(predictions,test_data)
    
    def write_predictions(self,predictions,file):
        with open(file,'w',encoding='utf8') as outfile:
            for k, v in predictions.items():
                outfile.write(f'{k}\t{v}\n')
    
    def read_predictions(self,file):
        predictions = {}
        with open(file,encoding='utf8') as infile:
            lines = csv.reader(infile,delimiter='\t')
            for line in lines:
                predictions[int(line[0])] = line[1]
        return predictions