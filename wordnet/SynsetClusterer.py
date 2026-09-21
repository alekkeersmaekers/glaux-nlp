import csv
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from sentence_transformers import SentenceTransformer
from vectors.VectorExtractor import VectorExtractor
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve
import pandas as pd
from classification.TabularClassifier import TabularClassifier
from bayes_opt import BayesianOptimization
import scipy.stats as stats
from tqdm import tqdm
import re
from collections import Counter
import json
import pickle
import warnings
from pickle import NONE

class SynsetClusterer:
    
    def __init__(self, sentence_transformer_model='all-MiniLM-L6-v2', token_transformer_model='microsoft/deberta-v3-base', token_transformer_layers=[11], gold_cluster_file='clustering/nouns_gold_cluster_data.tsv', greek_ic_file='ic-greek-nouns.dat', cluster_settings_file='clustering/clustering_settings.json', cluster_similarity_model_file='clustering/cluster_similarity_model', consec_input_dir='consec_inputs_sample', consec_output_dir='consec_outputs_sample', wn_langs=['eng','fin','tha','jpn','por','slv','cmn','ind','eus','arb']):
        if sentence_transformer_model is not None:
            self.sent_transformer = SentenceTransformer(sentence_transformer_model)
        if token_transformer_model is not None:
            self.extractor = VectorExtractor(transformer_path=token_transformer_model,layers=token_transformer_layers)
        if gold_cluster_file is not None:
            self.lemma_gold_clusters, self.lemma_synset_data = self.read_gold_data(gold_cluster_file)
        if greek_ic_file is not None:
            self.greek_ic = wordnet_ic.ic(greek_ic_file)
        if cluster_settings_file is not None:
            with open(cluster_settings_file,encoding='utf8') as infile:
                self.clustering_settings = json.load(infile)
        if cluster_similarity_model_file is not None:
            with open(cluster_similarity_model_file,'rb') as infile:
                self.cluster_similarity_model = pickle.load(infile)
        self.consec_input_dir = consec_input_dir
        self.consec_output_dir = consec_output_dir
        if consec_output_dir is not None:
            self.lemma_id = self.get_lemma_id_mappings(consec_output_dir)
        if wn_langs is not None:
            self.wn_langs = wn_langs
        self.lch_similarities = {}
        self.wup_similarities = {}
        self.res_similarities = {}
        self.jcn_similarities = {}
        self.lin_similarities = {}
        self.definition_similarities = {}
        self.lexicalization_similarities = {}
        self.def_vectors = {}
    
    def read_gold_data(self,file):
        lemma_gold_clusters = {}
        lemma_synset_data = {}
        with open(file,encoding='utf8') as infile:
            lines = csv.reader(infile,delimiter='\t')
            for line in lines:
                lemma = line[0]
                synset = wn.synset(line[1])
                cluster = line[3]
                if cluster != 'NA':
                    synset_cluster = lemma_gold_clusters.get(lemma,{})
                    synset_cluster[synset] = int(cluster)
                    lemma_gold_clusters[lemma] = synset_cluster
                synset_data = lemma_synset_data.get(lemma,{})
                instances = synset_data.get(synset,[])
                if line[2] == 'None':
                    instances.append([1,float(line[4])])
                else:
                    instances.append([float(line[2]),float(line[4])])
                synset_data[synset] = instances
                lemma_synset_data[lemma] = synset_data
        return lemma_gold_clusters, lemma_synset_data
    
    def get_lemma_id_mappings(self,consec_dir):
        lemma_id = {}
        for file in os.listdir(consec_dir):
            file = file.replace('.tsv','')
            split = file.split('_')
            lemma_id[split[1]] = split[0]
        return lemma_id
    
    def build_token_embeddings(self,lemmas,threshold=0,batch_size=50):
        synsets_vectors = {}
        for lemma in lemmas:
            lemma_id = self.lemma_id[lemma]
            wids = []
            tokens = []
            limit_ids = []
            glauxid_vector = {}
            with open(f'{self.consec_input_dir}/{lemma_id}_{lemma}.tsv',encoding='utf8') as infile:
                lines = infile.readlines()
                for line_no, line in enumerate(lines):
                    sl = line.strip().split('\t')
                    sent = sl[2].split(' ')
                    ids = []
                    for word_no, _ in enumerate(sent):
                        ids.append(f'{line_no}_{word_no}')
                    limit_ids.append(f'{line_no}_{sl[3]}')
                    wids.append(ids)
                    tokens.append(sent)
                    glauxid_vector[sl[0]] = f'{line_no}_{sl[3]}'
            self.extractor.limit_wids = limit_ids
            dataset = self.extractor.build_dataset(wids,tokens,batched=True,batch_size=batch_size)
            vectors = self.extractor.extract_vectors(dataset)
            synset_ids = {}
            with open(f'{self.consec_output_dir}/{lemma_id}_{lemma}.tsv',encoding='utf8') as infile:
                lines = infile.readlines()
                for line in lines:
                    sl = line.strip().split('\t')
                    if sl[2] == 'None' or float(sl[2]) >= threshold:
                        ids = synset_ids.get(sl[1],[])
                        ids.append(sl[0])
                        synset_ids[sl[1]] = ids
            synset_vector = {}
            for synset, ids in synset_ids.items():
                svectors = []
                for glaux_id in ids:
                    if glaux_id in glauxid_vector:
                        svectors.append(vectors[glauxid_vector[glaux_id]])
                if len(svectors) > 0:
                    vector = np.mean(svectors,axis=0)
                    synset_vector[synset] = vector
            synsets_vectors.update(synset_vector)
        matrix = []
        for vector in synsets_vectors.values():
            matrix.append(vector)
        cosine_similarities = cosine_similarity(matrix)
        cosine_index = list(synsets_vectors.keys())
        cosine_index = [wn.synset(x) for x in cosine_index]
        return cosine_similarities, cosine_index
    
    def get_synset_string(self,synset):
        return synset.definition()
    
    def get_definition_similarity_cluster(self,synset,cluster):
        similarities = []
        def1 = self.get_synset_string(synset)
        emb1 = self.sent_transformer.encode(def1)
        for synset2 in cluster:
            def2 = self.get_synset_string(synset2)
            emb2 = self.sent_transformer.encode(def2)
            def_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            similarities.append(def_sim)
        return np.max(similarities)
    
    def get_bert_similarity_cluster(self,synset,cluster,cosine_similarities,cosine_index):
        similarities = []
        for synset2 in cluster:
            similarities.append(cosine_similarities[cosine_index.index(synset)][cosine_index.index(synset2)])
        return np.max(similarities)
    
    def get_lexicalization_similarity_cluster(self,synset,cluster):
        similarities = []
        for synset2 in cluster:
            similarity = self.get_lexicalization_similarity(synset,synset2)
            similarities.append(similarity)
        return np.max(similarities)
    
    def get_lexicalization_similarity(self,synset1,synset2):
        total_shared = 0
        langs = []
        for lang in self.wn_langs:
            shared_lemmas = False
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore',message='cmn: invalid offset.*')
                lemma_names_1 = synset1.lemma_names(lang)
                lemma_names_2 = synset2.lemma_names(lang)
            if len(lemma_names_1)>0 and len(lemma_names_2)>0:
                langs.append(lang)
                for lemma in lemma_names_1:
                    if lemma in lemma_names_2:
                        shared_lemmas = True
                        break
                total_shared += shared_lemmas
        return total_shared/len(langs)
    
    def get_wordnet_similarities_cluster(self,synset,cluster):
        path_similarities = []
        lch_similarities = []
        wup_similarities = []
        res_similarities = []
        jcn_similarities = []
        lin_similarities = []
        for synset2 in cluster:
            path_similarities.append(synset.path_similarity(synset2))
            lch_similarities.append(synset.lch_similarity(synset2))
            wup_similarities.append(synset.wup_similarity(synset2))
            res_similarities.append(synset.res_similarity(synset2,self.greek_ic))
            jcn_similarities.append(synset.jcn_similarity(synset2,self.greek_ic))
            lin_similarities.append(synset.lin_similarity(synset2,self.greek_ic))
        return np.max(path_similarities), np.max(lch_similarities), np.max(wup_similarities), np.max(res_similarities), np.max(jcn_similarities), np.max(lin_similarities)
    
    def build_training_data(self,cosine_similarities=None,cosine_index=None):
        data = []
        for lemma, gold_clusters in tqdm(self.lemma_gold_clusters.items()):
            current_clusters = []
            synset_counts = {}
            for synset, synset_data in self.lemma_synset_data[lemma].items():
                synset_counts[synset] = len(synset_data)
            synset_counts = {k: v for k, v in sorted(synset_counts.items(), key=lambda item: item[1],reverse=True)}
            for synset in synset_counts.keys():
                if synset in gold_clusters:
                    found = False
                    for cluster in current_clusters:
                        definition_similarity = self.get_definition_similarity_cluster(synset,cluster)
                        bert_similarity = self.get_bert_similarity_cluster(synset,cluster,cosine_similarities,cosine_index)
                        lexicalization_similarity = self.get_lexicalization_similarity_cluster(synset,cluster)
                        _, lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity = self.get_wordnet_similarities_cluster(synset,cluster)
                        if gold_clusters[synset] == gold_clusters[cluster[0]]:
                            if not found:
                                cluster.append(synset)
                            found = True
                            data.append([lemma,synset,cluster[0],True,definition_similarity,bert_similarity,lexicalization_similarity,lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity])       
                            #data.append([synset,cluster[0],True,definition_similarity,bert_similarity,lexicalization_similarity,path_similarity, lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity])
                        else:
                            data.append([lemma,synset,cluster[0],False,definition_similarity,bert_similarity,lexicalization_similarity,lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity])
                            #data.append([synset,cluster[0],False,definition_similarity,bert_similarity,lexicalization_similarity,path_similarity, lch_similarity, wup_similarity, res_similarity, jcn_similarity, lin_similarity])
                    if not found:
                        new_cluster = [synset]
                        current_clusters.append(new_cluster)
        return data
    
    def train_classifier(self,data,nfold=True,group_by_lemma=False,ignore_columns=['LEMMA','SYNSET1','SYNSET2']):
        df = pd.DataFrame(data,columns=['LEMMA','SYNSET1','SYNSET2','MERGE','DEFINITION_SIM','TOKEN_SIM','LEX_SIM','LCH_SIM','WUP_SIM','RES_SIM','JCN_SIM','LIN_SIM'])
        classifier = TabularClassifier(model_type='logistic',ignore_columns=ignore_columns)
        classifier.training_data = df.copy()
        classifier.class_name = 'MERGE'
        classifier.training_data['MERGE'] = classifier.training_data['MERGE'].astype('category')
        if nfold:
            folds_by_column = None
            if group_by_lemma:
                folds_by_column = 'LEMMA'
            classifier.train_and_test_nfold(folds_by_column=folds_by_column,n=10,stratified=False,random_state=12345,model_params={'penalty':'l1','solver':'liblinear','C':1,'random_state':12345})
        else:
            classifier.train(random_state=12345,model_params={'penalty':'l1','solver':'liblinear','C':0.5,'random_state':12345})
            self.cluster_similarity_model = classifier.models[0]
        return classifier
    
    def get_optimal_threshold(self,classifier):
        pred_frame = classifier.training_data.copy()
        pred_frame['PRED'] = pred_frame.index.map(classifier.predictions)
        precision, recall, thresholds = precision_recall_curve(pred_frame['MERGE'], pred_frame['PRED'])
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]
        print("Threshold:", best_threshold)
        print("F1 score:", best_f1)
        print("Precision:", precision[best_idx])
        print("Recall:", recall[best_idx])
        return best_threshold
    
    def get_cluster_prob(self,synset,synset2,feature_names=['DEFINITION_SIM','LEX_SIM','LCH_SIM','RES_SIM'],cosine_similarities=None,cosine_index_map=None):
        name1 = synset.name()
        name2 = synset2.name()
        if name1 > name2:
            key = name1 + '_' + name2
        else:
            key = name2 + '_' + name1
        features = []
        if 'DEFINITION_SIM' in feature_names:
            def_sim = self.definition_similarities.get(key)
            if def_sim is None:
                emb1 = self.def_vectors.get(synset)
                emb2 = self.def_vectors.get(synset2)
                if emb1 is None:
                    def1 = self.get_synset_string(synset)
                    emb1 = self.sent_transformer.encode(def1)
                    self.def_vectors[synset] = emb1
                if emb2 is None:
                    def2 = self.get_synset_string(synset2)
                    emb2 = self.sent_transformer.encode(def2)
                    self.def_vectors[synset2] = emb2
                def_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                self.definition_similarities[key] = def_sim
            features.append(def_sim)
        if 'TOKEN_SIM' in feature_names:
            bert_sim = None
            index1 = cosine_index_map.get(synset,None)
            index2 = cosine_index_map.get(synset2,None)
            if index1 is not None and index2 is not None:
                bert_sim = cosine_similarities[index1,index2]
            features.append(bert_sim)
        if 'LEX_SIM' in feature_names:
            lex_sim = self.lexicalization_similarities.get(key)
            if lex_sim is None:
                lex_sim = self.get_lexicalization_similarity(synset,synset2)
                self.lexicalization_similarities[key] = lex_sim
            features.append(lex_sim)
        if 'LCH_SIM' in feature_names:
            lch_sim = self.lch_similarities.get(key)
            if lch_sim is None:
                lch_sim = synset.lch_similarity(synset2)
                self.lch_similarities[key] = lch_sim
            features.append(lch_sim)
        if 'WUP_SIM' in feature_names:
            wup_sim = self.wup_similarities.get(key)
            if wup_sim is None:
                wup_sim = synset.wup_similarity(synset2)
                self.wup_similarities[key] = wup_sim
            features.append(wup_sim)
        if 'RES_SIM' in feature_names:
            res_sim = self.res_similarities.get(key)
            if res_sim is None:
                res_sim = synset.res_similarity(synset2,self.greek_ic)
                self.res_similarities[key] = res_sim
            if res_sim > 10000:
                # Special case where res_sim returns a ridiculously high value
                res_sim = 0
            features.append(res_sim)
        if 'JCN_SIM' in feature_names:
            jcn_sim = self.jcn_similarities.get(key)
            if jcn_sim is None:
                jcn_sim = synset.jcn_similarity(synset2,self.greek_ic)
                self.jcn_similarities[key] = jcn_sim
            if jcn_sim > 10000:
                # Special case where jcn_sim returns a ridiculously high value
                jcn_sim = 0
            features.append(jcn_sim)
        if 'LIN_SIM' in feature_names:
            lin_sim = self.lin_similarities.get(key)
            if lin_sim is None:
                lin_sim = synset.lin_similarity(synset2,self.greek_ic)
                self.lin_similarities[key] = lin_sim
            features.append(lin_sim)
        preds = self.cluster_similarity_model.predict_proba(pd.DataFrame([features],columns=feature_names))
        return preds[0][1], features
    
    def merge_synsets(self, sorted_synsets, prob_threshold, abs_freq_threshold,rel_freq_threshold):
        total_count = 0
        for freq in sorted_synsets.values():
            total_count += freq
        clusters = {}
        cluster_no = 0
        for synset, freq in sorted_synsets.items():
            most_similar = None
            best_similarity = 0
            for synset2 in clusters:
                similarity, _ = self.get_cluster_prob(synset,synset2,feature_names=self.cluster_similarity_model.feature_names_in_)
                if similarity > best_similarity:
                    best_similarity = similarity
                    most_similar = synset2
            if best_similarity >= prob_threshold and most_similar is not None:
                clusters[synset] = clusters[most_similar]
            elif freq >= abs_freq_threshold and (freq / total_count) >= rel_freq_threshold:
                cluster_no += 1
                clusters[synset] = cluster_no
        return clusters
    
    def evaluate_gold_standard(self,pred_clusters,gold_clusters,pred_synset_counts,gold_synset_counts):
        pred_index_synsets = {}
        gold_index_synsets = {}
        for key, val in pred_clusters.items():
            synsets = pred_index_synsets.get(val,set())
            synsets.add(key)
            pred_index_synsets[val] = synsets
        for key, val in gold_clusters.items():
            synsets = gold_index_synsets.get(val,set())
            synsets.add(key)
            gold_index_synsets[val] = synsets
        pred_cluster_list = list(pred_index_synsets.values())
        gold_cluster_list = list(gold_index_synsets.values())
        precisions = []
        for pred_cluster in pred_cluster_list:
            true_positives = 0
            false_positives = 0
            best_gold_cluster = None
            best_count = 0
            for gold_cluster in gold_cluster_list:
                count = 0
                for synset in gold_cluster:
                    if synset in pred_cluster:
                        count += pred_synset_counts[synset]
                if count > best_count:
                    best_gold_cluster = gold_cluster
                    best_count = count
            total_count = 0
            for synset in pred_cluster:
                total_count += pred_synset_counts[synset]
            true_positives += best_count
            false_positives += (total_count - best_count)
            if true_positives + false_positives == 0:
                precisions.append(0)
            else:
                precisions.append(true_positives / (true_positives + false_positives))
        recalls = []
        for gold_cluster in gold_cluster_list:
            true_positives = 0
            false_negatives = 0
            best_pred_cluster = None
            best_count = 0
            for pred_cluster in pred_cluster_list:
                count = 0
                for synset in pred_cluster:
                    if synset in gold_cluster:
                        count += pred_synset_counts[synset]
                if count > best_count:
                    best_pred_cluster = pred_cluster
                    best_count = count
            total_count = 0
            for synset in gold_cluster:
                total_count += gold_synset_counts[synset]
            true_positives += best_count
            false_negatives += (total_count - best_count)
            if true_positives + false_negatives == 0:
                recalls.append(0)
            else:
                recalls.append(true_positives / (true_positives + false_negatives))
        return np.mean(precisions), np.mean(recalls)
    
    def get_wa_score_threshold(self,min_zscore,items):
        wa_scores = []
        for synset_data in items.values():
            for val in synset_data:
                wa_scores.append(val[1])
        z_scores = stats.zscore(wa_scores)
        wa_score_threshold = 0
        for no, z_score in enumerate(z_scores):
            if z_score < min_zscore:
                wa_score = wa_scores[no]
                if wa_score > wa_score_threshold:
                    wa_score_threshold = wa_score
        return wa_score_threshold
    
    def evaluate_cluster_algorithm(self,min_wa_zscore,min_wsd_prob,cluster_prob_threshold,abs_freq_threshold,rel_freq_threshold):
        precisions = []
        recalls = []
        for lemma in self.lemma_gold_clusters.keys():
            gold_synset_counts = {}
            for synset, synset_data in self.lemma_synset_data[lemma].items():
                gold_synset_counts[synset] = len(synset_data)
            synset_counts = {}
            min_wa_score = self.get_wa_score_threshold(min_wa_zscore,self.lemma_synset_data[lemma])
            for synset, data in self.lemma_synset_data[lemma].items():
                count = 0
                for item in data:
                    if item[0] > min_wsd_prob and item[1] > min_wa_score:
                        count += 1
                synset_counts[synset] = count
            synset_counts = {k: v for k, v in sorted(synset_counts.items(), key=lambda item: item[1],reverse=True)}
            clusters = self.merge_synsets(synset_counts,prob_threshold=cluster_prob_threshold,abs_freq_threshold=abs_freq_threshold,rel_freq_threshold=rel_freq_threshold)
            precision, recall = self.evaluate_gold_standard(clusters,self.lemma_gold_clusters[lemma],synset_counts,gold_synset_counts)
            precisions.append(precision)
            recalls.append(recall)
        macro_precision = sum(precisions) / len(precisions)
        macro_recall = sum(recalls) / len(recalls)
        f1 = 1.25 * ((macro_precision * macro_recall) / ((0.25 * macro_precision) + macro_recall + 1e-9))
        return f1
    
    def optimize_cluster_algorithm(self,n_iter=1000,init_points=100):
        optimizer = BayesianOptimization(f=self.evaluate_cluster_algorithm,pbounds={"min_wa_zscore":(-5,-1),'min_wsd_prob':(0,0.9),'cluster_prob_threshold':(0,0.9),'abs_freq_threshold':(0,10,int),'rel_freq_threshold':(0,0.1)},random_state=12345,verbose=2)
        optimizer.probe(params={'min_wa_zscore':-4.573456169202341,'min_wsd_prob':0.31005365998973955,'cluster_prob_threshold':0.4573856172407377,'abs_freq_threshold':8,'rel_freq_threshold':0})
        optimizer.maximize(n_iter=n_iter,init_points=init_points)
        return optimizer
    
    def read_alignment_scores(self,alignment_dir):
        glauxid_alignmentscore = {}
        for file in tqdm(os.listdir(alignment_dir)):
            with open(f'{alignment_dir}/{file}',encoding='utf8') as infile:
                lines = csv.reader(infile,delimiter='\t',quoting=csv.QUOTE_NONE)
                for line in lines:
                    glauxid_alignmentscore[line[0]] = float(line[3])
        return glauxid_alignmentscore
    
    def build_prediction_data(self,alignment_dir,wsd_prediction_dir='all_consec_predictions'):
        glauxid_alignmentscore = self.read_alignment_scores(alignment_dir)
        lemma_synset_data_test = {}
        for file in tqdm(os.listdir(wsd_prediction_dir)):
            if file.endswith('.tsv'):
                lemma = re.sub('^[0-9]+_','',file.replace('.tsv',''))
                with open(f'{wsd_prediction_dir}/{file}',encoding='utf8') as infile:
                    lines = csv.reader(infile,delimiter='\t',quoting=csv.QUOTE_NONE)
                    for line in lines:
                        synset = wn.synset(line[1])
                        synset_data = lemma_synset_data_test.get(lemma,{})
                        instances = synset_data.get(synset,[])
                        if line[2] == 'None':
                            instances.append([1,glauxid_alignmentscore[line[0]],line[0]])
                        else:
                            instances.append([float(line[2]),glauxid_alignmentscore[line[0]],line[0]])
                        synset_data[synset] = instances
                        lemma_synset_data_test[lemma] = synset_data
        return lemma_synset_data_test
    
    def cluster_lemmas(self,lemmas,lemma_synset_data_test,min_threshold=10,show_progress=True,output_dir=None,print_discarded=False):
        for lemma in tqdm(lemmas,disable=not(show_progress)):
            gold_synset_counts = {}
            for synset, synset_data in lemma_synset_data_test[lemma].items():
                gold_synset_counts[synset] = len(synset_data)
            synset_counts = {}
            min_wa_score = self.get_wa_score_threshold(self.clustering_settings['min_wa_zscore'],lemma_synset_data_test[lemma])
            synset_examples = {}
            for synset, data in lemma_synset_data_test[lemma].items():
                count = 0
                examples = []
                for item in data:
                    if item[0] > self.clustering_settings['min_wsd_prob'] and item[1] > min_wa_score:
                        count += 1
                        examples.append(item[2])
                synset_counts[synset] = count
                synset_examples[synset] = examples
            synset_counts = {k: v for k, v in sorted(synset_counts.items(), key=lambda item: item[1],reverse=True)}
            clusters = self.merge_synsets(synset_counts,prob_threshold=self.clustering_settings['cluster_prob_threshold'],abs_freq_threshold=self.clustering_settings['abs_freq_threshold'],rel_freq_threshold=self.clustering_settings['rel_freq_threshold'])
            cluster_synsets = {}
            for synset, cluster in clusters.items():
                synsets = cluster_synsets.get(cluster,[])
                synsets.append(synset)
                cluster_synsets[cluster] = synsets
            cluster_counts = Counter()
            for synset, cluster in clusters.items():
                cluster_counts[cluster] += synset_counts[synset]
            cluster_synsets_red = {k:v for k,v in cluster_synsets.items() if cluster_counts[k] >= min_threshold}
            if output_dir is None:
                # If no output file is specified, we print the output
                print(lemma)
                for cluster, synsets in cluster_synsets_red.items():
                    print(f'{cluster}\t({cluster_counts[cluster]})')
                    for synset in synsets:
                        print(f'{synset.name()} ({", ".join(synset.lemma_names())}): {synset.definition()}')
                    print()
                if print_discarded:
                    discarded = []
                    print('Discarded')
                    for synset in lemma_synset_data_test[lemma].keys():
                        found = False
                        for synsets in cluster_synsets_red.values():
                            if synset in synsets:
                                found = True
                                break
                        if not found:
                            discarded.append(synset)
                    for synset in discarded:
                        print(f'{synset.name()} ({", ".join(synset.lemma_names())}): {synset.definition()}')
                print('---')
            else:
                if len(cluster_synsets_red) > 0:
                    with open(f'{output_dir}/{self.lemma_id[lemma]}_{lemma}.tsv','w',encoding='utf8') as outfile:
                        for cluster, synsets in cluster_synsets_red.items():
                            cnt = Counter()
                            all_examples = []
                            for synset in synsets:
                                cnt[synset] = synset_counts[synset]
                                all_examples.extend(synset_examples[synset])
                            most_frequent = cnt.most_common(1)[0][0]
                            synsets_str = [x.name() for x in synsets]
                            outfile.write(f"{most_frequent.name()}\t{'|'.join(synsets_str)}\t{('|').join(all_examples)}\n")