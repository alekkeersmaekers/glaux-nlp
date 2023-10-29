from tagger.Tagger import Tagger
from lexicon.LexiconProcessor import LexiconProcessor
from data.CONLLReader import CONLLReader
import os
from transformers import AutoConfig, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer, AutoModel
from itertools import product
from classification.Classifier import Classifier
from tokenization.Tokenization import normalize_tokens
import numpy as np
import logging
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
import torch
from torch.utils.data import DataLoader

from datasets import Dataset, DatasetDict

from tqdm import tqdm

from data import Datasets

import random

from tokenization import Tokenization

from transformers import AutoTokenizer

def align_wids_subwords(sentence, prefix_subword_id=None):
    wids = sentence['wids']
    input_ids = sentence['input_ids'][0]
    offset = sentence['offset_mapping'][0]
    wids_subwords = []
    id = -1
    for i, current_offset in enumerate(offset):
        x = current_offset[0]
        y = current_offset[1]
        if(x==0 and y!=0):
            id+=1
            wids_subwords.append(wids[id])
        elif(x==0 and y==0):
            wids_subwords.append('-1')
        elif(x!=0):
            wids_subwords.append(wids[id])
        if prefix_subword_id is not None and input_ids[i] == prefix_subword_id:
            id-=1
    sentence['wids_subwords'] =  wids_subwords
    return sentence

def get_embeddings(sentence,model):
    with torch.no_grad():
        output = model(input_ids=sentence['input_ids'],token_type_ids=sentence['token_type_ids'],attention_mask=sentence['attention_mask'])
    states = output.hidden_states
    embeddings = torch.stack(states).squeeze().permute(1,0,2)
    sentence['embeddings'] = embeddings
    return sentence

def get_vector(sentence,wid,layers=[-2],layer_combination_method='sum',subwords_combination_method='mean'):
    ids = []
    for no, subword_id in enumerate(sentence['wids_subwords']):
        if subword_id == wid:
            ids.append(no)
    vectors = []
    embeddings = sentence['embeddings']
    for i in ids:
        vectors.append(embeddings[i])
    if subwords_combination_method == 'mean':
        vector = torch.mean(torch.stack(vectors),dim=0)
    elif subwords_combination_method == 'first':
        vector = vectors[0]
    elif subwords_combination_method == 'last':
        vector = vectors[-1]
    vector_layers = []
    for layer in layers:
        vector_layers.append(vector[layer])
    if layer_combination_method == 'concatenate':
        return torch.cat(vector_layers,dim=0).numpy()
    elif layer_combination_method == 'sum':
        return torch.sum(torch.stack(vector_layers),dim=0).numpy()

def extract_vectors(dataset,output_file,limit_wids=None,limit_labels=None,label_name='MISC',layers=[-2],layer_combination_method='sum',subwords_combination_method='mean'):
    with open(output_file, 'w', encoding='UTF-8') as outfile:
        for sent in dataset:
            if limit_wids is not None:
                for word_no, wid in enumerate(sent['wids']):
                    if wid in limit_wids:
                        vector = get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                        outfile.write(wid+'\t'+sent[label_name][word_no])
                        for element in vector:
                            outfile.write("\t"+"{:0.5f}".format(element))
                        outfile.write('\n')
            elif limit_labels is not None:
                for word_no, wid in enumerate(sent['wids']):
                    label = sent[label_name][word_no]
                    if label in limit_labels:
                        vector = get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                        outfile.write(wid+'\t'+label)
                        for element in vector:
                            outfile.write("\t"+"{:0.5f}".format(element))
                        outfile.write('\n')
            else:
                for word_no, wid in enumerate(sent['wids']):
                    vector = get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                    outfile.write(wid+'\t'+sent[label_name][word_no])
                    for element in vector:
                        outfile.write("\t"+"{:0.5f}".format(element))
                    outfile.write('\n')

if __name__ == '__main__':
    
    #reader = CONLLReader(feature_cols={'ID':1,'FORM':2,'MISC':3})
    #tokenizer = AutoTokenizer.from_pretrained('C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer')
    #data = reader.parse_conll('C:/Users/u0111778/OneDrive - KU Leuven/Colleges/Computerlinguistiek voor klassieke talen/Materiaal 2022/Semantics/Data_Kosmos_Form.txt')
    #wids, tokens, tags = reader.read_tags('MISC',data,in_feats=False)
    #tokens = Tokenization.normalize_tokens(tokens, 'greek_glaux')
    #dataset = Datasets.build_dataset(tokens, {'MISC':tags}, wids)
    #dataset = dataset.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":tokenizer,"return_tensors":'pt'})
    #prefix_subword_id = tokenizer.convert_tokens_to_ids('▁')
    #print(dataset)
    #dataset = dataset.map(align_wids_subwords,fn_kwargs={"prefix_subword_id":prefix_subword_id})
    #dataset.set_format("pt", columns=["input_ids","token_type_ids","attention_mask"], output_all_columns=True)
    #print(dataset)
    #model = AutoModel.from_pretrained('C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model',output_hidden_states = True)
    #model.eval()
    #dataset = dataset.map(get_embeddings,fn_kwargs={"model":model})
    #print(dataset[0]['embeddings'].size())
    #extract_vectors(dataset,r'C:\Users\u0111778\Documents\LanguageModels\test_wsd\dataset_kosmos_vecs_conc1234layer.txt',limit_labels=['order','world','decoration'],layers=[1,2,3,4],layer_combination_method='concatenate')
    
    
    #classifier = Classifier(transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model',model_dir='C:/Users/u0111778/Documents/LanguageModels/test_wsd',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',training_data='C:/Users/u0111778/OneDrive - KU Leuven/Colleges/Computerlinguistiek voor klassieke talen/Materiaal 2022/Semantics/Data_Kosmos_Form.txt',ignore_label='_',unknown_label='[UNK]',feature_cols={'ID':1,'FORM':2,'MISC':3})
    #random.Random(123).shuffle(classifier.training_data)
    
    #wids, tokens, tags = classifier.reader.read_tags('MISC', classifier.training_data, in_feats=False)
    #tag2id, id2tag = classifier.id_label_mappings(tags)
    #training_data = Datasets.build_dataset(tokens,{'MISC':tags})
    #training_data = training_data.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":classifier.tokenizer})
    #training_data = training_data.map(classifier.align_labels,fn_kwargs={"prefix_subword_id":classifier.prefix_subword_id,"tag2id":tag2id})
    
    #n_fold = 10
    #fold_size = int(len(training_data) / n_fold)
    #folds = []
    #index = 0
    #for i in range(n_fold):
    #    data = {}
    #    start = index
    #    end = index + fold_size
    #    if i+1 == n_fold or end > (len(training_data)):
    #        end = len(training_data)
    #    rows = range(start,end)
    #    data['test'] = training_data.select(i for i in rows)
    #    data['train'] = training_data.select(i for i in range(len(training_data)) if i not in rows)
    #    folds.append(data)
    #    index = index + fold_size
    
    #n_epochs=[3]
    #batch_sizes=[16]
    #learning_rates = [2e-4]
    #data_collator = DataCollatorForTokenClassification(tokenizer=classifier.tokenizer)
    #classifier.config = AutoConfig.from_pretrained(classifier.transformer_path, num_labels=len(tag2id), id2label=id2tag, label2id=tag2id)
    #for epochs in n_epochs:
    #    for batch_size in batch_sizes:
    #        for learning_rate in learning_rates:
    #            predictions_folds = []
    #            for fold_index, fold in enumerate(folds):
    #                classifier.classifier_model = AutoModelForTokenClassification.from_pretrained(classifier.transformer_path,config=classifier.config)
    #                training_args = TrainingArguments(output_dir=classifier.model_dir,num_train_epochs=epochs,per_device_train_batch_size=batch_size,per_device_eval_batch_size=32,learning_rate=learning_rate,save_strategy='no',fp16=True, gradient_checkpointing=True)
    #                trainer = Trainer(model=classifier.classifier_model,args=training_args,train_dataset=fold['train'],tokenizer=classifier.tokenizer,data_collator=data_collator)
    #                trainer.train()
    #                predictions = trainer.predict(fold['test'])
    #                softmaxed_predictions = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions),dim=-1).tolist()
    #                all_preds = []
    #                for sent_no, sent in enumerate(fold['test']):
    #                    valid_subwords = classifier.get_valid_subwords(sent['offset_mapping'],sent['input_ids'],prefix_subword_id=classifier.prefix_subword_id)
    #                    for subword_no, valid_subword in enumerate(valid_subwords):
    #                        if valid_subword:
                        # If a (sub)word has the label defined by ignore_label, it is also set to -100 with classifier.align_labels, even though it is counted as a valid subword
    #                            if not ('labels' in sent and sent['labels'][subword_no] == -100):
    #                                preds = {}
    #                                for prob_n, prob in enumerate(softmaxed_predictions[sent_no][subword_no]):
    #                                    classname = id2tag[prob_n]
    #                                    preds[classname] = prob
    #                                all_preds.append(preds)
    #                predictions_folds.extend(all_preds)
    #            classifier.write_prediction(wids,tokens,tags,predictions_folds,f'C:/Users/u0111778/Documents/LanguageModels/test_wsd/predictions_kosmos_{epochs}_{batch_size}_{learning_rate}.txt','tab')
        
    #preds_total = []
    #with torch.no_grad():
    #    for sent in tqdm(test_data,desc='Making sentence prediction'):
    #        labels = sent[labels_name]
    #        valid_subwords = classifier.get_valid_subwords(sent['offset_mapping'][0],sent['input_ids'][0],prefix_subword_id=classifier.prefix_subword_id)
    #        input_ids = sent['input_ids']
    #        outputs = classifier.classifier_model(input_ids)
    #        predictions = outputs.logits[0]
    #        softmaxed_predictions = torch.nn.functional.softmax(predictions,dim=-1).tolist()
    #        token_index = -1
    #        for subword_n, softmaxed_prediction in enumerate(softmaxed_predictions):
    #            preds = {}
    #            for prob_n, prob in enumerate(softmaxed_prediction):
    #                classname = id2tag[prob_n]
    #                preds[classname] = prob
    #            if valid_subwords[subword_n] == True:
    #                token_index+=1
    #                if not(classifier.ignore_label is not None and labels[token_index] == classifier.ignore_label):
    #                    preds_total.append(preds)
    
    #classifier.write_prediction(wids,tokens,tags,preds_total,'C:/Users/u0111778/Documents/LanguageModels/test_wsd.txt','tab')
            
    
    #classifier.id2tag[-100] = '_'
    #for sent in tokenized_dataset:
    #    input_ids = sent["input_ids"]
    #    labels = sent["labels"]
    #    for i, input_id in enumerate(input_ids):
    #        print(classifier.tokenizer.decode(input_id)+' '+classifier.id2tag[labels[i]])
    
    
    
    #tagger.possible_tags = tagger.build_possible_tags()
    #for tag in tagger.possible_tags:
    #    print(tag)
    
    #wids, tokens = tagger.read_string('θύσανοι δὲ κατῃωρεῦντο φαεινοὶ χρύσειοι·',lang='greek_glaux')
    
    
    
    #tagger.test_data = tagger.reader.parse_conll('C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test_2_normalized.txt')
    #wids, tokens = tagger.reader.read_tags(data=tagger.test_data, feature=None, return_tags=False)
    #tagger.lexicon = tagger.read_lexicon('C:/Users/u0111778/Documents/LanguageModels/morphology_test/lexicon_decapitalized.txt')
    #tagger.trim_lexicon()
    #all_preds = tagger.tag_seperately(wids, tokens, True)
    #best_tags, all_tags, num_poss = tagger.tag_data(tokens,all_preds,True,True)
    #prediction = tagger.prediction_string(tokens,wids,all_tags)
    #print(prediction)
    #tagger.write_prediction(wids, tokens, best_tags, 'C:/Users/u0111778/Documents/LanguageModels/morphology_test/prediction.txt', 'tab', num_poss)
    #print(all_tags_probs[0])
    
    #combinations = list(product(list1, list2, list3))

    #for combination in combinations:
    #    print(combination)
    
    #tagger = Tagger(training_data='C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test.txt',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/discriminator',feats=['XPOS','FEATS'],model_dir='C:/Users/u0111778/Documents/LanguageModels/morphology_test/models')
    #tagger.train_individual_models(batch_size=2,epochs=1,multicore=True)
    
    #tagger = Tagger(training_data='C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test.txt',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/discriminator',feats=['XPOS','FEATS'],model_dir='C:/Users/u0111778/Documents/LanguageModels/morphology_test/models',unknown_label='_')
    #tagger.lexicon = {}
    #lp = LexiconProcessor(tagger.lexicon)
    #reader = CONLLReader()
    #data = reader.parse_conll('C:/Users/u0111778/OneDrive - KU Leuven/Glaux/1.0/nlp/Morpheus/lexicon_all_conll.txt')
    #lp.add_data(data,feats=['lemma','XPOS','number','gender','case','tense','voice','person','mood','degree'],col_token=0,col_lemma=1,col_xpos=2,col_morph=3)
    #lp.write_lexicon('C:/Users/u0111778/Documents/LanguageModels/morphology_test/lexicon.txt',morph_feats=['number','gender','case','tense','voice','person','mood','degree'])
    
    #tagger = Tagger(training_data='C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test.txt',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/discriminator',feats=['XPOS','FEATS'],model_dir='C:/Users/u0111778/Documents/LanguageModels/morphology_test/models',unknown_label='_')
    
    
    #tagger.test_data = tagger.reader.parse_conll('C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test_2.txt')
    #wids, tokens = tagger.reader.read_tags(data=tagger.test_data, feature=None, return_tags=False)
    #all_preds = tagger.tag_seperately(wids, tokens, multicore=True)
    #for feat in all_preds:
    #    print(all_preds[feat])    