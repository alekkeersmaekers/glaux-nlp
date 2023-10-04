from tagger.Tagger import Tagger
from lexicon.LexiconProcessor import LexiconProcessor
from data.CONLLReader import CONLLReader
import os
from transformers import AutoConfig, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer
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

if __name__ == '__main__':
    
    classifier = Classifier(transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model',model_dir='C:/Users/u0111778/Documents/LanguageModels/test_wsd',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',test_data='C:/Users/u0111778/OneDrive - KU Leuven/Treebank/RANLP/WSD/TestData_glossa_wsd.txt',ignore_label='_',unknown_label='[UNK]',feature_cols={'ID':1,'FORM':2,'MISC':3})
        
    wids,tokens, tags = classifier.reader.read_tags('MISC', classifier.test_data, in_feats=False)
    tokens_norm = tokens
    tags_dict = {'MISC':tags}
    test_data = classifier.build_dataset(tokens,tags_dict)
    test_data = test_data.map(classifier.tokenize_sentence)
    #test_data = test_data.map(classifier.tokenize_sentence,fn_kwargs={'return_tensors':'pt'})
    #test_data.set_format("pt",columns=["input_ids"],output_all_columns=True)
    
    classifier.classifier_model = AutoModelForTokenClassification.from_pretrained(classifier.model_dir)
    classifier.config = AutoConfig.from_pretrained(classifier.model_dir)
    id2tag = classifier.config.id2label
    
    test_data = test_data.map(classifier.align_labels,fn_kwargs={"prefix_subword_id":classifier.prefix_subword_id,"tag2id":classifier.config.label2id})
    
    labels_name = 'MISC'
    
    data_collator = DataCollatorForTokenClassification(tokenizer=classifier.tokenizer)
    training_args = TrainingArguments(output_dir=classifier.model_dir,per_device_eval_batch_size=16)
    trainer = Trainer(model=classifier.classifier_model,args=training_args,tokenizer=classifier.tokenizer,data_collator=data_collator)
    predictions = trainer.predict(test_data)
    softmaxed_predictions = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions),dim=-1).tolist()
    # This only works when padding is set to the right, since the padded predictions will be longer than valid_subword
    all_preds = []
    for sent_no, sent in enumerate(test_data):
        valid_subwords = classifier.get_valid_subwords(sent['offset_mapping'],sent['input_ids'],prefix_subword_id=classifier.prefix_subword_id)
        for subword_no, valid_subword in enumerate(valid_subwords):
            if valid_subword:
                # If a (sub)word has the label defined by ignore_label, it is also set to -100 with classifier.align_labels, even though it is counted as a valid subword
                if not ('labels' in sent and sent['labels'][subword_no] == -100):
                    preds = {}
                    for prob_n, prob in enumerate(softmaxed_predictions[sent_no][subword_no]):
                        classname = id2tag[prob_n]
                        preds[classname] = prob
                    all_preds.append(preds)
    
    classifier.write_prediction(wids,tokens,tags,all_preds,'C:/Users/u0111778/Documents/LanguageModels/test_wsd.txt','tab')
        
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