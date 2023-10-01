from tagger.Tagger import Tagger
from lexicon.LexiconProcessor import LexiconProcessor
from data.CONLLReader import CONLLReader
import os
from transformers import AutoConfig 
from itertools import product
from classification.Classifier import Classifier
from tokenization.Tokenization import normalize_tokens
import numpy as np
import logging

from datasets import Dataset, DatasetDict

def tokenize_and_align_labels(data, classifier, tag2id, print_output):
    
    print(data)
    
    encodings = classifier.tokenizer(data['tokens'], padding='max_length', truncation=True, max_length=512, is_split_into_words=True, return_offsets_mapping=True)
    labels = data['labels']
    
    if classifier.ignore_label is not None:
        tag2id[classifier.ignore_label] = -100
    
    if classifier.unknown_label is not None and not classifier.unknown_label in tag2id:
        tag2id[classifier.unknown_label] = 0
    
    # SentencePiece uses a special character, (U+2581) 'lower one eighth block' to reconstruct spaces. Sometimes this character gets tokenized into its own subword, needing special control behavior.
    # Below, we set all seperately tokenized U+2581 to -100.
    prefix_subword_id = None
    if '▁' in classifier.tokenizer.vocab.keys():
        prefix_subword_id = classifier.tokenizer.convert_tokens_to_ids('▁')

    encoded_labels = []
    idx = 0

    # We use a subword tokenizer, so there are more tokens than labels. We need the offset mapping to link each label to the last subword of each original word.
    for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
        initial_length = len(doc_labels)
        doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100  # we initialize an array with a length equal to the number of subwords, and fill it with -100, an unreachable number to avoid confusion.

        current_x, current_y, label_match_id = -1, -1, -1
        subword_counter = 0
        prefix_counter = 0
        if idx <= 5 and print_output:
            print(idx)
        
        if len(doc_offset) > 512:
            print("Offset wrong")
        
        for i, (x, y) in enumerate(doc_offset):  # We loop through the offset of each document to match the word endings.
            if i == len(doc_offset) - 1:  # Catches the edge case in which there are 512 subwords.
                if (x, y) != (0, 0):
                    label_match_id += 1
                    doc_enc_labels[i] = doc_labels[label_match_id]
                    if idx <= 5 and print_output:
                        print(i, label_match_id, x, y)
                    subword_counter = 0
            else:
                next_x, next_y = doc_offset[i + 1]

                # Each new word starts with x = 0. Subsequent subwords follow the pattern y = next_x
                # For example (0, 1) (1, 4) (4, 6)
                # If a sentence does not need the full 512 subwords (most cases): the remaining ones are filled with (0,0): see edge case supra
                
                # Necessary for SentencePiece, see above.                    
                if prefix_subword_id is not None and encodings[idx].ids[i] == prefix_subword_id:
                    doc_enc_labels[i] = -100
                    prefix_counter += 1

                elif y != next_x and next_x == 0:
                    label_match_id += 1
                    doc_enc_labels[i] = doc_labels[label_match_id]  # Switches the initial -100 to the correct label.
                    if idx <= 5 and print_output:
                        print(i, label_match_id,classifier.tokenizer.decode(encodings.encodings[idx].ids[i - subword_counter:i + 1]),classifier.id2tag[doc_labels[label_match_id]])
                    subword_counter = 0

                else:
                    subword_counter += 1

        result = 0
        for number in doc_enc_labels:  # Sanity check: the number of labels should be equal to the number of words at the end of sentence.
            if number != -100:
                result += 1

        if initial_length != result:
            logging.log(0, f"Result doesn't match length at {idx}")

        encoded_labels.append(doc_enc_labels.tolist())

        idx += 1
    
    encodings['labels'] = encoded_labels

    return encoded_labels


if __name__ == '__main__':
    
    classifier = Classifier('C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer','C:/Users/u0111778/Documents/LanguageModels/morphology_test/models',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',training_data='C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test.txt')
    wids, tokens, tags = classifier.reader.read_tags(data=classifier.training_data, feature='XPOS')
    tokens_norm = normalize_tokens(tokens,'greek_glaux')
    unique_tags = set(tag for doc in tags for tag in doc)
    classifier.tag2id = {tag: s_id for s_id, tag in enumerate(sorted(unique_tags))}
    classifier.id2tag = {s_id: tag for tag, s_id in classifier.tag2id.items()}
        
    training_dataset = []
    
    for sent_id, sent in enumerate(tokens_norm):
        sent_dict = dict()
        sent_dict['tokens'] = tokens_norm[sent_id]
        sent_dict['labels'] = [classifier.tag2id[tag] for tag in tags[sent_id]]
        training_dataset.append(sent_dict)
        

    dataset = Dataset.from_list(training_dataset)
    
    print(dataset[0])
    
    #dataset = dataset.map(lambda examples: tokenize_and_align_labels(examples,classifier=classifier,tag2id=classifier.tag2id,print_output=False), batched=True)
        
    #tokenized_data = dataset.map(tokenize_and_align_labels,batched=True,fn_kwargs={"classifier":classifier,"tag2id":classifier.tag2id,"print_output":False})
        
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