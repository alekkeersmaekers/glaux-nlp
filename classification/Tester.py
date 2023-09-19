from tagger.Tagger import Tagger
from lexicon.LexiconProcessor import LexiconProcessor
from data.CONLLReader import CONLLReader
import os
from transformers import AutoConfig 
from itertools import product

if __name__ == '__main__':
    
    tagger = Tagger(training_data='C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test.txt',tokenizer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/tokenizer',transformer_path='C:/Users/u0111778/Documents/LanguageModels/greek_small_cased_model/discriminator',feats=['XPOS','FEATS'],model_dir='C:/Users/u0111778/Documents/LanguageModels/morphology_test/models',unknown_label='_')
    tagger.possible_tags = tagger.build_possible_tags()
    #for tag in tagger.possible_tags:
    #    print(tag)
    
    wids, tokens = tagger.read_string('θύσανοι δὲ κατῃωρεῦντο φαεινοὶ χρύσειοι·',lang='greek_glaux')
    
    
    
    #tagger.test_data = tagger.reader.parse_conll('C:/Users/u0111778/Documents/LanguageModels/morphology_test/data_test_2_normalized.txt')
    #wids, tokens = tagger.reader.read_tags(data=tagger.test_data, feature=None, return_tags=False)
    tagger.lexicon = tagger.read_lexicon('C:/Users/u0111778/Documents/LanguageModels/morphology_test/lexicon_decapitalized.txt')
    tagger.trim_lexicon()
    all_preds = tagger.tag_seperately(wids, tokens, True)
    best_tags, all_tags, num_poss = tagger.tag_data(tokens,all_preds,True,True)
    prediction = tagger.prediction_string(tokens,wids,all_tags)
    print(prediction)
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