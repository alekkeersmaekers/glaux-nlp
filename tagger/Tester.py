from tagger.Tagger import Tagger
from lexicon.WordListExtractor import WordListExtractor
from lexicon.MorpheusProcessor import MorpheusProcessor
import torch
from transformers import ElectraTokenizerFast
import os
import multiprocessing
from functools import partial
import glob
from lexicon.LexiconProcessor import LexiconProcessor
from data.CONLLReader import CONLLReader

def read_lexicon_with_lemma(tagger, file):
    lexicon = {}
    file = open(file, encoding='utf-8')
    raw_text = file.read().strip()
    lines = raw_text.split('\n')
    header = lines.pop(0).split('\t')
    feat_col = {}
    for col, feat in enumerate(header):
        feat_col[feat] = col
    for line in lines:
        entry = line.split('\t')
        form = entry[0]
        tag = []
        for feat in tagger.feature_dict:
            tag.append((feat, entry[feat_col[feat]]))
        tag.append(('lemma', entry[feat_col['lemma']]))
        tag = tuple(tag)
        if form in lexicon:
            tags = lexicon[form]
            if tag not in tags:
                tags.append(tag)
        else:
            tags = []
            tags.append(tag)
            lexicon[form] = tags
    return lexicon

def process_file(file, tagger, tagged_dir):
    tagger.test_reader = CONLLReader(file)
    tagger.test_data = tagger.test_reader.parse_conll()
    wids, tokens = tagger.read_tags(data=tagger.test_data, feature=None, return_tags=False)
    all_preds = {}
    for feat in tagger.feature_dict:
        tags = tagger.read_tags(feat, tagger.test_data, return_words=False)
        preds = tagger.predict_feature(f"{tagger.model_dir}/{feat}", wids, tokens, tags)
        all_preds[feat] = preds
        print("Predicted "+feat)
    tagger.tag_data(wids=wids, tokens=tokens, preds=all_preds,output_data=tagged_dir+'/'+os.path.basename(file),output_format='CONLL')

if __name__ == '__main__':
    tagger = Tagger(transformer_model='mercelisw/electra-grc',training_data='files/greek/Data_Training.txt', include_upos=False,include_xpos=True, model_dir='models')
    tagger.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(tagger.device)
    tagger.possible_tags = tagger.read_possible_tags('files/greek/PossibleTags.txt')
    tagger.tokenizer = ElectraTokenizerFast.from_pretrained('mercelisw/electra-grc', do_lower_case=False, strip_accents=False,model_max_length=512)
    tagger.lexicon = read_lexicon_with_lemma(tagger,r'C:\Users\u0111778\OneDrive - KU Leuven\Glaux\1.0\nlp\Morpheus\lexicon_65BC.txt')
    lp = LexiconProcessor(tagger.lexicon)
    lp.add_data(tagger.training_data,['XPOS','person','number','tense','mood','voice','gender','case','degree','lemma'],col_token=1,col_lemma=2,col_upos=3,col_xpos=4,col_morph=5)
    lp.write_lexicon(r'C:\Users\u0111778\OneDrive - KU Leuven\Glaux\1.0\nlp\Morpheus\lexicon_65BC_lemmatization.txt','CONLL',['number','gender','case','degree','person','tense','mood','voice'])
    tagger.trim_lexicon()
    pool = multiprocessing.Pool()
    files = glob.glob('C:/Users/u0111778/OneDrive - KU Leuven/Glaux/1.0/nlp/Tokenized_TXT/test/**')
    func = partial(process_file, tagger=tagger, tagged_dir='C:/Users/u0111778/OneDrive - KU Leuven/Glaux/1.0/nlp/tagged')
    pool.map(func,files)