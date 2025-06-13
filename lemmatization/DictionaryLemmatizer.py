import pickle
from tqdm import tqdm
from tokenization import Tokenization

class DictionaryLemmatizer:

    def __init__(self,lemmatizer_file=None):
        self.lemmatizer = {}
        if lemmatizer_file is not None:
            self.load_lemmatizer(lemmatizer_file)

    def build_greek_lemmatizer(self,data,feats={'form':'word','lemma':'lemma_string','pos':'POS_pos','person':'POS_person','number':'POS_number','tense':'POS_tense','mood':'POS_mood','voice':'POS_diathese','gender':'POS_gender','case':'POS_morph_case','degree':'POS_degree'}):
        lemmatizer = {}
        for row in tqdm(data):
            form = row[feats['form']]
            form_norm = Tokenization.normalize_token(form,'greek_glaux')
            pos = row[feats['pos']] + row[feats['person']] + row[feats['number']] + row[feats['tense']] + row[feats['mood']] + row[feats['voice']] + row[feats['gender']] + row[feats['case']] + row[feats['degree']]
            postags = lemmatizer.get(form_norm,{})
            lemma_freqs = postags.get(pos,{})
            lemma = row[feats['lemma']]
            freq = lemma_freqs.get(lemma,0)
            freq += 1
            lemma_freqs[lemma] = freq
            postags[pos] = lemma_freqs
            lemmatizer[form_norm] = postags
        self.lemmatizer = lemmatizer

    def lemmatize(self,form,pos):
        form_norm = Tokenization.normalize_token(form,'greek_glaux')
        lemma_freqs = self.lemmatizer.get(form_norm,{}).get(pos,{})
        if len(lemma_freqs) > 0:
            return max(lemma_freqs,key=lemma_freqs.get)
        else:
            return form_norm

    def load_lemmatizer(self,lemmatizer_file):
        with open(lemmatizer_file,'rb') as infile:
            self.lemmatizer = pickle.load(infile)