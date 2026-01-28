import unicodedata as ud
from nltk.corpus import wordnet as wn
import spacy
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from nltk.tokenize import MWETokenizer
import math
from vectors.VectorExtractor import VectorExtractor
import numpy as np
import gensim.downloader as api
from lightgbm import LGBMRanker
import pandas as pd
from string import punctuation
from alignment.Datasets import build_dataset
from alignment.Scorer import Scorer
from tqdm import tqdm
import random

def is_punct(token):
    return token in punctuation or token == '·' or token == ';' or token == '·'

def strip_accents(s):
    return ''.join(c for c in ud.normalize('NFD', s)
                  if ud.category(c) != 'Mn')

class WordAligner:
    
    def __init__(self, lexicon_file, vectors_name, large_corpus, training_data, gold_data, language_model, is_roberta=False,batch_size=100,keep_large_corpus=False):
        self.lexicon = self.build_lexicon(lexicon_file)
        self.mwes = self.build_mwes()
        self.mwe_tokenizer = MWETokenizer(self.mwes)
        self.nlp = spacy.load("en_core_web_trf")
        vectors = api.load(vectors_name)
        vectors_red = {}
        for k in vectors.key_to_index.keys():
            if k.startswith('/c/en/') or k.startswith('/c/grc/'):
                vectors_red[k] = vectors[k]
        self.vectors = vectors_red
        if is_roberta:
            self.extractor = VectorExtractor(transformer_path=language_model,tokenizer_add_prefix_space=True,layers=[8])
        else:
            self.extractor = VectorExtractor(transformer_path=language_model,tokenizer_add_prefix_space=False,layers=[8])
        self.sentences_large = build_dataset(**large_corpus,nlp=self.nlp)
        self.sentences_train = build_dataset(**training_data,nlp=self.nlp)
        self.sentences_gold = build_dataset(**gold_data,nlp=self.nlp)
        self.build_phrases(corpora=[self.sentences_large,self.sentences_train,self.sentences_gold])
        self.phrase_analyze(self.sentences_large)
        self.phrase_analyze(self.sentences_train)
        self.phrase_analyze(self.sentences_gold)
        self.total_sent_count = 0
        self.grc_lemmas_sent = {}
        self.en_lemmas_sent = {}
        self.grc_en_sent = {}
        self.get_sent_list_freqs(self.sentences_large)
        self.get_sent_list_freqs(self.sentences_train)
        self.get_sent_list_freqs(self.sentences_gold)
        self.grc_en_pmis = self.get_pmis()
        self.vectors_en_td, self.vectors_grc_td = self.get_bilingual_embeddings(self.sentences_train, batch_size=batch_size)
        self.vectors_en_gold, self.vectors_grc_gold = self.get_bilingual_embeddings(self.sentences_gold, batch_size=batch_size)
        self.vectors_cosines = {}
        self.lexicon_cosines = {}
        if not keep_large_corpus:
            self.sentences_large = None
    
    def build_lexicon(self,lexicon_file):
        lexicon = {}
        with open(lexicon_file,'r',encoding='utf8') as infile:
            lines = infile.readlines()
            for line in lines:
                sl = line.strip().split('\t')
                translations = sl[1].split('|')
                for n, translation in enumerate(translations):
                    translations[n] = translation.lower().replace(' ','_')
                lexicon[ud.normalize('NFD',sl[0])] = set(translations)
        return lexicon

    def build_mwes(self):
        mwes = list()
        wordnet_lemmas = set(i for i in wn.words())
        for word in wordnet_lemmas:
            if '_' in word:
                mwes.append(word.split('_'))
        for translations in self.lexicon.values():
            for translation in translations:
                if ' ' in translation:
                    mwes.append(translation.split(' '))
        return mwes

    def build_phrases(self,corpora):
        lemma_data = []
        for corpus in corpora:
            for sent in corpus:
                en_tokens = sent['en_tokens']
                lemmas = []
                for token in en_tokens:
                    lemmas.append(token.lemma_.lower())
                lemma_data.append(lemmas)
        connector_words = list(ENGLISH_CONNECTOR_WORDS)
        connector_words.append('-')
        connector_words.append("'")
        connector_words.append("’")
        connector_words = frozenset(connector_words)
        self.phrase_model = Phrases(lemma_data, connector_words=connector_words,scoring='npmi',threshold=0.2)
        self.phrase_model = self.phrase_model.freeze()

    def phrase_analyze(self,sentences):
        for sentence in tqdm(sentences, desc='Phrase analysis'):
            lemmas = []
            for word in sentence['en_tokens']:
                lemmas.append(word.lemma_.lower())
            mwe_tokenized = self.mwe_tokenizer.tokenize(lemmas)
            phrase_tokenized = self.phrase_model[lemmas]
            sentence['mwe_tokenized'] = mwe_tokenized
            sentence['phrase_tokenized'] = phrase_tokenized

    def get_sent_list_freqs(self,sentences):
        for sent_no, sent in tqdm(enumerate(sentences),desc='Counting frequencies',total=len(sentences)):
            self.total_sent_count += 1
            en_tokens = sent['en_tokens']
            mwe_tokenized = sent['mwe_tokenized']
            phrase_tokenized = sent['phrase_tokenized']
            grc_lemma_sent = sent['grc_lemmas']
            grc_lemmas = set(grc_lemma_sent)
            en_lemmas = set()
            for token in en_tokens:
                en_lemmas.add(token.lemma_.lower())
            for lemma in mwe_tokenized:
                if '_' in lemma:
                    en_lemmas.add(lemma)
            for lemma in phrase_tokenized:
                if '_' in lemma:
                    en_lemmas.add(lemma)
            self.add_counts(grc_lemmas,en_lemmas)

    def add_counts(self, grc_lemmas,en_lemmas):
        for grc_lemma in grc_lemmas:
            sent_count = 0
            if grc_lemma in self.grc_lemmas_sent:
                sent_count = self.grc_lemmas_sent[grc_lemma]
            sent_count += 1
            self.grc_lemmas_sent[grc_lemma] = sent_count
            en_with_grc_counts = dict()
            if grc_lemma in self.grc_en_sent:
                en_with_grc_counts = self.grc_en_sent[grc_lemma]
            for en_lemma in en_lemmas:
                en_count = 0
                if en_lemma in en_with_grc_counts:
                    en_count = en_with_grc_counts[en_lemma]
                en_count +=1
                en_with_grc_counts[en_lemma] = en_count
            self.grc_en_sent[grc_lemma] = en_with_grc_counts
        for en_lemma in en_lemmas:
            sent_count = 0
            if en_lemma in self.en_lemmas_sent:
                sent_count = self.en_lemmas_sent[en_lemma]
            sent_count += 1
            self.en_lemmas_sent[en_lemma] = sent_count

    def get_pmis(self,alpha=0.75):        
        grc_en_pmis = dict()
        for grc, ens in tqdm(self.grc_en_sent.items(),desc='Calculating pmis'):
            grc_count = self.grc_lemmas_sent[grc]
            en_pmis = dict()
            for en, observed in ens.items():
                expected = ((grc_count/self.total_sent_count) * pow(self.en_lemmas_sent[en]/self.total_sent_count,alpha))
                en_pmis[en] = math.log((observed/self.total_sent_count)/expected,2)
            grc_en_pmis[grc] = en_pmis
        return grc_en_pmis

    def get_bilingual_embeddings(self, sentences, batch_size=100):
        en_tokenized = []
        en_ids = []
        grc_tokenized = []
        grc_ids = []
        en_index = 0
        grc_index = 0
        for sent in sentences:
            en_split = []
            for token in sent['en_tokens']:
                en_split.append(token.text)
            grc_split = sent['grc']
            en_tokenized.append(en_split)
            grc_tokenized.append(grc_split)
            en_id = list(range(en_index,en_index+len(en_split)))
            en_id = [str(x) for x in en_id]
            sent['en_ids'] = en_id
            en_ids.append(en_id)
            grc_id = list(range(grc_index,grc_index+len(grc_split)))
            grc_id = [str(x) for x in grc_id]
            sent['grc_ids'] = grc_id
            grc_ids.append(grc_id)
            en_index += len(en_split)
            grc_index += len(grc_split)
        dataset = self.extractor.build_dataset(en_ids,en_tokenized,batched=True,batch_size=batch_size)
        vectors_en = self.extractor.extract_vectors(dataset)
        dataset = self.extractor.build_dataset(grc_ids,grc_tokenized,batched=True,batch_size=batch_size)
        vectors_grc = self.extractor.extract_vectors(dataset)
        vectors_en = {k: np.round(v,3).astype(np.float16) for k, v in vectors_en.items()}
        vectors_grc = {k: np.round(v,3).astype(np.float16) for k, v in vectors_grc.items()}
        return vectors_en, vectors_grc
    
    def get_positions(self,sentence):
        positions = {}
        count = -1
        for word_no, word in enumerate(sentence):
            if not is_punct(word):
                count+= 1
                positions[word_no] = count
            else:
                positions[word_no] = count
        sent_length = count + 1
        return positions, sent_length

    def get_cosine_lexicon(self,grc,en):
        if grc+'_'+en in self.lexicon_cosines:
            return self.lexicon_cosines[grc+'_'+en]
        best_cosine = 0
        en_vector_str = '/c/en/'+en
        if en_vector_str in self.vectors and grc in self.lexicon:
            entries = self.lexicon[grc]
            for entry in entries:
                entry_vector_str = '/c/en/'+entry
                if entry_vector_str in self.vectors:
                    cosine = np.dot(self.vectors[en_vector_str], self.vectors[entry_vector_str])/(np.linalg.norm(self.vectors[en_vector_str])* np.linalg.norm(self.vectors[entry_vector_str]))
                    if cosine > best_cosine:
                        best_cosine = cosine
        self.lexicon_cosines[grc+'_'+en] = best_cosine
        return best_cosine

    def get_en_pos_multiword(self,postags):
        scores = {'VERB':1,'PROPN':2,'NOUN':3,'INTJ':3,'PRON':4,'PTCP':5,'ADJ':5,'NUM':5,'ADV':6,'ADP':7,'SCONJ':7,'DET':8,'CCONJ':8,'X':8,'PART':8,'PUNCT':8,'SYM':8,'AUX':8,'':1000}
        best_postag = ''
        best_score = 1000
        for postag in postags:
            pos = ''
            if postag in scores:
                pos = postag
            else:
                print(postag)
            if scores[pos] < best_score:
                best_postag = pos
                best_score = scores[pos]
        return best_postag

    def get_group_size(self,data):
        return data.reset_index().groupby("GROUP")['GROUP'].count()
    
    def build_wa_dataset(self,sentences,vectors_grc,vectors_en,has_gold_alignment=True,use_phrase_model=True,add_phrase_candidates=True):
        group_rows = {}
        gold_alignments = {}
        gold_alignments_tokens = {}
        data = []
        row_index = -1
        for sent_no, sent in tqdm(enumerate(sentences),total=len(sentences)):
            candidates = []
            candidates_all = []
            if use_phrase_model:
                word_candidates, phrase_candidates = self.get_candidates(sent,self.mwes,self.phrase_model.phrasegrams,use_phrase_model)
                candidates.extend(word_candidates)
                if add_phrase_candidates:
                    candidates.extend(phrase_candidates)
                candidates_all.extend(word_candidates)
                candidates_all.extend(phrase_candidates)
            else:
                candidates = self.get_candidates(sent,None,None,use_phrase_model)
                candidates_all = candidates
            ignore_candidates = []
            if len(candidates) > 0:
                sent_ids_grc = sent['grc_ids']
                sent_ids_eng = sent['en_ids']
                grc_matrix = []
                en_matrix = []
                for wid in sent_ids_grc:
                    if wid in vectors_grc:
                        # Unfortunately, the UGARIT tokenizer sometimes makes splits that exceed the subword limit - these tokens will be ignored (see below)
                        grc_matrix.append(vectors_grc[wid])
                for candidate in candidates_all:
                    valid = True
                    vecs = []
                    for index in candidate:
                        # Also, very occasionally an English candidate needs to be ignored because of the UGARIT tokenizer
                        if sent_ids_eng[index] in vectors_en:
                            vecs.append(vectors_en[sent_ids_eng[index]])
                        else:
                            valid = False
                    if valid:
                        vecs = np.array(vecs)
                        vector = np.mean(vecs,axis=0)
                        en_matrix.append(vector)
                    else:
                        ignore_candidates.append(candidate)
                grc_matrix = np.array(grc_matrix)
                en_matrix = np.array(en_matrix)
                dotproducts = np.dot(grc_matrix, en_matrix.T)
                grc_norms = np.linalg.norm(grc_matrix, axis=1, keepdims=True)
                en_norms = np.linalg.norm(en_matrix, axis=1, keepdims=True)
                cosine_similarities = dotproducts / (grc_norms @ en_norms.T)
                cosine_index = []
                for candidate in candidates_all:
                    if not candidate in ignore_candidates:
                        cosine_index.append(candidate)
                grc_lemmas = sent['grc_lemmas']
                grc_tags = None
                if 'grc_pos' in sent:
                    grc_tags = sent['grc_pos']
                grc_words = sent['grc']
                en_tokens = sent['en_tokens']
                alignments = None
                if has_gold_alignment:
                    alignments = sent['alignment']
                grc_positions, grc_length = self.get_positions(grc_words)
                en_words = []
                for word in en_tokens:
                    en_words.append(word.text)
                en_positions, en_length = self.get_positions(en_words)
                for grc_no, grc_token in enumerate(grc_words):
                    grc_id = int(sent_ids_grc[grc_no])
                    if not is_punct(grc_token) and str(grc_id) in vectors_grc:
                        # The word is ignored if the sentence exceeds the subword limit (see also above)
                        grc_lemma = grc_lemmas[grc_no]
                        grc_pos = None
                        if grc_tags is not None:
                            grc_pos = grc_tags[grc_no]
                        if alignments is not None:
                            alignment = alignments.get(grc_no,[])
                        pmis = self.grc_en_pmis.get(grc_lemma,{})
                        grc_en_freqs = self.grc_en_sent.get(grc_lemma,{})
                        grc_freq = self.grc_lemmas_sent.get(grc_lemma,0)
                        #grc_loc = grc_no / (len(grc_words)-1)
                        grc_loc = 0
                        if grc_length > 1:
                            grc_loc = grc_positions[grc_no] / (grc_length-1)
                        if has_gold_alignment:
                            gold_alignments[grc_id] = alignment
                            gold_alignment_tokens = []
                            for index in alignment:
                                gold_alignment_tokens.append(en_tokens[index].text)
                            gold_alignments_tokens[grc_id] = ' '.join(gold_alignment_tokens)
                        for candidate in candidates:
                            # Again, ignored because of tokenizer limits
                            if not candidate in ignore_candidates:
            #                    en_loc = ((candidate[0]+candidate[len(candidate)-1])/2) / (len(en_tokens)-1)
                                en_loc = 0
                                if en_length > 1:
                                    en_loc = ((en_positions[candidate[0]] + en_positions[candidate[len(candidate)-1]]) / 2) / (en_length-1)
                                pos_diff = abs(grc_loc-en_loc)
                                candidate_en_tokens = []
                                candidate_en_lemmas = []
                                candidate_en_postags = []
                                for index in candidate:
                                    candidate_en_tokens.append(en_tokens[index].text)
                                    candidate_en_lemmas.append(en_tokens[index].lemma_.lower())
                                    candidate_en_postags.append(en_tokens[index].pos_)
                                if len(candidate_en_postags) == 1:
                                    en_postag = candidate_en_postags[0]
                                else:
                                    en_postag = self.get_en_pos_multiword(candidate_en_postags)
                                candidate_en_lemmas_str = '_'.join(candidate_en_lemmas)
                                if has_gold_alignment:
                                    match = 0
                                    if add_phrase_candidates:
                                        if candidate == alignment:
                                            match = 1
                                    else:
                                        if candidate[0] in alignment:
                                            match = 1
                                cosine = cosine_similarities[grc_no][cosine_index.index(candidate)]
                                pmi = pmis.get(candidate_en_lemmas_str,None)
                                if use_phrase_model and not add_phrase_candidates and pmi is not None:
                                    for candidate in phrase_candidates:
                                        phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                        phrase_pmi = pmis.get(phrase_candidate_en_lemmas_str,0)
                                        if phrase_pmi > pmi:
                                            pmi = phrase_pmi
                                rel_freq = 0
                                if grc_freq > 0:
                                    rel_freq = grc_en_freqs.get(candidate_en_lemmas_str,0) / grc_freq
                                    if use_phrase_model and not add_phrase_candidates:
                                        for candidate in phrase_candidates:
                                            phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                            phrase_rel_freq = grc_en_freqs.get(phrase_candidate_en_lemmas_str,0) / grc_freq
                                            if phrase_rel_freq > rel_freq:
                                                rel_freq = phrase_rel_freq
                                rel_freq_en = 0
                                en_freq = self.en_lemmas_sent.get(candidate_en_lemmas_str,0)
                                if en_freq > 0:
                                    rel_freq_en = grc_en_freqs.get(candidate_en_lemmas_str,0) / en_freq
                                    if use_phrase_model and not add_phrase_candidates:
                                        for candidate in phrase_candidates:
                                            phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                            phrase_rel_freq = grc_en_freqs.get(phrase_candidate_en_lemmas_str,0) / en_freq
                                            if phrase_rel_freq > rel_freq_en:
                                                rel_freq_en = phrase_rel_freq
                                grc_stripped = (strip_accents(grc_lemma).replace('ς','σ')).lower()
                                cosine_static = None
                                if grc_stripped + '_' + candidate_en_lemmas_str.replace('_-','') in self.vectors_cosines:
                                    cosine_static = self.vectors_cosines[grc_stripped + '_' + candidate_en_lemmas_str.replace('_-','')]
                                elif '/c/grc/'+grc_stripped in self.vectors and '/c/en/'+candidate_en_lemmas_str.replace('_-','') in self.vectors:
                                    cosine_static = np.dot(self.vectors['/c/grc/'+grc_stripped], self.vectors['/c/en/'+candidate_en_lemmas_str.replace('_-','')])/(np.linalg.norm(self.vectors['/c/grc/'+grc_stripped])* np.linalg.norm(self.vectors['/c/en/'+candidate_en_lemmas_str.replace('_-','')]))
                                    self.vectors_cosines[grc_stripped + '_' + candidate_en_lemmas_str.replace('_-','')] = cosine_static
                                if cosine_static is not None and use_phrase_model and not add_phrase_candidates:
                                    for candidate in phrase_candidates:
                                        phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                        if grc_stripped + '_' + phrase_candidate_en_lemmas_str.replace('_-','') in self.vectors_cosines:
                                            phrase_cosine_static = self.vectors_cosines[grc_stripped + '_' + phrase_candidate_en_lemmas_str.replace('_-','')]
                                        elif '/c/grc/'+grc_stripped in self.vectors and '/c/en/'+phrase_candidate_en_lemmas_str.replace('_-','') in self.vectors:
                                            phrase_cosine_static = np.dot(self.vectors['/c/grc/'+grc_stripped], self.vectors['/c/en/'+phrase_candidate_en_lemmas_str.replace('_-','')])/(np.linalg.norm(self.vectors['/c/grc/'+grc_stripped])* np.linalg.norm(self.vectors['/c/en/'+phrase_candidate_en_lemmas_str.replace('_-','')]))
                                            self.vectors_cosines[grc_stripped + '_' + phrase_candidate_en_lemmas_str.replace('_-','')] = phrase_cosine_static
                                            if phrase_cosine_static > cosine_static:
                                                cosine_static = phrase_cosine_static
                                in_lexicon = False
                                if grc_lemma in self.lexicon:
                                    in_lexicon = candidate_en_lemmas_str.replace('_-','') in self.lexicon[grc_lemma]
                                    if not in_lexicon and use_phrase_model and not add_phrase_candidates:
                                        for candidate in phrase_candidates:
                                            phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                            phrase_in_lexicon = phrase_candidate_en_lemmas_str.replace('_-','') in self.lexicon[grc_lemma]
                                            if phrase_in_lexicon:
                                                in_lexicon = True
                                                break
                                cosine_lexicon = self.get_cosine_lexicon(grc_lemma,candidate_en_lemmas_str.replace('_-',''))
                                if use_phrase_model and not add_phrase_candidates:
                                    for candidate in phrase_candidates:
                                        phrase_candidate_en_lemmas_str = '_'.join([en_tokens[index].lemma_.lower() for index in candidate])
                                        phrase_cosine_lexicon = self.get_cosine_lexicon(grc_lemma,phrase_candidate_en_lemmas_str.replace('_-',''))
                                        if phrase_cosine_lexicon > cosine_lexicon:
                                            cosine_lexicon = phrase_cosine_lexicon
                                is_phrase = len(candidate) > 1
                                if has_gold_alignment:
                                    if add_phrase_candidates:
                                        data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,match,cosine,pmi,rel_freq,pos_diff,in_lexicon,cosine_static,cosine_lexicon,is_phrase])
                                    else:
                                        data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,match,cosine,pmi,rel_freq,pos_diff,in_lexicon,cosine_static,cosine_lexicon])
    #                                data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,match,cosine,pmi,rel_freq,rel_freq_en,pos_diff,in_lexicon,cosine_static,cosine_lexicon,is_phrase])
                                else:
                                    if add_phrase_candidates:
                                        data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,cosine,pmi,rel_freq,pos_diff,in_lexicon,cosine_static,cosine_lexicon,is_phrase])
                                    else:
                                        data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,match,cosine,pmi,rel_freq,pos_diff,in_lexicon,cosine_static,cosine_lexicon])
    #                                data.append([grc_id,sent_no,grc_token,' '.join(candidate_en_tokens),grc_no,candidate,cosine,pmi,rel_freq,rel_freq_en,pos_diff,in_lexicon,cosine_static,cosine_lexicon,is_phrase])
                                row_index += 1
                                if grc_id in group_rows:
                                    group_rows[grc_id][1] = row_index
                                else:
                                    group_rows[grc_id] = [row_index,row_index]
            #                    data.append([group_index,grc_token,' '.join(candidate_en_tokens),candidate,match,cosine,pmi,rel_freq,pos_diff,in_lexicon,cosine_static,cosine_lexicon,is_phrase,grc_pos,en_postag])
        if has_gold_alignment:
            return data, group_rows, gold_alignments, gold_alignments_tokens
        else:
            return data, group_rows
        
    def train(self,evaluate=True,ignore_features=None,build_datasets=False):
        if (not hasattr(self,'training_data')) or build_datasets:
            self.training_data, self.td_group_rows, self.td_alignments, self.td_alignments_tokens = self.build_wa_dataset(self.sentences_train,self.vectors_grc_td,self.vectors_en_td,use_phrase_model=True,add_phrase_candidates=True)
        train = pd.DataFrame(self.training_data,columns=['GROUP','SENT','GRC','EN','GRC_INDEX','EN_INDICES','ALIGNED','COSINE','PMI','REL_FREQ','POS_DIFF','IN_LEXICON','COSINE_STATIC','COSINE_LEXICON','IS_PHRASE'])
        if ignore_features is not None:
            train.drop(columns=ignore_features,inplace=True)
        self.model = self.train_model(train,False,seed=12345,n_estimators=300)
        if evaluate:
            if (not hasattr(self,'gold_data')) or build_datasets:
                self.gold_data, self.gold_group_rows, self.gold_alignments, self.gold_alignments_tokens = self.build_wa_dataset(self.sentences_gold,self.vectors_grc_gold,self.vectors_en_gold,use_phrase_model=True,add_phrase_candidates=True)
            gold = pd.DataFrame(self.gold_data,columns=['GROUP','SENT','GRC','EN','GRC_INDEX','EN_INDICES','ALIGNED','COSINE','PMI','REL_FREQ','POS_DIFF','IN_LEXICON','COSINE_STATIC','COSINE_LEXICON','IS_PHRASE'])
            if ignore_features is not None:
                gold.drop(columns=ignore_features,inplace=True)
            predictions = np.array(self.model.predict(gold.copy().drop(columns=['GROUP','SENT','GRC','EN','GRC_INDEX','EN_INDICES','ALIGNED'])))
            results = []
            groups = np.array(gold['GROUP'])
            sentnos = np.array(gold['SENT'])
            grc = np.array(gold['GRC'])
            en = np.array(gold['EN'])
            grc_indices = np.array(gold['GRC_INDEX'])
            alignments = np.array(gold['EN_INDICES'])
            aligned = np.array(gold['ALIGNED'])
            group_nos = list(gold['GROUP'].unique())
            for group in group_nos:
                indices = np.where(groups == group)[0]
                candidates = predictions[indices[0]:indices[len(indices)-1]+1]
                candidates_sents = sentnos[indices[0]:indices[len(indices)-1]+1]
                candidates_grc_indices = grc_indices[indices[0]:indices[len(indices)-1]+1]
                candidates_alignments = alignments[indices[0]:indices[len(indices)-1]+1]
                candidates_aligned = aligned[indices[0]:indices[len(indices)-1]+1]
                candidates_grc = grc[indices[0]:indices[len(indices)-1]+1]
                candidates_en = en[indices[0]:indices[len(indices)-1]+1]
                best_candidate = np.argmax(candidates)
                gold_alignment = self.gold_alignments[group]
                gold_alignment_tokens = self.gold_alignments_tokens[group]
                no_alignment = len(gold_alignment) == 0
                results.append([group,candidates_sents[best_candidate],candidates[best_candidate],candidates_grc_indices[best_candidate],candidates_alignments[best_candidate],candidates_aligned[best_candidate],candidates_grc[best_candidate],candidates_en[best_candidate],no_alignment,gold_alignment,gold_alignment_tokens])
            self.results = results
            self.results_df = pd.DataFrame(results,columns=['GROUP','SENT','SCORE','GRC_INDEX','BEST_CANDIDATE','CORRECT','GREEK','EN','UNALIGNED','GOLD','GOLD_TOKENS'])
            scores_unaligned = sorted(list(self.results_df['SCORE'][self.results_df['UNALIGNED']==True]),reverse=True)
            best_threshold = None
            best_accuracy = 0
            best_correct = 0
            scorer = Scorer(self.results,self.sentences_gold)
            for threshold in scores_unaligned:
                accuracy, correct, total = scorer.get_accuracy(threshold)
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    best_threshold = threshold
                    best_correct = correct
            self.threshold = best_threshold
            match1 = (self.results_df['SCORE']<self.threshold)&(self.results_df['UNALIGNED']==True)
            match2 = (self.results_df['SCORE']<self.threshold)&(self.results_df['UNALIGNED']==False)
            self.results_df.loc[match1, 'CORRECT'] = 1
            self.results_df.loc[match2, 'CORRECT'] = 0
            print(f'Accuracy: {best_accuracy}')
            print(f'N correct: {best_correct}')
            print(f'N total: {total}')
            print(f'Best threshold: {best_threshold}')
            precision, recall, f1, aer = scorer.get_standard_metrics(best_threshold)
            print(f'Precision: {precision}')
            print(f'Recall: {recall}')
            print(f'F1: {f1}')
            print(f'AER: {aer}')
        
    def train_model(self,data,train_test_split,train_size=0.9,seed=None,n_estimators=500):
        groups = list(set(data['GROUP']))
        if train_test_split:
            if seed is not None:
                random.seed(seed)
            random.shuffle(groups)
            train_no = round(train_size*len(groups))
            train_indices = groups[0:train_no]
            test_indices = groups[train_no:len(groups)]
            train_instances = data[data['GROUP'].isin(train_indices)].copy()
            test_instances = data[data['GROUP'].isin(test_indices)].copy()
            train_groups = self.get_group_size(train_instances)
        else:
            train_instances = data.copy()
            train_groups = self.get_group_size(data)
        model = LGBMRanker(objective="lambdarank",n_estimators=n_estimators,random_state=seed)
        model.fit(train_instances.drop(columns=['GROUP','SENT','GRC','EN','GRC_INDEX','EN_INDICES','ALIGNED']),train_instances['ALIGNED'],group=train_groups)
        if train_test_split:
            test_instances = self.make_predictions(model,test_instances)
            return model, test_instances
        else:
            return model

    def get_candidates(self,sentence,mwes,phrases,use_phrase_model=True):
        word_candidates = []
        phrase_candidates = []
        for token_no, token in enumerate(sentence['en_tokens']):
            if not is_punct(token.text):
                word_candidates.append([token_no])
        if use_phrase_model:
            i = -1
            for token in sentence['mwe_tokenized']:
                i += 1
                n_words = token.count('_') + 1
                if token == '_':
                    n_words = 1
                if n_words > 1:
                    indices = list(range(i,i+n_words))
                    phrase_candidates.append(indices)
                    i += (n_words-1)
            i = -1
            for token in sentence['phrase_tokenized']:
                i+=1
                n_words = token.count('_') + 1
                if token == '_':
                    n_words = 1
                if n_words > 1:
                    indices = list(range(i,i+n_words))
                    if not indices in phrase_candidates:
                        phrase_candidates.append(indices)
                    i += (n_words-1)
            token_list = list(sentence['en_tokens'])
            for token_no, token in enumerate(sentence['en_tokens']):
                if (token.dep_ == 'advmod' or token.dep_ == 'prt') and (token.head.pos_ == 'VERB' or token.head.pos_ == 'AUX') and token_list.index(token.head) < token_list.index(token):
                    expr = []
                    expr.append(token.head.lemma_.lower())
                    expr.append(token.lemma_.lower())
                    if expr in mwes or '_'.join(expr) in phrases.keys():
                        indices = []
                        indices.append(token_list.index(token.head))
                        indices.append(token_list.index(token))
                        if not indices in phrase_candidates:
                            phrase_candidates.append(indices)
                elif token.dep_ == 'acomp' and token.head.pos_ == 'AUX':
                    expr = []
                    expr.append(token.head.lemma_.lower())
                    expr.append(token.lemma_.lower())
                    if expr in mwes or '_'.join(expr) in phrases.keys():
                        indices = []
                        indices.append(token_list.index(token.head))
                        indices.append(token_list.index(token))
                        indices.sort()
                        if not indices in phrase_candidates:
                            phrase_candidates.append(indices)
                elif token.dep_ == 'pobj' and token.head.dep_ == 'prep' and (token.head.head.pos_ == 'VERB' or token.head.head.pos_ == 'AUX'):
                    expr = []
                    expr.append(token.head.head.lemma_.lower())
                    expr.append(token.head.lemma_.lower())
                    expr.append(token.lemma_.lower())
                    if expr in mwes or '_'.join(expr) in phrases.keys():
                        indices = []
                        indices.append(token_list.index(token.head.head))
                        indices.append(token_list.index(token.head))
                        indices.append(token_list.index(token))
                        indices.sort()
                        if not indices in phrase_candidates:
                            phrase_candidates.append(indices)
                elif token.dep_ == 'oprd' and (token.head.pos_ == 'VERB' or token.head.pos_ == 'AUX'):
                    expr = []
                    expr.append(token.head.lemma_.lower())
                    expr.append(token.lemma_.lower())
                    if expr in mwes or '_'.join(expr) in phrases.keys():
                        indices = []
                        indices.append(token_list.index(token.head))
                        indices.append(token_list.index(token))
                        indices.sort()
                        if not indices in phrase_candidates:
                            phrase_candidates.append(indices)
                elif (token.dep_ == 'dobj' or token.dep_ == 'ccomp' or token.dep_ == 'nsubjpass') and (token.head.pos_ == 'VERB'):
                    expr = []
                    expr.append(token.head.lemma_.lower())
                    expr.append(token.lemma_.lower())
                    if expr in mwes or '_'.join(expr) in phrases.keys():
                        indices = []
                        indices.append(token_list.index(token.head))
                        indices.append(token_list.index(token))
                        indices.sort()
                        if not indices in phrase_candidates:
        #                    print(f'{expr}\t{sentence['eng']}')
                            phrase_candidates.append(indices)
        if use_phrase_model:
            return word_candidates, phrase_candidates
        else:
            return word_candidates