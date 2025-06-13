from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import stopwords
from tqdm import tqdm
import re
from spacy.tokens import Doc
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
import unicodedata as ud
import math
import string
import gensim.downloader as api
import numpy
from tokenization.Tokenization import strip_accents
import pandas as pd
import random
from lightgbm import LGBMRanker

class WordAligner:
    
    def __init__(self, lexicon_file, vectors_name):
        self.lexicon = self.build_lexicon(lexicon_file)
        self.mwe_tokenizer = MWETokenizer(self.build_mwes())
        self.stop_words = self.build_stopwords()
        self.total_sent_count = 0
        self.grc_lemmas_sent = {}
        self.en_lemmas_sent = {}
        self.grc_en_sent = {}
        self.vectors = api.load(vectors_name)
        self.lexicon_cosines = {}
        self.vectors_cosines = {}
    
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
    
    def build_stopwords(self):
        stop_words = set(stopwords.words('english'))
        stop_words.add('may')
        stop_words.add('might')
        stop_words.add('shall')
        stop_words.add('would')
        stop_words.add('must')
        stop_words.add('also')
        stop_words.add('yet')
        stop_words.add('could')
        stop_words.add('let')
        stop_words.add('us')
        stop_words.add('begin')
        return stop_words
    
    def analyze_eng_sentences(self, sentences, nlp, mwe_tokenizer=None, phrase_model=None, phrase_analyze=True, is_tokenized=False, strip_punctuation=False, linguistic_analysis=True):
        documents = []
        sentence_ids = []
        if linguistic_analysis:
            for id, sentence in tqdm(sentences.items(),desc='Tokenizing sentences'):
                sentence_ids.append(id)
                sent = sentence['en_sent']
                if not is_tokenized:
                    tokenized = word_tokenize(sent)
                else:
                    tokenized = sent
                tokenized_cleaned = []
                for word in tokenized:
                    if strip_punctuation:
                        word = word.replace('.','')
                        word = word.replace('\'','')
                    if '—' in word or '-' in word:
                        word_s = re.split('([—-])',word)
                        for s in word_s:
                            if (not ((s == '—' or s == '-') and strip_punctuation)) and s != '':
                                tokenized_cleaned.append(s)
                    else:
                        tokenized_cleaned.append(word)
                if strip_punctuation:
                    tokenized_cleaned = [str for str in tokenized_cleaned if str not in string.punctuation and str != '``' and str!='\'\'' and str !='“' and str!='”' and str!='’' and str!='‘' and str != '...' and str!='—']
                sentence['tokenized'] = tokenized_cleaned
                documents.append(Doc(nlp.vocab, words=tokenized_cleaned))
            analysis = tqdm(nlp.pipe(documents), total=len(documents), desc='Analyzing sentences')
            for index, doc in enumerate(analysis):
                pos_tagged_sent = []
                lemmatized_sent = []
                for word in doc:
                    pos = word.pos_
                    if word.tag_ == 'VBG' or word.tag_ == 'VBN':
                        pos = 'PTCP'
                    pos_tagged_sent.append(pos)
                    if word.lemma_ is None:
                        word.lemma_ = word.text
                    lemmatized_sent.append(word.lemma_.lower())
                sentences[sentence_ids[index]]['pos_tagged'] = pos_tagged_sent
                sentences[sentence_ids[index]]['lemmatized'] = lemmatized_sent
        if phrase_analyze:
            for sentence in tqdm(sentences.values(), desc='Phrase analysis'):
                mwe_tokenized = mwe_tokenizer.tokenize(sentence['lemmatized'])
                phrase_tokenized = phrase_model[sentence['lemmatized']]
                sentence['mwe_tokenized'] = mwe_tokenized
                sentence['phrase_tokenized'] = phrase_tokenized
        
    def get_lemma_phrases(self,sentences):
        lemma_data = []
        for sent in sentences.values():
            lemmas = sent['lemmatized']
            lemma_data.append(lemmas)
        phrase_model = Phrases(lemma_data, connector_words=ENGLISH_CONNECTOR_WORDS,scoring='npmi',threshold=0.2)
        phrase_model = phrase_model.freeze()
        return phrase_model
    
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
    
    def get_database_freqs(self,rows,en_sentences):
        currentSent = ''
        en_lemmas = set()
        grc_lemmas = set()
        for row in tqdm(rows,desc='Counting frequencies'):
            sent = row['sentence_id']
            if currentSent == sent:
                grc_lemmas.add(row['lemma_string'])
            else:
                self.total_sent_count += 1
                self.add_counts(grc_lemmas,en_lemmas)
                en_lemmas.clear()
                grc_lemmas.clear()
                en_sent = en_sentences[sent]
                lemmatized_sentence = en_sent['lemmatized']
                mwe_tokenized = en_sent['mwe_tokenized']
                phrase_tokenized = en_sent['phrase_tokenized']
                for word in lemmatized_sentence:
                    en_lemmas.add(word)
                for word in mwe_tokenized:
                    if '_' in word:
                        en_lemmas.add(word)
                for word in phrase_tokenized:
                    if '_' in word:
                        en_lemmas.add(word)
            currentSent = sent
        self.add_counts(grc_lemmas,en_lemmas)
    
    def get_sent_list_freqs(self,lemmas,sentences):
        for sent_no, sent in tqdm(enumerate(sentences),desc='Counting frequencies',total=len(sentences)):
            lemmatized_sentence = sent['lemmatized']
            mwe_tokenized = sent['mwe_tokenized']
            phrase_tokenized = sent['phrase_tokenized']
            grc_lemma_sent = lemmas[sent_no]
            grc_lemmas = set(grc_lemma_sent)
            en_lemmas = set(lemmatized_sentence)
            for lemma in mwe_tokenized:
                if '_' in lemma:
                    en_lemmas.add(lemma)
            for lemma in phrase_tokenized:
                if '_' in lemma:
                    en_lemmas.add(lemma)
            self.add_counts(grc_lemmas,en_lemmas)
    
    def get_file_freqs(self,file,greek_lemma_list,sentences):
        lemma_count = -1
        sent_id = 0
        with open(file,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines,desc='Counting frequencies'):
                sl = line.strip().split('\t')
                if len(sl)==3:
                    sent_id += 1
                    self.total_sent_count +=1
                    grc = sl[0].split(' ')
                    sent = sentences[sent_id]
                    lemmatized_sentence = sent['lemmatized']
                    mwe_tokenized = sent['mwe_tokenized']
                    phrase_tokenized = sent['phrase_tokenized']
                    grc_lemmas = set()
                    for word in grc:
                        lemma_count +=1
                        grc_lemmas.add(greek_lemma_list[lemma_count])
                    en_lemmas = set()
                    for lemma in lemmatized_sentence:
                        en_lemmas.add(lemma)
                    for lemma in mwe_tokenized:
                        if '_' in lemma:
                            en_lemmas.add(lemma)
                    for lemma in phrase_tokenized:
                        if '_' in lemma:
                            en_lemmas.add(lemma)
                    self.add_counts(grc_lemmas,en_lemmas)
    
    def get_pmis(self):        
        grc_en_pmis = dict()
        for grc, ens in tqdm(self.grc_en_sent.items(),desc='Calculating pmis'):
            grc_count = self.grc_lemmas_sent[grc]
            en_pmis = dict()
            for en, observed in ens.items():
                expected = ((grc_count/self.total_sent_count) * pow(self.en_lemmas_sent[en]/self.total_sent_count,0.75))
                en_pmis[en] = math.log((observed/self.total_sent_count)/expected,2)
                #expected = (grc_count * en_lemmas_sent[en])/total_sent_count
                #en_pmis[en] = math.log(observed/expected,2)
            grc_en_pmis[grc] = en_pmis
        return grc_en_pmis
    
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
    
    def get_phrase_indices(self,mwe_tokenized,phrase_tokenized):
        phrase_indices = []
        i = -1
        for word in mwe_tokenized:
            i+=1
            n_words = word.count('_') + 1
            if n_words > 1:
                indices = list(range(i,i+n_words))
                phrase_indices.append(indices)
                i += (n_words-1)
        i = -1
        for word in phrase_tokenized:
            i+=1
            n_words = word.count('_') + 1
            if n_words > 1:
                indices = list(range(i,i+n_words))
                if not indices in phrase_indices:
                    phrase_indices.append(indices)
                i += (n_words-1)
        return phrase_indices
    
    def realign_indices(self,simple_tokenized):
        indices_map = {}
        current_index = -1
        for old_index, word in enumerate(simple_tokenized):
            current_index +=1
            indices = [current_index]
            if '’s' in word or '’t' in word or word == 'cannot':
                current_index +=1
                indices.append(current_index)
            elif '—' in word or '-' in word:
                word_s = re.split('([—-])',word)
                for s in word_s[1:len(word_s)]:
                    if not (s == '—' or s == '-' or s==''):
                        current_index +=1
                        indices.append(current_index)
            elif word == '':
                current_index-=1
            indices_map[old_index] = indices
        return indices_map
    
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
                    cosine = numpy.dot(self.vectors[en_vector_str], self.vectors[entry_vector_str])/(numpy.linalg.norm(self.vectors[en_vector_str])* numpy.linalg.norm(self.vectors[entry_vector_str]))
                    if cosine > best_cosine:
                        best_cosine = cosine
        self.lexicon_cosines[grc+'_'+en] = best_cosine
        return best_cosine
    
    def sentence_alignment_test_data(self,sentence,grc_sent,grc_lemmas,grc_postags):
        prediction_data = []
        # Note: make sure that punctuation etc is already stripped from the Greek sentence, otherwise positions won't match
        group_index = 0
        group_indices = []
        phrase_indices = self.get_phrase_indices(sentence['mwe_tokenized'],sentence['phrase_tokenized'])
        possibilities = []
        for en_index, en_word in enumerate(sentence['lemmatized']):
            possibilities.append([en_index])
        possibilities.extend(phrase_indices)
        for index, grc_lemma in enumerate(grc_lemmas):
            if len(grc_lemmas)>1:
                grc_loc = index / (len(grc_lemmas)-1)
            else:
                grc_loc = 0
            pmis = {}
            if grc_lemma in self.grc_en_pmis:
                pmis = self.grc_en_pmis[grc_lemma]
            en_freqs = {}
            if grc_lemma in self.grc_en_sent:
                en_freqs = self.grc_en_sent[grc_lemma]
            grc_freq = 0
            if grc_lemma in self.grc_lemmas_sent:
                grc_freq = self.grc_lemmas_sent[grc_lemma]
            group_index += 1
            group_indices.append(group_index)
            grc_postag = grc_postags[index]
            for possibility in possibilities:
                if len(sentence['tokenized'])>1:
                    en_loc = ((possibility[0] + possibility[len(possibility)-1]) / 2) / (len(sentence['tokenized'])-1)
                else:
                    en_loc = 0
                pos_diff = abs(grc_loc-en_loc)
                en_words = []
                en_words_original = []
                en_postags = []
                for index in possibility:
                    en_words.append(sentence['lemmatized'][index])
                    if len(sentence['lemmatized']) != len(sentence['tokenized']):
                        print(sentence['lemmatized'])
                        print(sentence['tokenized'])
                    en_words_original.append(sentence['tokenized'][index])
                    en_postags.append(sentence['pos_tagged'][index])
                en_words_str = '_'.join(en_words)
                en_postag = ''
                if len(en_postags) == 1:
                    en_postag = en_postags[0]
                else:
                    en_postag = self.get_en_pos_multiword(en_postags)
                pmi = 0
                if en_words_str in pmis:
                    pmi = pmis[en_words_str]
                rel_freq = 0.0
                if en_words_str in en_freqs:
                    rel_freq = en_freqs[en_words_str] / grc_freq
                in_lexicon = False
                if grc_lemma in self.lexicon:
                    in_lexicon = en_words_str in self.lexicon[grc_lemma]
                cosine = 0
                grc_stripped = (strip_accents(grc_lemma).replace('ς','σ')).lower()
                if grc_stripped + '_' + en_words_str in self.vectors_cosines:
                    cosine = self.vectors_cosines[grc_stripped + '_' + en_words_str]
                elif '/c/grc/'+grc_stripped in self.vectors and '/c/en/'+en_words_str in self.vectors:
                    cosine = numpy.dot(self.vectors['/c/grc/'+grc_stripped], self.vectors['/c/en/'+en_words_str])/(numpy.linalg.norm(self.vectors['/c/grc/'+grc_stripped])* numpy.linalg.norm(self.vectors['/c/en/'+en_words_str]))
                self.vectors_cosines[grc_stripped + '_' + en_words_str] = cosine
                cosine_lexicon = self.get_cosine_lexicon(grc_lemma,en_words_str)
                is_phrase = len(en_words) > 1
                if not en_postag == ':' and not en_postag=='(' and not en_postag==')':
                    prediction_data.append([group_index,grc_lemma,' '.join(en_words_original),' '.join(en_words),','.join(map(str,possibility)),pmi,rel_freq,pos_diff,in_lexicon,cosine,grc_postag,en_postag,cosine_lexicon,is_phrase])
        prediction_data_pd = pd.DataFrame(prediction_data,columns=['GROUP','GRC','EN','EN_LEMMA','INDICES','PMI','RELFREQ','POSITION','LEXICON','COSINE','GRCPOS','ENPOS','COSINE_LEXICON','IS_PHRASE'])
        prediction_data_pd = prediction_data_pd.astype({"GRCPOS": "category", "ENPOS": "category"})
        return prediction_data_pd, group_indices
    
    def alignment_data_from_file(self,file,sentences,grc_lemmas_list,grc_pos_list):
        sent_no = 0
        dataset = []
        groups = []
        lemma_index = -1
        group_index = 0
        with open(file,'r',encoding='utf8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines,desc='Creating alignment dataset'):
                sl = line.strip().split('\t')
                if len(sl)==3:
                    sent_no += 1
                    grc = sl[0].split(' ')
                    en_sent = sentences[sent_no]
                    tokenized_simple = en_sent['en_sent'].split(' ')
                    for index, word in enumerate(tokenized_simple):
                        tokenized_simple[index] = re.sub(r'[\.,“!”‘:\?;\(\)]','',word)
                    indices_new = self.realign_indices(tokenized_simple)
                    tokenized = en_sent['tokenized']
                    mwe_tokenized = en_sent['mwe_tokenized']
                    phrase_tokenized = en_sent['phrase_tokenized']
                    lemmatized_sentence = en_sent['lemmatized']
                    pos_tagged = en_sent['pos_tagged']
                    alignments = sl[2].split(' ')
                    grc_en = dict()
                    for alignment in alignments:
                        al_s = alignment.replace('P','').split('-')
                        grc_index = int(al_s[0])
                        en_index = int(al_s[1])
                        en_indices_realigned = indices_new[en_index]
                        en_indices = []
                        if grc_index in grc_en:
                            en_indices = grc_en[grc_index]
                        en_indices.extend(en_indices_realigned)
                        grc_en[grc_index] = en_indices
                    phrase_indices = self.get_phrase_indices(mwe_tokenized,phrase_tokenized)
                    grc_lemmas = []
                    grc_pos = []
                    for grc_index, word in enumerate(grc):
                        lemma_index +=1
                        grc_lemmas.append(grc_lemmas_list[lemma_index])
                        grc_pos.append(grc_pos_list[lemma_index])
                    for grc_index, word in enumerate(grc):
                        if grc_index in grc_en and grc_lemmas[grc_index] in self.grc_lemmas_sent:
                            en_indices = grc_en[grc_index]
                            valid = True
                            if len(en_indices) > 1:
                                if not en_indices in phrase_indices:
                                    stopwords = []
                                    i = -1
                                    while(True):
                                        i+=1
                                        if i==len(en_indices):
                                            break
                                        elif not lemmatized_sentence[en_indices[i]] in self.stop_words:
                                            break
                                        else:
                                            stopwords.append(en_indices[i])
                                    i = len(en_indices)
                                    while(True):
                                        i-=1
                                        if i==-1:
                                            break
                                        elif not lemmatized_sentence[en_indices[i]] in self.stop_words:
                                            break
                                        else:
                                            stopwords.append(en_indices[i])
                                    en_no_stopwords = [i for i in en_indices if not i in stopwords]
                                    if len(en_no_stopwords) == 1 or en_no_stopwords in phrase_indices:
                                        en_indices = en_no_stopwords
                                    else:
                                        valid = False
                            if valid:
                                group_index += 1
                                groups.append(group_index)
                                grc_lemma = grc_lemmas[grc_index]
                                grc_postag = grc_pos[grc_index]
                                grc_loc = grc_index / (len(grc)-1)
                                pmis = self.grc_en_pmis[grc_lemma]
                                en_freqs = self.grc_en_sent[grc_lemma]
                                grc_freq = self.grc_lemmas_sent[grc_lemma]
                                possibilities = []
                                for en_index, en_word in enumerate(tokenized):
                                    possibilities.append([en_index])
                                possibilities.extend(phrase_indices)
                                for possibility in possibilities:
                                    aligned = 0
                                    if possibility == en_indices:
                                        aligned = 1
                                    en_loc = ((possibility[0] + possibility[len(possibility)-1]) / 2) / (len(tokenized)-1)
                                    pos_diff = abs(grc_loc-en_loc)
                                    en_words = []
                                    en_postags = []
                                    for index in possibility:
                                        if index >= len(lemmatized_sentence):
                                            print(index)
                                            print(indices_new)
                                            print(grc_en)
                                            print(lemmatized_sentence)
                                            print(tokenized_simple)
                                            print(en_sent['en_sent'])
                                        en_words.append(lemmatized_sentence[index])
                                        en_postags.append(pos_tagged[index])
                                    en_postag = ''
                                    if len(en_postags) == 1:
                                        en_postag = en_postags[0]
                                    else:
                                        en_postag = self.get_en_pos_multiword(en_postags)
                                    en_words_str = '_'.join(en_words)
                                    pmi = 0
                                    if en_words_str in pmis:
                                        pmi = pmis[en_words_str]
                                    rel_freq = 0.0
                                    if en_words_str in en_freqs:
                                        rel_freq = en_freqs[en_words_str] / grc_freq
                                    in_lexicon = False
                                    if grc_lemma in self.lexicon:
                                        in_lexicon = en_words_str in self.lexicon[grc_lemma]
                                    cosine = 0
                                    grc_stripped = (strip_accents(grc_lemma).replace('ς','σ')).lower()
                                    if grc_stripped + '_' + en_words_str in self.vectors_cosines:
                                        cosine = self.vectors_cosines[grc_stripped + '_' + en_words_str]
                                    elif '/c/grc/'+grc_stripped in self.vectors and '/c/en/'+en_words_str in self.vectors:
                                        cosine = numpy.dot(self.vectors['/c/grc/'+grc_stripped], self.vectors['/c/en/'+en_words_str])/(numpy.linalg.norm(self.vectors['/c/grc/'+grc_stripped])* numpy.linalg.norm(self.vectors['/c/en/'+en_words_str]))
                                    self.vectors_cosines[grc_stripped + '_' + en_words_str] = cosine
                                    cosine_lexicon = self.get_cosine_lexicon(grc_lemma,en_words_str)
                                    is_phrase = len(en_words) > 1
                                    dataset.append([group_index,grc_lemma,en_words_str,','.join(map(str,possibility)),aligned,pmi,rel_freq,pos_diff,in_lexicon,cosine,grc_postag,en_postag,cosine_lexicon,is_phrase])
        pd_dataset = pd.DataFrame(dataset,columns=['GROUP','GRC','EN','INDICES','ALIGNED','PMI','RELFREQ','POSITION','LEXICON','COSINE','GRCPOS','ENPOS','COSINE_LEXICON','IS_PHRASE'])
        pd_dataset = pd_dataset.astype({"GRCPOS": "category", "ENPOS": "category"})
        return pd_dataset, groups
    
    def get_group_size(self,data):
        return data.reset_index().groupby("GROUP")['GROUP'].count()
    
    def train_model(self,data,train_test_split,train_size=0.9,seed=None):
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
        model = LGBMRanker(objective="lambdarank",n_estimators=500,random_state=seed)
        model.fit(train_instances.drop(columns=['GROUP','GRC','EN','INDICES','ALIGNED']),train_instances['ALIGNED'],group=train_groups)
        if train_test_split:
            test_instances = self.make_predictions(model,test_instances)
            return model, test_instances
        else:
            return model
    
    def make_predictions(self,model,test_data,has_gold=True):
        if has_gold:
            predictions = model.predict(test_data.drop(columns=['GROUP','GRC','EN','INDICES','ALIGNED']))
        else:
            predictions = model.predict(test_data.drop(columns=['GROUP','GRC','EN','INDICES']))
        test_data['PRED'] = predictions
        return test_data
        
    def score_predictions(self,test_data,group_indices,score_threshold=None,evaluate_accuracy=False):
        final_results = []
        for group in tqdm(group_indices,desc='Scoring predictions'):
            group_predictions = test_data[test_data['GROUP']==group]
            if len(group_predictions) > 0:
                best_prediction = group_predictions.loc[group_predictions['PRED'].idxmax()]
                best_score = best_prediction['PRED']
                if score_threshold is None or best_score >= score_threshold:
                    row = [best_prediction['GROUP'],best_prediction['GRC'],best_prediction['EN'],best_prediction['INDICES'],best_score]
                    if evaluate_accuracy:
                        row.append(best_prediction['ALIGNED'])
                    final_results.append(row)
        if evaluate_accuracy:
            final_results = pd.DataFrame(final_results,columns=['GROUP','GRC','EN','INDICES','SCORE','ALIGNED'])
            print(final_results[final_results['ALIGNED']==1].shape[0] / final_results.shape[0])
        else:
            final_results = pd.DataFrame(final_results,columns=['GROUP','GRC','EN','INDICES','SCORE'])
        return final_results