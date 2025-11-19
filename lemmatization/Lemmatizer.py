from collections import Counter
from data.CONLLReader import CONLLReader
from stanza.models.lemma.trainer import Trainer
from lemmatization.LemmaDataLoader import LemmaDataLoader
from tokenization.Tokenization import normalize_token

class Lemmatizer:
    
    def __init__(self,lexicon_file,model_file,training_data,test_data=None,data_preset='CONLLU',feature_cols=None,morph_features=None,pos_name='XPOS',normalization=None):
        self.pos_name = pos_name
        self.morph_features = morph_features
        if morph_features is None:
            self.lexicon, self.morph_features = self.load_lemma_lexicon(lexicon_file,True)
        else:
            self.lexicon = self.load_lemma_lexicon(lexicon_file)
        self.reader = CONLLReader(data_preset,feature_cols)
        self.all_lemma_counts = Counter()
        self.formtag_lemma_counts = {}
        self.model = Trainer(model_file=model_file, device='cuda')
        self.training_data = self.reader.parse_conll(training_data)
        self.add_counts(self.training_data)
        if test_data is not None:
            self.test_data = self.reader.parse_conll(test_data)
        self.normalization = normalization

    def load_lemma_lexicon(self,file,return_features=False):
        lexicon = {}
        with open(file,encoding='utf8') as infile:
            lines = infile.readlines()
            header = lines[0].strip('\n').split('\t')
            columns = {i: c for i, c in enumerate(header)}
            for line in lines[1:len(lines)]:
                sl = line.strip('\n').split('\t')
                lemma = None
                form_tag = []
                for i, c in enumerate(sl):
                    if columns[i] == 'lemma':
                        lemma = c
                    else:
                        form_tag.append((columns[i], c))
                form_tag = tuple(form_tag)
                lemmas = lexicon.get(form_tag,set())
                lemmas.add(lemma)
                lexicon[form_tag] = lemmas
        if return_features:
            feats = header.copy()
            feats.remove(self.pos_name)
            feats.remove('lemma')
            feats.remove('form')
            return lexicon, feats
        else:
            return lexicon

    def get_unknown_words(self,data):
        unknown_words = []
        for sent in data:
            for word in sent:
                lemma = word[self.reader.feature_cols['LEMMA']]
                if lemma is None:
                    lemma = '_'
                form = word[self.reader.feature_cols['FORM']]
                if self.normalization is not None:
                    form = normalize_token(form,self.normalization)
                pos = word[self.reader.feature_cols[self.pos_name]]
                morph = word[self.reader.feature_cols['FEATS']]
                form_tag = self.get_lexicon_key(form,pos,morph)
                if not form_tag in self.lexicon:
                    morph_list = []
                    for k, v in morph.items():
                        morph_list.append(f'{k}={v}')
                    morph_str = '|'.join(morph_list)
                    unknown_words.append([form,f'{pos}_{morph_str}',lemma])
        return unknown_words

    def add_counts(self,data):
        for sent in data:
            for word in sent:
                lemma = word[self.reader.feature_cols['LEMMA']]
                self.all_lemma_counts[lemma] += 1
                form = word[self.reader.feature_cols['FORM']]
                pos = word[self.reader.feature_cols[self.pos_name]]
                morph = word[self.reader.feature_cols['FEATS']]
                form_tag = self.get_lexicon_key(form,pos,morph)
                lemma_counts = self.formtag_lemma_counts.get(form_tag,Counter())
                lemma_counts[lemma] += 1
                self.formtag_lemma_counts[form_tag] = lemma_counts

    def get_lexicon_key(self,form,pos,morph):
        form_tag = []
        form_tag.append(('form',form))
        form_tag.append((self.pos_name,pos))
        for feat in self.morph_features:
            if feat in morph:
                form_tag.append((feat,morph[feat]))
            else:
                form_tag.append((feat,'_'))
        form_tag = tuple(form_tag)
        return form_tag

    def lemmatize(self,data,return_possibilities=False,beam_size=1):

        unknown_words = self.get_unknown_words(data)
        unknown_words_forms = []
        for unknown_word in unknown_words:
            unknown_words_forms.append(unknown_word[0])
        loaded_args, vocab = self.model.args, self.model.vocab
        batch = LemmaDataLoader(None, 50, loaded_args, vocab=vocab, evaluation=True,data=unknown_words)
        preds = []
        edits = []
        for i, b in enumerate(batch):
            ps, es = self.model.predict(b, beam_size)
            preds += ps
            if es is not None:
                edits += es
        preds = self.model.postprocess(unknown_words_forms, preds, edits=edits)
    
        unknown_count = -1   
        all_lemmas = []
        all_poss = []
        
        for sent in data:
            sent_lemmas = []
            sent_poss = []
            for word in sent:
                form = word[self.reader.feature_cols['FORM']]
                if self.normalization is not None:
                    form = normalize_token(form,self.normalization)
                pos = word[self.reader.feature_cols[self.pos_name]]
                morph = word[self.reader.feature_cols['FEATS']]
                form_tag = self.get_lexicon_key(form,pos,morph)
                possibilities = self.lexicon.get(form_tag,set())
                sent_poss.append(possibilities)
                if len(possibilities) == 0:
                    unknown_count += 1
                    sent_lemmas.append(preds[unknown_count])
                elif len(possibilities) == 1:
                    sent_lemmas.append(list(possibilities)[0])
                else:
                    sent_lemmas.append(self.get_most_frequent(possibilities,form_tag))
            all_lemmas.append(sent_lemmas)
            all_poss.append(sent_poss)
            
        if return_possibilities:
            return all_lemmas, all_poss
        else:
            return all_lemmas
    
    def get_most_frequent(self,possibilities,form_tag):
        best_lemmas = []
        best_count = 0
        for lemma in possibilities:
            count = self.formtag_lemma_counts.get(form_tag,Counter())[lemma]
            if count > best_count:
                best_lemmas = []
                best_lemmas.append(lemma)
                best_count = count
            elif count == best_count:
                best_lemmas.append(lemma)
        if len(best_lemmas) > 1:
            new_best_lemmas = []
            new_best_count = 0
            for lemma in possibilities:
                count = self.all_lemma_counts[lemma]
                if count > new_best_count:
                    new_best_lemmas = []
                    new_best_lemmas.append(lemma)
                    new_best_count = count
                elif count == new_best_count:
                    new_best_lemmas.append(lemma)
            # Technically there can still be multiple possibilities, but then we just always take the first one on the list
            return new_best_lemmas[0]
        else:
            return best_lemmas[0]
    
    def write_prediction(self, predicted_lemmas, output_file, output_format='CONLLU', output_gold=True, output_sentence=True, possibilities=None):

        with open(output_file, 'w', encoding='UTF-8') as outfile:
            if output_format == 'tab':
                outfile.write(f'id\ttoken\t{self.pos_name}\tmorph')
                if possibilities is not None:
                    outfile.write("\tno_poss\tposs")
                if output_gold:
                    outfile.write('\tgold')
                outfile.write('\tpred')
                if output_sentence:
                    outfile.write("\tsentence")
                outfile.write('\n')
            for sent_no, sent in enumerate(self.test_data):
                for word_no, word in enumerate(sent):
                    feats_lst = []
                    for k, v in word[self.reader.feature_cols['FEATS']].items():
                        feats_lst.append(f'{k}={v}')
                    feats_str = '|'.join(feats_lst)
                    if output_format == 'CONLLU':
                        outfile.write(f"{word[self.reader.feature_cols['ID']]}\t{word[self.reader.feature_cols['FORM']]}\t{predicted_lemmas[sent_no][word_no]}\t{word[self.reader.feature_cols['UPOS']]}\t{word[self.reader.feature_cols['XPOS']]}\t{feats_str}\t_\t_\t_\t_\n")
                    elif output_format == 'tab':
                        outfile.write(f"{word[self.reader.feature_cols['ID']]}\t{word[self.reader.feature_cols['FORM']]}\t{word[self.reader.feature_cols[self.pos_name]]}\t{feats_str}")
                        if possibilities is not None:
                            poss = possibilities[sent_no][word_no]
                            outfile.write(f"\t{len(poss)}\t{'|'.join(poss)}")
                        if output_gold:
                            outfile.write(f"\t{word[self.reader.feature_cols['LEMMA']]}")
                        outfile.write(f"\t{predicted_lemmas[sent_no][word_no]}")
                        if output_sentence:
                            sent_str = ''
                            for word_id_2, word_2 in enumerate(sent):
                                if word_id_2 == word_no:
                                    sent_str += '['
                                sent_str += word_2[self.reader.feature_cols['FORM']]
                                if word_id_2 == word_no:
                                    sent_str += ']'
                                sent_str += ' '
                            outfile.write(f'\t{sent_str.strip()}')
                        outfile.write('\n')
                if output_format == 'CONLLU':
                    outfile.write('\n')