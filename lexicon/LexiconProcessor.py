from tokenization import Tokenization
import unicodedata

class LexiconProcessor:
        
    def __init__(self,lexicon):
        self.lexicon = lexicon
    
    def add_data(self, data, feats, col_token=1, col_lemma=2, col_upos=3, col_xpos=4, col_morph=5,
                 normalization_rule=None, reader_preset="CONLLU"):
        for sent in data:
            for word in sent:
                morph = word[col_morph]
                morph_dict = {}
                if reader_preset == "CONLLU":
                    if morph != {}:
                        morph_dict = morph
                else:
                    if morph != '_':
                        for feat_val in morph.split('|'):
                            feat, val = feat_val.split('=')
                            morph_dict[feat] = val
                tag = []
                for feat in feats:
                    if feat == 'UPOS':
                        tag.append((feat, word[col_upos]))
                    elif feat == 'XPOS':
                        tag.append((feat, word[col_xpos]))
                    elif feat == 'lemma':
                        tag.append((feat, word[col_lemma]))
                    else:
                        if feat in morph_dict:
                            tag.append((feat, morph_dict[feat]))
                        else:
                            tag.append((feat, '_'))
                tag = tuple(tag)
                form = word[col_token]
                if normalization_rule == 'greek_glaux':
                    form = Tokenization.normalize_greek_punctuation(form)
                    form = Tokenization.normalize_greek_nfd(form)
                    form = Tokenization.normalize_greek_accents(form)
                elif normalization_rule == 'NFD' or normalization_rule == 'NFKD' or normalization_rule == 'NFC' or normalization_rule == 'NFKC':
                    form = unicodedata.normalize(normalization_rule,form)
                if form in self.lexicon:
                    tags = self.lexicon[form]
                    if tag not in tags:
                        tags.append(tag)
                else:
                    tags = []
                    tags.append(tag)
                    self.lexicon[form] = tags
    
    def write_lexicon(self,output,morph_feats,output_format='tab',lemma_name='lemma',pos_name='XPOS',separator_feat='|',separator_val='='):
        entries_processed = set()
        with open(output, 'w', encoding='UTF-8') as outfile:
            if output_format=='tab':
                outfile.write('form\tlemma\tXPOS')
                for feat in morph_feats:
                    outfile.write('\t'+feat)
                outfile.write('\n')
            for form, entry in self.lexicon.items():
                for analysis in entry:
                    analysis_dict = dict(analysis)
                    if output_format=='CONLL':
                        morph = ''
                        for feat in morph_feats:
                            val = analysis_dict[feat]
                            if val != '_':
                                morph += feat + separator_val + val + separator_feat
                        if len(morph) == 0:
                            morph = '_'
                        else:
                            morph = morph.rstrip(morph[-1])
                        line = form+'\t'+analysis_dict[lemma_name]+'\t'+analysis_dict[pos_name]+'\t'+morph+'\n'
                        if not line in entries_processed:
                            outfile.write(line)
                        entries_processed.add(line)
                    elif output_format=='tab':
                        line = form+'\t'+analysis_dict[lemma_name]+'\t'+analysis_dict[pos_name]
                        for feat in morph_feats:
                            line += '\t'+analysis_dict[feat]
                        line+='\n'
                        if not line in entries_processed:
                            outfile.write(line)
                        entries_processed.add(line)
    
    def trim_lexicon(self,features):
        # Removes unnecessary information and sorts the tag in the order of the feature dict
        lexicon_new = {}
        for form, tags in self.lexicon.items():
            new_tags = []
            for tag in tags:
                new_tag = []
                for feat in features:
                    for tag_feat in tag:
                        if tag_feat[0] == feat:
                            new_tag.append(tag_feat)
                new_tag = tuple(new_tag)
                new_tags.append(new_tag)
            lexicon_new[form] = new_tags
        self.lexicon = lexicon_new
                        