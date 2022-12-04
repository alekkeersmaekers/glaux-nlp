class LexiconProcessor:
    
    def __init__(self,lexicon):
        self.lexicon = lexicon
    
    def add_data(self,data,feats,col_token=2,col_lemma=3,col_upos=4,col_xpos=5,col_morph=6):
        for sent in data:
            for line in sent.split("\n"):
                split = line.split("\t")
                morph = split[col_morph]
                morph_dict = {}
                if morph != '_':
                    for feat_val in morph.split('|'):
                        feat, val = feat_val.split('=')
                        morph_dict[feat] = val
                tag = []
                for feat in feats:
                    if feat == 'UPOS':
                        tag.append((feat, split[col_upos]))
                    elif feat == 'XPOS':
                        tag.append((feat, split[col_xpos]))
                    elif feat == 'lemma':
                        tag.append((feat, split[col_lemma]))
                    else:
                        if feat in morph_dict:
                            tag.append((feat, morph_dict[feat]))
                        else:
                            tag.append((feat, '_'))
                tag = tuple(tag)
                form = split[col_token]
                if form in self.lexicon:
                    tags = self.lexicon[form]
                    if tag not in tags:
                        tags.append(tag)
                else:
                    tags = []
                    tags.append(tag)
                    self.lexicon[form] = tags
    
    def write_lexicon(self,output,output_format,morph_feats,lemma_name='lemma',pos_name='XPOS'):
        entries_processed = set()
        with open(output, 'w', encoding='UTF-8') as outfile:
            for form, entry in self.lexicon.items():
                for analysis in entry:
                    analysis_dict = dict(analysis)
                    if output_format=='CONLL':
                        morph = ''
                        for feat in morph_feats:
                            val = analysis_dict[feat]
                            if val != '_':
                                morph += feat + '=' + val + '|'
                        if len(morph) == 0:
                            morph = '_'
                        else:
                            morph = morph.rstrip(morph[-1])
                        line = form+'\t'+analysis_dict[lemma_name]+'\t'+analysis_dict[pos_name]+'\t'+morph+'\n'
                        if not line in entries_processed:
                            outfile.write(form+'\t'+analysis_dict[lemma_name]+'\t'+analysis_dict[pos_name]+'\t'+morph+'\n')
                        entries_processed.add(line)