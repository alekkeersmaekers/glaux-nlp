from bleach._vendor.html5lib.filters import lint
class LexiconProcessor:
    
    def __init__(self,lexicon):
        self.lexicon = lexicon
    
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