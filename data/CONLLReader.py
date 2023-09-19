import re

class CONLLReader:
    
    def __init__(self,preset='UD',feature_cols=None):
        if feature_cols is None:
            feature_cols = dict()
            if preset == 'UD':
                feature_cols['ID'] = 0
                feature_cols['FORM'] = 1
                feature_cols['LEMMA'] = 2
                feature_cols['UPOS'] = 3
                feature_cols['XPOS'] = 4
                feature_cols['FEATS'] = 5
                feature_cols['HEAD'] = 6
                feature_cols['DEPREL'] = 7
                feature_cols['DEPS'] = 8
                feature_cols['MISC'] = 9
            elif preset == 'simple':
                feature_cols['ID'] = 0
                feature_cols['FORM'] = 1
                feature_cols['MISC'] = 2
            self.feature_cols = feature_cols
        else:
            self.feature_cols = feature_cols
    
    def parse_conll(self,filename):
        file = open(filename,encoding='utf-8')
        raw_text = file.read().strip()
        raw_sents = re.split(r'\n\n', raw_text)
        return raw_sents
    
    def read_tags(self, feature, data, in_feats=False, return_tags=True, return_words=True):
        # Reads the values for a given feature (UPOS, XPOS or morphological) from the training set, as well as the wids and forms
        wid_sents = []
        token_sents = []
        tag_sents = []

        for sent in data:
            wids = []
            tokens = []
            tags = []
            for line in sent.split("\n"):
                split = line.split("\t")
                if return_words:
                    wids.append(split[self.feature_cols['ID']])
                    tokens.append(split[self.feature_cols['FORM']])
                if return_tags:
                    if not in_feats:
                        tags.append(split[self.feature_cols[feature]])
                    else:
                        if split[self.feature_cols['FEATS']] == '_':
                            #feature_values = self.feature_dict[feature]
                            #feature_values.add('_')
                            tags.append('_')
                        else:
                            morph = split[self.feature_cols['FEATS']].split('|')
                            in_conll = False
                            for feature_value in morph:
                                feat_split = feature_value.split('=')
                                if feat_split[0] == feature:
                                    tags.append(feat_split[1])
                                    in_conll = True
                                    break
                            if not in_conll:
                                #feature_values = self.feature_dict[feature]
                                #feature_values.add('_')
                                tags.append('_')
            wid_sents.append(wids)
            token_sents.append(tokens)
            tag_sents.append(tags)

        if return_tags and return_words:
            return wid_sents, token_sents, tag_sents
        elif return_tags:
            return tag_sents
        else:
            return wid_sents, token_sents
