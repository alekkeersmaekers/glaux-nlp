import re
import pyconll

class CONLLReader:
    
    def __init__(self, preset='CONLL', feature_cols=None):
        self.preset = preset
        if feature_cols is None:
            feature_cols = dict()
            if preset == "CONLL":
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
    
    def parse_conll(self, filename):
        if self.preset == "CONLL":
            conll = pyconll.load_from_file(filename) # implies UTF-8
            raw_sents = ["\n".join([token.conll() for token in sentence]) for sentence in conll]

            return raw_sents
        else:
            file = open(filename,encoding='utf-8')
            raw_text = file.read().strip()
            raw_sents = re.split(r'\n\n', raw_text)

            return raw_sents
    
    def read_tags(self, feature, data, in_feats=False, return_wids=True, return_tokens=True, return_tags=True):
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
                if return_wids:
                    wids.append(split[self.feature_cols['ID']])
                if return_tokens:
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
            
        if return_wids:
            if return_tokens:
                if return_tags:
                    return wid_sents, token_sents, tag_sents
                else:
                    return wid_sents, token_sents
            elif return_tags:
                return wid_sents, tag_sents
            else:
                return wid_sents
        else:
            if return_tokens:
                if return_tags:
                    return token_sents, tag_sents
                else:
                    return token_sents
            else:
                if return_tags:
                    return tag_sents
                else:
                    return None