import xml.etree.cElementTree as ET
import treebanks.TreeOperations as to
from treebanks.Tagsets import ud_to_feats, perseus_to_feats
from tokenization.Tokenization import normalize_token

class TreebankLoader():

    def __init__(self, wordid, normalization=None,sent_attr=None,token_attr=None,operations=None):
        self.normalization = normalization
        self.sent_attr = sent_attr
        self.token_attr = token_attr
        self.operations = operations
        self.wordid = wordid
        
    def set_heads(self,sentence):
        for token in sentence['tokens']:
            token['head'] = None
            if not token['headid'] == '0':
                for token2 in sentence['tokens']:
                    if token['headid'] == token2['id']:
                        token['head'] = token2
                        break
            del(token['headid'])
                              
    def add_sent(self,currentSent,treebank):
        if self.token_attr is None or ('head' in self.token_attr and 'id' in self.token_attr):
            self.set_heads(currentSent)
        if self.operations is not None:
            for operation in self.operations:
                method = getattr(to,operation[0])
                if len(operation)==1:
                    method(tree=currentSent['tokens'],wordid=self.wordid)
                else:
                    method(tree=currentSent['tokens'],wordid=self.wordid,**operation[1])
        if self.normalization is not None:
            for token in currentSent['tokens']:
                if 'form' in token:
                    token['form'] = normalize_token(token['form'],self.normalization)
                if 'lemma' in token:
                    token['lemma'] = normalize_token(token['lemma'],self.normalization)
        sent = {}
        sent.update(currentSent)
        treebank.append(sent)
        currentSent.clear()
    
    def load_database(self,connector,texts,manual_only=False):
        conn = connector.initiate_connection()
        texts_str = ','.join(map(str,texts))
        try:
            with conn.cursor() as cursor:
                if not manual_only:
                    query = f'select glaux_id, sentence_id, unit_id, POS_pos, POS_person, POS_number, POS_tense, POS_mood, POS_diathese, POS_gender, POS_morph_case, POS_degree, artificial, word, lemma_string, morphosyntax, target_glaux_word from ((wordorder join written_word_strings on wordorder.written_word_id = written_word_strings.ID) join lemma_strings on written_word_strings.lemma_id = lemma_strings.lemma_id) join text_heads on wordorder.glaux_id = text_heads.source_glaux_word where unit_id in ({texts_str}) order by unit_id, place_in_unit'
                else:
                    query = f'select glaux_id, sentence_id, unit_id, POS_pos, POS_person, POS_number, POS_tense, POS_mood, POS_diathese, POS_gender, POS_morph_case, POS_degree, artificial, word, lemma_string, morphosyntax, target_glaux_word from (((wordorder join written_word_strings on wordorder.written_word_id = written_word_strings.ID) join lemma_strings on written_word_strings.lemma_id = lemma_strings.lemma_id) join text_heads on wordorder.glaux_id = text_heads.source_glaux_word) join confirmed_components on wordorder.sentence_id = confirmed_components.structure_id where (unit_id in ({texts_str}) and degree = 2) order by unit_id, place_in_unit'
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            conn.close()
        treebank = []
        currentSent = {'tokens':[]}
        currentSentId = None
        for row in rows:
            sent_id = row['sentence_id']
            if currentSentId is None:
                currentSent['id'] = sent_id
                currentSent['unit_id'] = row['unit_id']
            if currentSentId is not None and not sent_id == currentSentId:
                self.add_sent(currentSent,treebank)
                currentSent['tokens'] = []
                if self.sent_attr is None or 'id' in self.sent_attr:
                    currentSent['id'] = sent_id
                if self.sent_attr is None or 'text' in self.sent_attr:
                    currentSent['text'] = row['unit_id']
            currentSentId = sent_id
            word = {}
            if self.token_attr is None or 'id' in self.token_attr:
                word['id'] = row['glaux_id']
                word['wordid'] = row['glaux_id']
            if self.token_attr is None or 'form' in self.token_attr:
                word['form'] = row['word']
            if self.token_attr is None or 'lemma' in self.token_attr:
                word['lemma'] = row['lemma_string']
            if self.token_attr is None or 'postag' in self.token_attr:
                postag = row['POS_pos'] + row['POS_person'] + row['POS_number'] + row['POS_tense'] + row['POS_mood'] + row['POS_diathese'] + row['POS_gender'] + row['POS_morph_case'] + row['POS_degree']
                word['morph'] = perseus_to_feats(postag,word['id'])
            if self.token_attr is None or 'head' in self.token_attr:
                word['headid'] = row['target_glaux_word']
            if self.token_attr is None or 'relation' in self.token_attr:
                word['relation'] = row['morphosyntax']
            if row['artificial'] == 1:
                word['artificial'] = True
            currentSent['tokens'].append(word)
        self.add_sent(currentSent,treebank)
        return treebank
            
    
    def parse_xml(self,xml_file):
        treebank = []
        parsed = ET.iterparse(xml_file,events=("start","end"))
        parsed = iter(parsed)
        currentSent = {}
        for event, elem in parsed:
            tag = elem.tag
            if event == 'end' and tag == 'sentence':
                self.add_sent(currentSent,treebank)
            elif event == 'start':
                if tag == 'sentence':
                    for name, val in elem.items():
                        if self.sent_attr is None or name in self.sent_attr:
                            currentSent[name] = val
                    currentSent['tokens'] = []
                elif tag == 'word':
                    word = {}
                    for name, val in elem.items():
                        if self.token_attr is None or name in self.token_attr:
                            if name == 'head':
                                word['headid'] = val
                            elif name == 'artificial':
                                word['artificial'] = True
                            elif name == 'postag':
                                word['morph'] = perseus_to_feats(val, word['wordid'])
                            elif name == self.wordid:
                                word['wordid'] = val
                                if self.wordid == 'id':
                                    word['id'] = val
                            else:
                                word[name] = val
                    if (self.token_attr is None or 'head' in self.token_attr) and 'headid' in word:
                    # We only append tokens to the tree that have a defined head
                        currentSent['tokens'].append(word)
            elem.clear()
        return treebank
    
    def parse_conll(self,conll_file,columns=['id','form','lemma','upos','xpos','feats','head','relation','deps','misc']):
        treebank = []
        with open(conll_file,'r',encoding='utf8') as infile:
            lines = infile.readlines()
            currentSent = {}
            for line in lines:
                line = line.strip()
                if line == '':
                    self.add_sent(currentSent,treebank)
                elif line[0] == '#':
                    if not line.startswith("# text") and not line.startswith("# file"):
                        split = line.split(" = ")
                        name = split[0].replace("# ","")
                        val = split[1]
                        if self.sent_attr is None or name in self.sent_attr:
                            currentSent[name] = val
                else:
                    word = {}
                    if not 'tokens' in currentSent:
                        currentSent['tokens'] = []
                    split = line.split('\t')
                    for i, val in enumerate(split):
                        name = columns[i]
                        if self.token_attr is None or name in self.token_attr:
                            if name == 'head':
                                word['headid'] = val
                            elif name == 'feats':
                                if val == '':
                                    val = '_'
                                pos = None
                                if 'upos' in word and word['upos'] != '_':
                                    pos = word['upos']
                                elif 'xpos' in word and word ['xpos'] != '_':
                                    pos = word['xpos']
                                if pos is not None:
                                    word['morph'] = ud_to_feats([pos,val])
                                if 'upos' in word:
                                    del word['upos']
                                if 'xpos' in word:
                                    del word['xpos']
                            elif name == self.wordid:
                                word[self.wordid] = val
                            else:
                                if name == 'form' and val == 'E':
                                    word['artificial'] = True
                                word[name] = val
                    currentSent['tokens'].append(word)
            if 'tokens' in currentSent:
                self.add_sent(currentSent,treebank)
        return treebank
