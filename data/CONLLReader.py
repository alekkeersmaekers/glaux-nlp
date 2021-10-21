import re

class CONLLReader:
    
    def __init__(self,file):
        self.file = file
        self.col_id = 0
        self.col_token = 1
        self.col_upos = 3
        self.col_xpos = 4
        self.col_morph = 5
    
    def parse_conll(self):
        file = open(self.file,encoding='utf-8')
        raw_text = file.read().strip()
        raw_sents = re.split(r'\n\n', raw_text)
        return raw_sents
    