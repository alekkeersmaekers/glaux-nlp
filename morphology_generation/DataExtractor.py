from treebanks.Tagsets import feats_to_perseus, ud_to_feats
import re
from tqdm import tqdm
import random
import numpy as np
from sklearn.model_selection import KFold

class DataExtractor:
    
    def __init__(self,lexicon_file,connector=None,dialects=None,poetic=True,prose=True,genres=None,texts=None,lemma_strings_file=None,written_word_strings_file=None,word_order_file=None):
        self.wwid_word = {}
        self.form_morph = {}
        if lemma_strings_file is not None:
            self.read_corpus_ids(lemma_strings_file,written_word_strings_file)
        self.read_lexicon(lexicon_file)
        if word_order_file is not None:
            self.create_data_file(word_order_file,texts)
        else:
            self.create_data_sql(connector,dialects,poetic,prose,genres)
    
    def read_corpus_ids(self,lemma_strings_file,written_word_strings_file):
        lemmaid_lemma = {}
        with open(lemma_strings_file,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in lines:
                sl = line.strip('\n').split('\t')
                lemmaid_lemma[sl[0]] = sl[1]
        with open(written_word_strings_file,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in lines:
                sl = line.strip('\n').split('\t')
                self.wwid_word[sl[0]] = [sl[2],lemmaid_lemma[sl[1]]]
        
    def read_lexicon(self,lexicon_file):
        with open(lexicon_file,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines,desc='Reading lexicon'):
                sl = line.strip('\n').split('\t')
                morph = re.sub(r'\$infl:.*','',sl[3])
                if sl[3].startswith('infl'):
                    morph = '_'
                # Problem with Morpheus/Glaux, can be removed if this is fixed
                morph = re.sub(r'(tense=(pf|plupf).*)voice=pass',r'\1voice=mid',morph).replace('$','|').replace(':','=')
                perseus_tag = feats_to_perseus(ud_to_feats([sl[2],morph]))
                parts = re.sub(r'.*\$(infl:.*)',r'\1',sl[3]).split('$')
                for i, part in enumerate(parts):
                    parts[i] = re.sub('^[^:]+:','',part)
                self.form_morph[(sl[0],sl[1],perseus_tag)] = parts
    
    def create_data_sql(self,connector,dialects=None,poetic=True,prose=True,genres=None):
        print('Querying glaux')
        data = []
        query = "SELECT glaux_id, word, lemma_string, POS_pos, POS_person, POS_number, POS_tense, POS_mood, POS_diathese, POS_gender, POS_morph_case, POS_degree FROM ((`wordorder` JOIN text_metadata ON unit_id = text_metadata.ID) JOIN written_word_strings ON written_word_id = written_word_strings.ID) JOIN lemma_strings ON written_word_strings.lemma_id = lemma_strings.lemma_id WHERE word != 'E' AND lemma_string != 'G' AND POS_pos != 'u' and word != ''"
        if dialects is not None:
            dialects_str = ','.join([f"'{x}'" for x in dialects])
            query += f" AND languageVariety IN ({dialects_str})"
        if genres is not None:
            genres_str = ','.join([f"'{x}'" for x in genres])
            query += f" AND category IN ({genres_str})"
        elif not poetic:
            query += f" AND category NOT IN ('Comedy', 'Epic poetry', 'Lyric poetry', 'Religious Poetry', 'Scientific Poetry', 'Tragedy')"
        elif not prose:
            query += f" AND category IN ('Comedy', 'Epic poetry', 'Lyric poetry', 'Religious Poetry', 'Scientific Poetry', 'Tragedy')"
        conn = connector.initiate_connection()
        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            conn.close()
        for row in tqdm(rows,desc='Processing glaux results'):
            glaux_id = row['glaux_id']
            pos_proper = row['POS_pos']
            pos = f"{pos_proper}{row['POS_person']}{row['POS_number']}{row['POS_tense']}{row['POS_mood']}{row['POS_diathese']}{row['POS_gender']}{row['POS_morph_case']}{row['POS_degree']}"
            form = row['word']
            lemma = row['lemma_string']
            form = re.sub(r'([͂́])(.*)́', r'\1\2',form.replace("̀", "́"))
            parts = self.form_morph.get((form,lemma,pos))
            if parts is not None and (pos_proper in ['v','a','n','d','l','m','p']):
                entry = [glaux_id,form,lemma,pos,parts[0],parts[1],parts[2],parts[3],parts[4],parts[6]]
            else:
                entry = [glaux_id,form,lemma,pos,'_','_','_','_','_','_']
            data.append(entry)
        self.data = data
    
    def create_data_file(self,word_order_file,texts):
        data = []
        with open(word_order_file,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in lines:
                sl = line.strip('\n').split('\t')
                if texts is None or sl[3] in texts:
                    pos = ''.join(x for x in sl[6:15])
                    word = self.wwid_word[sl[2]]
                    if word[0] != 'E' and not 'G' in word[0] and sl[6] != 'u' and word[0] != '':
                        word[0] = re.sub(r'([͂́])(.*)́', r'\1\2',word[0].replace("̀", "́"))
                        parts = self.form_morph.get((word[0],word[1],pos))
                        if parts is not None:
                            entry = [sl[0],word[0],word[1],pos,parts[0],parts[1],parts[2],parts[3],parts[4],parts[6]]
                        else:
                            entry = [sl[0],word[0],word[1],pos,'_','_','_','_','_','_']
                        data.append(entry)
        self.data = data
        
    def training_test_split(self,prop_training=0.9,seed=None):
        data = self.data.copy()
        if seed is not None:
            random.Random(seed).shuffle(data)
        else:
            random.shuffle(data)
        start_test = round(len(data)*prop_training,0)
        train = data[0:start_test]
        test = data[start_test:len(data)]
        return train, test
    
    def split_n_fold(self,n=10,seed=None):
        data = np.array(self.data)
        kf = KFold(n_splits=n,shuffle=True,random_state=seed)
        train_folds = []
        test_folds = []
        for train_index, test_index in kf.split(data):
            train_folds.append(data[train_index].tolist())
            test_folds.append(data[test_index].tolist())
        return train_folds, test_folds
    
    def write_data(self,data,output,training=True):
        with open(output,'w',encoding='utf8') as outfile:
            for entry in data:
                if training:
                    output = '\t'.join(x for x in entry[1:len(entry)])
                else:
                    output = '\t'.join(str(x) for x in entry[0:4])
                outfile.write(f'{output}\n')
        