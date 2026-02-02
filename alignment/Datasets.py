from tqdm import tqdm
from spacy.tokens import Doc
from spacy.tokens import DocBin
import ast

def build_dataset(input_file,analyze=False,has_grc_forms=True,nlp=None,docbin=None,grc_lemmas=None,grc_tags=None):
    sentences = []
    en_index = 0
    grc_index = 0
    with open(input_file,encoding='utf8') as infile:
        lines = infile.readlines()
        for line in lines:
            sl = line.strip('\n').split('\t')
            sent = {}
            if has_grc_forms:
                grc_split = sl[0].split(' ')
                sent['grc'] = grc_split
                grc_id = list(range(grc_index,grc_index+len(grc_split)))
                grc_id = [str(x) for x in grc_id]
                sent['grc_ids'] = grc_id
                sent['eng'] = sl[1]
                en_split = sl[1].split(' ')
                en_id = list(range(en_index,en_index+len(en_split)))
                en_id = [str(x) for x in en_id]
                sent['en_ids'] = en_id
                if len(sl) == 3:
                    sent['alignment'] = ast.literal_eval(sl[2])
                    sentences.append(sent)
                en_index += len(en_split)
                grc_index += len(grc_split)
            else:
                sent['eng'] = sl[0]
                en_split = sl[0].split(' ')
                en_id = list(range(en_index,en_index+len(en_split)))
                en_id = [str(x) for x in en_id]
                sent['en_ids'] = en_id
                sentences.append(sent)
                en_index += len(en_split)
    if analyze:
        documents = []
        for sent in sentences:
            documents.append(sent['eng'])
        analysis = tqdm(nlp.pipe(documents), total=len(documents), desc='Analyzing sentences')
        for index, doc in enumerate(analysis):
            sentences[index]['en_tokens'] = doc
        if docbin is not None:
            docs = []
            for sent in sentences:
                doc = sent['en_tokens']
                docs.append(doc)
            doc_bin = DocBin(docs=docs)
            doc_bin.to_disk(docbin)
    elif docbin is not None:
        doc_bin = DocBin().from_disk(docbin)
        docs = list(doc_bin.get_docs(nlp.vocab))
        for doc_no, doc in enumerate(docs):
            sentences[doc_no]['en_tokens'] = doc    
    if grc_lemmas is not None:
        with open(grc_lemmas,encoding='utf8') as infile:
            lemmas_sent = []
            lines = infile.readlines()
            for line in lines:
                lemmas_sent.append(line.strip('\n').split(' '))
        for sent_no, sent in enumerate(lemmas_sent):
            if sent_no < len(sentences):
                sentences[sent_no]['grc_lemmas'] = sent
    if grc_tags is not None:
        with open(grc_tags,encoding='utf8') as infile:
            lemmas_sent = []
            lines = infile.readlines()
            for line in lines:
                lemmas_sent.append(line.strip('\n').split(' '))
        for sent_no, sent in enumerate(lemmas_sent):
            if sent_no < len(sentences):
                sentences[sent_no]['grc_pos'] = sent
    return sentences

def reduce_alignment(tokens,grc,en,mwes,phrases,grc_pos,do_print=False):
    token_list = list(tokens)
    full_tokens = []
    for index in en:
        full_tokens.append(tokens[index])
    full_tokens_str = []
    for token in full_tokens:
        full_tokens_str.append(token.text)
    heads = find_head(full_tokens)
    if do_print:
        print(full_tokens_str)
        print(heads)
    candidates_remain = []
    remove_candidates_remain = []
    rules = []
    if len(heads) > 0:
#        if full_tokens[len(full_tokens)-1].text == 'born':
#            print(full_tokens)
        for token_no, token in enumerate(full_tokens):
            if (not token.head in heads) and token.head.head in heads and token.dep_ == 'mark' and token.head.dep_ == 'ccomp' and (token.head.head.pos_ == 'VERB' or token.head.head.pos_ == 'AUX'):
                # Verb + complementizer (typically 'that')
                rules.append(1)
                pass
            elif (token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ' or token.pos_ == 'NUM') and token.dep_ == 'compound' and (token.head.head.pos_ == 'ADP' or token.head.head.pos_ == 'SCONJ'):
                # of Jesus Christ and things like that
                rules.append(2)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head.head)
            elif len(full_tokens) == 2 and (token.dep_ == 'det') and not token.head in full_tokens and token.head.head in full_tokens and token.head.head.dep_ == 'prep':
                # of all, of the, ...
                rules.append(3)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head.head)
            elif token.lemma_ == 'to' and token.dep_ == 'aux' and token == full_tokens[-1] and not token.head in full_tokens:
                # have to, continue to etc
                rules.append(4)
                pass
            elif token in heads:
                rules.append(5)
                candidates_remain.append(token)
            elif token.head in heads and (token.head.pos_ == 'VERB' or token.head.pos_ == 'AUX') and (token.pos_ == 'PRON' or token.pos_ == 'AUX' or token.pos_ == 'PART' or token.lemma_ == 'people'):
                # All kind of removable subjects and auxiliaries with verbs 
                rules.append(6)
                pass
            elif token.head in heads and (token.head.pos_ == 'VERB' or token.head.pos_ == 'AUX' or token.head.pos_ == 'ADV') and (token.dep_ == 'prep' or token.dep_ == 'dative'):
                # Unfilled prepositions with verbs
                rules.append(7)
                hasChildren = False
                for child in token.children:
                    if child in full_tokens:
                        hasChildren = True
                        break
                if not hasChildren:
                    pass
                else:
                    candidates_remain.append(token)
            elif token_no == 0 and token.pos_ == 'DET':
                rules.append(8)
                # All determiners
                pass
            elif (token.pos_ == 'NOUN' or token.pos_ == 'PROPN' or token.pos_ == 'ADJ' or token.pos_ == 'NUM' or token.pos_ == 'PRON') and (token.head.pos_ == 'ADP' or token.head.pos_ == 'SCONJ') and not token.head.head in candidates_remain and not full_tokens_str in mwes:
                # Not sure what this is, is a little annoying to test at the moment
                rules.append(9)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head)
            elif token.head in heads and token.head.lemma_ == 'let' and (token.pos_ == 'VERB' or token.pos_ == 'AUX'):
                # Let's go and such
                rules.append(10)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head)
                for child in token.head.children:
                    if child.pos_ == 'PRON':
                        remove_candidates_remain.append(child)
            elif token.head in heads and (token.head.lemma_ == 'continue' or token.head.lemma_ == 'be' or token.head.lemma_ == 'use' or token.head.lemma_ == 'begin' or token.head.lemma_ == 'go') and (token.pos_ == 'VERB' or token.pos_ == 'AUX'):
                # Continue to, was to, used to, is going to etc
                rules.append(11)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head)
                for child in token.children:
                    if child.lemma_ == 'to' or child.pos_=='AUX':
                        remove_candidates_remain.append(child)
            elif token.head in heads and (token.head.pos_ == 'VERB' or token.head.pos_=='AUX') and (token.dep_ == 'mark' or (token.pos_=='SCONJ' and token.dep_=='advmod')):
                # Any conjunctions (e.g. with participle constructions)
                rules.append(12)
                pass
            elif token.head in heads and (token.head.dep_ == 'prep') and (token.pos_ == 'VERB' or token.pos_ == 'AUX'):
                # Similar, but prepositions, also typically with participles
                rules.append(13)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head)
            elif token.head in heads and (token.head.lemma_ == 'thing' or token.head.lemma_ == 'one' or token.head.lemma_ == 'man' or token.head.lemma_ == 'person' or token.head.lemma_ == 'people') and token.dep_ == 'amod' and grc_pos != 'noun':                        
                # righteous man etc.
                rules.append(14)
                candidates_remain.append(token)
                remove_candidates_remain.append(token.head)
            elif token.head in heads and (token.head.pos_ == 'NOUN' or token.head.pos_ == 'PROPN') and (token.dep_ == 'case'):
                # genitive s
                rules.append(15)
                pass
            elif token.head in heads and (token.head.pos_ == 'ADJ' or token.head.pos_ == 'ADV') and (token.lemma_ == 'more' or token.lemma_ == 'most'):
                # comparatives, superlatives
                rules.append(16)
                pass
            elif token.head in heads and (token.head.pos_ == 'NOUN' or token.head.pos_ == 'PROPN') and (token.dep_ == 'prep'):
                # additional prepositions
                rules.append(17)
                hasChildren = False
                for child in token.children:
                    if child in full_tokens:
                        hasChildren = True
                        break
                if hasChildren:
                    candidates_remain.append(token)
            else:
                candidates_remain.append(token)
    if do_print:
        print(candidates_remain)
        print(remove_candidates_remain)
    for candidate in remove_candidates_remain:
        if candidate in candidates_remain:
            candidates_remain.remove(candidate)
    if len(candidates_remain) > 1:
        if candidates_remain[0].pos_ == 'DET':
            rules.append(18)
            candidates_remain.remove(candidates_remain[0])
        elif candidates_remain[0].pos_ == 'AUX':
            # Copula is removed if no phrase with 'be' can be found
            rules.append(19)
            candidates_remain_str = []
            for candidate in candidates_remain:
                candidates_remain_str.append(candidate.text)
            if not candidates_remain_str in mwes and not '_'.join(candidates_remain_str) in phrases:
                candidates_remain.remove(candidates_remain[0])
    if do_print:
        print(rules)
    new_indices = []
    for candidate in candidates_remain:
        new_indices.append(token_list.index(candidate))
    return new_indices

def find_head(tokens):
    head_candidates = []
    for token in tokens:
        if token.head == token or not token.head in tokens:
            head_candidates.append(token)
    return head_candidates