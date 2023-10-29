import re
import unicodedata

def tokenize_sentence(sentence, tokenizer, return_tensors=None):
    encodings = tokenizer(sentence['tokens'], truncation=True, max_length=512, is_split_into_words=True, return_offsets_mapping=True,return_tensors=return_tensors)
    return encodings

def normalize_greek_nfd(word):
    word_norm = ''
    for c in word:
        if c=='ʹ' or c=='·':
            word_norm += c
        else:
            word_norm += unicodedata.normalize('NFD',c)
    return word_norm
        

def normalize_tokens(tokens,normalization_rule):
    tokens_norm = []
    for sent in tokens:
        sent_norm = []
        for word in sent:
            if normalization_rule == 'greek_glaux':
                word = normalize_greek_punctuation(word)
                word = normalize_greek_nfd(word)
                word = normalize_greek_accents(word)
            elif normalization_rule == 'NFD' or normalization_rule == 'NFKD' or normalization_rule == 'NFC' or normalization_rule == 'NFKC':
                word = unicodedata.normalize(normalization_rule,word)
            sent_norm.append(word)
        tokens_norm.append(sent_norm)
    return tokens_norm

def normalize_greek_accents(regularized: str):
    """
    Replace gravis by acutus, removes second acutus if the following word is enclitic
    """
    
    
    if '\u0300' in regularized:  # check for gravis and replace with acutus
        return regularized.replace('\u0300', '\u0301')

    # Match the whole word, containing acutus/perispomenus up until the second acutus in group 1, and the rest, excluding the acutus in group 2.
    double_acutus_pattern = '([^\u0301\u0342]*[\u0301\u0342][^\u0301\u0342]*)\u0301([^\u0301\u0342]*)'
    m = re.match(double_acutus_pattern, regularized)

    if m:
        return re.sub(double_acutus_pattern, m.group(1) + m.group(2), regularized)

    return regularized

def normalize_greek_punctuation(word):
    word = re.sub(r'[᾽\'ʼ\\u0313´]', '’',word)
    word = re.sub(r'[‑—]', '—',word)
    word = re.sub('--', '—',word)
    word = re.sub(r'[“”„‘«»ʽ"]', '"',word)
    word = re.sub(r'[:··•˙]', '·',word)
    word = re.sub(';', ';',word)
    word = re.sub(r'[（\(]', r'(',word)
    word = re.sub(r'[）\)]', r')',word)
    return word

def greek_glaux_to_tokens(string):
    string = re.sub(r'([\.,—"·;\(\)‑“”„‘«»ʽ:·•˙;（）†])|(--)',r' \1 ',string)
    string = re.sub(r'\s+',' ',string)
    string = re.sub(r'^ ', '',string);
    string = re.sub(r' $', '',string);
    tokens_str = string.split(' ')
    return tokens_str
