import re
import unicodedata

def check_subword_limit(sentences, tokenizer, max_length=512, check_min_tokens=128):
    sentence_nos = []
    for sentence_no, sentence in enumerate(sentences):
        if len(sentence) >= check_min_tokens:
            encodings = tokenizer(sentence, truncation=True, max_length=max_length, is_split_into_words=True)
            if len(encodings['input_ids']) == max_length:
                sentence_nos.append(sentence_no)
    return sentence_nos
    
def tokenize_sentence(sentence, tokenizer, return_tensors=None):
    encodings = tokenizer(sentence['tokens'], truncation=True, max_length=512, is_split_into_words=True,return_tensors=return_tensors)
    # You will get an error if a TokenizerFast cannot be used!!!
    encodings['subword_ids'] = encodings.word_ids()
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
            word = normalize_token(word,normalization_rule)
            sent_norm.append(word)
        tokens_norm.append(sent_norm)
    return tokens_norm

def normalize_token(token,normalization_rule):
    if normalization_rule == 'greek_glaux':
        token = normalize_greek_punctuation(token)
        token = normalize_greek_nfd(token)
        token = normalize_greek_accents(token)
    elif normalization_rule == 'NFD' or normalization_rule == 'NFKD' or normalization_rule == 'NFC' or normalization_rule == 'NFKC':
        token = unicodedata.normalize(normalization_rule,token)
    return token

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

def strip_accents(string):
    return ''.join(c for c in unicodedata.normalize('NFD', string) if unicodedata.category(c) != 'Mn')