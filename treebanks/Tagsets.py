def perseus_to_feats(perseus_tag, ref):
    feats = {}
    pos = ""
    morph = ""
    
    if len(perseus_tag) >= 2:
        pos = perseus_tag[0]
        morph = perseus_tag[1:]
    else:
        pos = "-"
        morph = "--------"

    pos_map = {
        "v": "verb", "n": "noun", "l": "article", "a": "adjective", "d": "adverb",
        "c": "conjunction", "r": "preposition", "g": "particle", "m": "numeral",
        "i": "interjection", "u": "PUNCT", "p": "pronoun", "b": "coordinator",
        "z": "GAP", "q": "ellipsis"
    }
    
    pos_str = pos_map.get(pos, "UNKNOWN")
    if pos_str == "UNKNOWN":
        pos_str = "GAP"
        print(f"{pos}\tpos:unknown\t{ref}")
    
    # Initialize morphological variables
    person = morph[0].lower() if len(morph) > 0 else '-'
    number = morph[1].lower() if len(morph) > 1 else '-'
    tense = morph[2].lower() if len(morph) > 2 else '-'
    mood = morph[3].lower() if len(morph) > 3 else '-'
    voice = morph[4].lower() if len(morph) > 4 else '-'
    gender = morph[5].lower() if len(morph) > 5 else '-'
    ncase = morph[6].lower() if len(morph) > 6 else '-'
    degree = morph[7].lower() if len(morph) > 7 else '-'

    # Mapping for each feature
    person_str = {'1': '1', '2': '2', '3': '3', '-': '_',}.get(person, "UNKNOWN")
    if person_str == 'UNKNOWN':
        person_str = '_'
        print(person+'\tperson:unknown')
    
    number_str = {'s': 'sg', 'p': 'pl', 'd': 'dual', '-': '_'}.get(number, "UNKNOWN")
    if number_str == 'UNKNOWN':
        number_str = '_'
        print(number+'\tnumber:unknown')
    
    tense_str = {'p': 'pres', 'i': 'impf', 'a': 'aor', 'f': 'fut', 'r': 'pf', 'l': 'plupf', 't': 'futpf', '-': '_'}.get(tense, "UNKNOWN")
    if tense_str == 'UNKNOWN':
        tense_str = '_'
        print(tense+'\ttense:unknown')
    
    mood_str = {'i': 'ind', 's': 'subj', 'o': 'opt', 'm': 'imp', '-': '_', 'p': '_', 'n': '_', 'g': '_'}.get(mood, "UNKNOWN")
    if mood_str == 'UNKNOWN':
        mood_str = '_'
        print(mood+'\tmood:unknown')   
    # gerund: not a one-to-one translation!
    
    voice_str = {'a': 'act', 'm': 'mid', 'p': 'pass', 'e': 'mid', '-': '_'}.get(voice, "UNKNOWN")
    if voice_str == 'UNKNOWN':
        voice_str = '_'
        print(voice+'\tvoice:unknown')
    # Not a one-to-one translation: mediopassive is eliminated!
    
    gender_str = {'m': 'masc', 'f': 'fem', 'n': 'neut', 'c': 'comm', '-': '_'}.get(gender, "UNKNOWN")
    if gender_str == 'UNKNOWN':
        gender_str = '_'
        print(gender+'\tgender:unknown')
    
    case_str = {'n': 'nom', 'g': 'gen', 'd': 'dat', 'a': 'acc', 'v': 'voc', '-': '_'}.get(ncase, "UNKNOWN")
    if case_str == 'UNKNOWN':
        case_str = '_'
        print(ncase+'\tcase:unknown')
    
    degree_str = {'c': 'comp', 's': 'sup', 'p': '_', '-': '_'}.get(degree, "UNKNOWN")
    if degree_str == 'UNKNOWN':
        degree_str = '_'
        print(degree+'\tdegree:unknown')

    if pos_str in ["adjective", "adverb"] and degree_str == "_":
        degree_str = "pos"
    if mood == 'p':
        pos_str = 'participle'
    elif mood == 'n':
        pos_str = 'infinitive'
    elif mood == 'g':
        pos_str = 'adjective'

    # Add features to the map
    feats["pos"] = pos_str
    feats["person"] = person_str
    feats["number"] = number_str
    feats["tense"] = tense_str
    feats["mood"] = mood_str
    feats["voice"] = voice_str
    feats["gender"] = gender_str
    feats["case"] = case_str
    feats["degree"] = degree_str
    
    return feats

def feats_to_perseus(feats):
    fullmorph = ""
    
    pos_map = {
        "verb": 'v', "participle": 'v', "infinitive": 'v', "noun": 'n', "article": 'l',
        "adjective": 'a', "adverb": 'd', "conjunction": 'c', "preposition": 'r',
        "particle": 'g', "numeral": 'm', "interjection": 'i', "PUNCT": 'u', "pronoun": 'p',
        "personal": 'p', "indefinite": 'p', "demonstrative": 'p', "relative": 'p',
        "interrogative": 'p', "coordinator": 'c', "GAP": 'z', "ellipsis": 'q'
    }
    pos_str = feats.get('pos')
    pos = pos_map.get(pos_str, 'UNKNOWN')
    if pos == 'UNKNOWN':
        pos = 'z'
        print(pos_str+'\tpos:unknown')

    person = {'1': '1', '2': '2', '3': '3', '_': '-'}.get(feats.get("person"), 'UNKNOWN')
    if person == 'UNKNOWN':
        person = '-'
        print(feats.get("person")+'\tperson:unknown')

    number = {'sg': 's', 'pl': 'p', 'dual': 'd', 'none': '-', '_': '-'}.get(feats.get("number"), 'UNKNOWN')
    if number == 'UNKNOWN':
        number = '-'
        print(feats.get("number")+'\tnumber:unknown')
    
    tense = {'pres': 'p', 'impf': 'i', 'aor': 'a', 'fut': 'f', 'pf': 'r', 'plupf': 'l', 'futpf': 't', '_': '-'}.get(feats.get("tense"), 'UNKNOWN')
    if tense == 'UNKNOWN':
        tense = '-'
        print(feats.get("tense")+'\ttense:unknown')
    
    mood = {'ind': 'i', 'subj': 's', 'opt': 'o', 'imp': 'm', '_': '-'}.get(feats.get("mood"), 'UNKNOWN')
    if mood == 'UNKNOWN':
        mood = '-'
        print(feats.get("mood")+'\tmood:unknown')
    if pos_str == 'infinitive':
        mood = 'n'
    elif pos_str == 'participle':
        mood = 'p'
    
    voice = {'act': 'a', 'mid': 'm', 'pass': 'p', '_': '-'}.get(feats.get("voice"), 'UNKNOWN')
    if voice == 'UNKNOWN':
        voice = '-'
        print(feats.get("voice")+'\tvoice:unknown')
    
    gender = {'masc': 'm', 'fem': 'f', 'neut': 'n', 'comm': 'c', 'none': '-', '_': '-'}.get(feats.get("gender"), 'UNKNOWN')
    if gender == 'UNKNOWN':
        gender = '-'
        print(feats.get("gender")+'\tgender:unknown')
    
    ncase = {'nom': 'n', 'gen': 'g', 'dat': 'd', 'acc': 'a', 'voc': 'v', 'none': '-', '_': '-'}.get(feats.get("case"), 'UNKNOWN')
    if ncase == 'UNKNOWN':
        ncase = '-'
        print(feats.get("ncase")+'\tncase:unknown')
    
    degree = {'comp': 'c', 'sup': 's', 'pos': '-', '_': '-'}.get(feats.get("degree"), 'UNKNOWN')
    if degree == 'UNKNOWN':
        degree = '-'
        print(feats.get("degree")+'\tncase:unknown')

    fullmorph += pos
    fullmorph += person
    fullmorph += number
    fullmorph += tense
    fullmorph += mood
    fullmorph += voice
    fullmorph += gender
    fullmorph += ncase
    fullmorph += degree
    
    return fullmorph

def ud_to_feats(ud_tag):
    feats = {
        "pos": ud_tag[0],
        "person": "_",
        "number": "_",
        "tense": "_",
        "mood": "_",
        "voice": "_",
        "gender": "_",
        "case": "_",
        "degree": "_"
    }
    
    if ud_tag[1] != "_":
        morph = ud_tag[1].split("|")
        for m in morph:
            m_split = m.split("=")
            if len(m_split) == 2:
                feats[m_split[0]] = m_split[1]
            else:
                print(f"Error in UD tag: {ud_tag}")
    
    return feats