import re
from collections import Counter
from tqdm import tqdm
import unicodedata as ud
import beta_code

class ParadigmGenerator:

    def __init__(self,training_data,min_form_freq=10,allow_voice_change=True,most_frequent_voice=False,voice_change_threshold=20,nfd=False):
        self.min_form_freq = min_form_freq
        self.allow_voice_change = allow_voice_change
        self.most_frequent_voice = most_frequent_voice
        self.voice_change_threshold = voice_change_threshold
        self.nfd = nfd
        self.lemma_stems = {}
        # Goes from lemma (e.g. σπένδω) to principal part type (e.g. future stem) to stem (e.g. speis) to frequency (e.g. 5)
        self.lemma_stem_inflClass = {}
        # Goes from lemma (e.g. σπένδω) to principal part type (e.g. future stem) to stem (e.g. speis) to inflection class (e.g. reg_fut) to frequency (e.g. 5)
        self.inflClass_pos_endings = {}
        # Goes from inflection class (e.g. reg_fut) to pos (e.g. v--fim---) to ending (e.g. esqai) to frequency (e.g. 5)
        self.lemma_stem_augments = {}
        # Goes from lemma (e.g. ἐξεγείρω) to stem (e.g. e)geir) to augment (e.g. e)>h)) to frequency (e.g. 5)
        # Note that we assume that augments are only added to past indicatives (this is consistent with Morpheus)
        self.lemma_stem_prefix = {}
        # Goes from lemma (e.g. ἐξεγείρω) to stem (e.g. e)geir) to prefix (e.g. e)k). Sometimes prefices are part of the stem for some reason, so we have to go to the stem before deciding the prefix. For each stem, there is only one prefix.
        self.augment_strategies = {}
        # Goes from starting vowel (or consonant) (e.g. i)) to augment strategy (e.g. i)>i_)) to frequency (e.g. 5)
        self.voice_frequencies = {}
        # Goes from lemma (e.g. σπένδω) to principal part type (e.g. future stem) to stem (e.g. speis) to inflection class (e.g. reg_fut) to voice (e.g. a) to frequency (e.g. 5)
        self.form_frequencies = {}
        # Goes from lemma (e.g. εἰμί) to pos (e.g. v3spoa---) to form (e.g. εἴη) to frequency (e.g. 5)
        self.nomLemma_stems = {}
        # Goes from lemma (e.g. βασιλεύς) to stem (e.g. basil) to frequency (e.g. 5)
        self.nomLemma_stem_conjClass = {}
        # Goes from lemma (e.g. βασιλεύς) to stem (e.g. basil) to conjugation class (e.g. eus_ews) to frequency (e.g. 5)
        self.nomLemma_stem_conjClass_accentClass = {}
        # Goes from lemma (e.g. βασιλεύς) to stem (e.g. basil) to conjugation class (e.g. eus_ews) to accentClass (e.g. suff_acc) to frequency (e.g. 5)
        self.conjClass_pos_endings = {}
        #  Goes from conjugation class (e.g. eus_ews) to pos (e.g. n-s---mg-) to ending (e.g. hos) to frequency (e.g. 5)
        self.train(training_data)
        
    def train(self,training_data):
        with open(training_data,encoding='utf8') as infile:
            lines = infile.readlines()
            for line in tqdm(lines,desc='Training'):
                sl = line.strip('\n').split('\t')
                form = sl[0]
                lemma = sl[1]
                pos = sl[2]
                infl = sl[3]
                prefix = sl[4]
                stem = sl[5]
                augment = sl[6]
                ending = sl[7]
                accent_class = sl[8]
                form = re.sub(r'([͂́])(.*)́', r'\1\2',form.replace("̀", "́"))
                if pos.startswith('v'):
                    # This can be removed as soon as the passive perfect error in GLAUx/Morpheus is fixed
                    if pos[3] in ['r','l'] and pos[5] == 'p':
                        pos = pos[0:5] + 'm' + pos[6:]
                    if infl != '_':
                        if ending != '':
                            # No ending: irregular form
                            # Replacement: sometimes the stem is wrong...
                            if 'ss' in stem and 'ττ' in form:
                                stem = stem.replace('ss','tt')
                            stems = self.lemma_stems.get(lemma,{})
                            partType = self.get_part_type(pos)
                            stemFreqs = stems.get(partType,Counter())
                            stemFreqs[stem] += 1
                            stems[partType] = stemFreqs
                            self.lemma_stems[lemma] = stems
                            
                            part_stems = self.lemma_stem_inflClass.get(lemma,{})
                            stem_inflClass = part_stems.get(partType,{})
                            inflClass_freq = stem_inflClass.get(stem,Counter())
                            inflClass_freq[infl] += 1
                            stem_inflClass[stem] = inflClass_freq
                            part_stems[partType] = stem_inflClass
                            self.lemma_stem_inflClass[lemma] = part_stems
                            
                            pos_endings = self.inflClass_pos_endings.get(infl,{})
                            endings = pos_endings.get(pos,Counter())
                            endings[ending] += 1
                            pos_endings[pos] = endings
                            self.inflClass_pos_endings[infl] = pos_endings
                            
                            if pos[4]=='i' and pos[3] in ['a','i','l']:
                                stem_augments = self.lemma_stem_augments.get(lemma,{})
                                augment_freqs = stem_augments.get(stem,Counter())
                                augment_freqs[augment] += 1
                                stem_augments[stem] = augment_freqs
                                self.lemma_stem_augments[lemma] = stem_augments
                            
                                starting_sound = self.get_starting_sound(stem)
                                augmentStrat_freq = self.augment_strategies.get(starting_sound,Counter())
                                augmentStrat_freq[augment] += 1
                                self.augment_strategies[starting_sound] = augmentStrat_freq
                            
                            stem_prefix = self.lemma_stem_prefix.get(lemma,{})
                            prefix_comp = stem_prefix.get(stem)
                            if prefix_comp is not None and prefix_comp != prefix:
                                # print(f'Multiple prefix options {word[0]} {word[1]} {prefix} {parts[1]}')
                                pass
                            else:
                                if prefix != '':
                                    stem_prefix[stem] = prefix
                            self.lemma_stem_prefix[lemma] = stem_prefix
                            
                            principalPart_voices = self.voice_frequencies.get(lemma,{})
                            stem_voices = principalPart_voices.get(partType,{})
                            inflClass_voices = stem_voices.get(stem,{})
                            voice_frequency = inflClass_voices.get(infl,Counter())
                            voice_frequency[pos[5]] += 1
                            inflClass_voices[infl] = voice_frequency
                            stem_voices[stem] = inflClass_voices
                            principalPart_voices[partType] = stem_voices
                            self.voice_frequencies[lemma] = principalPart_voices
                elif pos[0] in ['n','a','l','p','m','d']:
                    if infl != '_':
                        if ending != '':
                            # No ending: irregular form
                            # Replacement for stem
                            if 'ss' in stem and 'ττ' in form:
                                stem = stem.replace('ss','tt')
                            if re.match(r'^\*?su',stem):
                                stem = stem.replace('s','c',1)
                            if ending == '*':
                                ending = ''
                            # ier-rule Attic
                            if infl in ['h_hs','a_hs','hs_ou'] and re.match(r'.*[ier]\+?_?$',stem):
                                infl = infl + '_ier'
                            stem_freqs = self.nomLemma_stems.get(lemma,Counter())
                            stem_freqs[stem] += 1
                            self.nomLemma_stems[lemma] = stem_freqs
                            
                            stem_conjClass = self.nomLemma_stem_conjClass.get(lemma,{})
                            conjClass_freqs = stem_conjClass.get(stem,Counter())
                            conjClass_freqs[infl] += 1
                            stem_conjClass[stem] = conjClass_freqs
                            self.nomLemma_stem_conjClass[lemma] = stem_conjClass
                            
                            stem_conjClass_accentClass = self.nomLemma_stem_conjClass_accentClass.get(lemma,{})
                            conjClass_accentClass = stem_conjClass_accentClass.get(stem,{})
                            accentClass_freqs = conjClass_accentClass.get(infl,Counter())
                            accentClass_split = accent_class.split(',')
                            for accent in accentClass_split:
                                accentClass_freqs[accent] += 1
                            conjClass_accentClass[infl] = accentClass_freqs
                            stem_conjClass_accentClass[stem] = conjClass_accentClass
                            self.nomLemma_stem_conjClass_accentClass[lemma] = stem_conjClass_accentClass
                            
                            pos_endings = self.conjClass_pos_endings.get(infl,{})
                            endings_freqs = pos_endings.get(pos,Counter())
                            endings_freqs[ending] += 1
                            pos_endings[pos] = endings_freqs
                            self.conjClass_pos_endings[infl] = pos_endings
                pos_forms = self.form_frequencies.get(lemma,{})
                form_freqs = pos_forms.get(pos,Counter())
                form_freqs[form] += 1
                pos_forms[pos] = form_freqs
                self.form_frequencies[lemma] = pos_forms
    
    def get_starting_sound(self,stem):
        starting_sound = 'consonant'
        if ')' in stem or '(' in stem:
            if re.match(r'\*[\)\(]',stem):
                starting_sound = re.sub(r'(\*[\)\(][aehiouw]+).*',r'\1',stem,1)    
            else:
                if stem.startswith('e)oi'):
                    starting_sound = 'e)oi'
                else:
                    starting_sound = re.sub(r'([\)\(]).*',r'\1',stem,1)
        elif len(stem) > 0 and stem[0] in ['l','s','n','d']:
            starting_sound = stem[0]
        return starting_sound
    
    def accent_syllable(self,syllable,word,isOptative=False,forceCircumflex=False):
        accentedWord = ''
        splittedWord = self.split_syllable_rough(word)
        lastLong = False
        last = splittedWord[len(splittedWord)-1]
        if (re.match(r'.*(h|w|au|eu|ou|ei|ui|\||_).*',last) or re.match(r'.*[oa]i[^$].*',last) or (re.match('.*[oa]i$',last) and isOptative)) and not '^' in last:
            lastLong = True
        for i, s in enumerate(splittedWord):
            if syllable == 0 or len(splittedWord) == 1:
                if i == len(splittedWord)-1:
                    if re.match(r'\*[\)\(](?!r).*',s):
                        if forceCircumflex:
                            accentedWord += re.sub(r'^\*([\)\(])',r'\*\1=',s,1)
                        else:
                            accentedWord += re.sub(r'^\*([\)\(])',r'\*\1/',s,1)
                    else:
                        if forceCircumflex:
                            accented_syllable = re.sub(r'([^aehiouw\+\)\(]*)$',r'=\1',s,1)
                            accented_syllable = re.sub(r'(h|w|a_)i=',r'\1=i',accented_syllable,1)
                            accentedWord += accented_syllable
                        else:
                            accented_syllable = re.sub(r'([^aehiouw\+\)\(]*)$',r'/\1',s,1)
                            accented_syllable = re.sub(r'(h|w|a_)i/',r'\1/i',accented_syllable,1)
                            accentedWord += accented_syllable
                else:
                    accentedWord += s
            elif syllable == 1 or len(splittedWord)==2 or lastLong:
                if i==len(splittedWord)-2:
                    if lastLong:
                        if re.match(r'\*[\)\(](?!r).*',s):
                            accentedWord += re.sub(r'^\*([\)\(])',r'\*\1/',s,1)
                        else:
                            accented_syllable = re.sub(r'([^aehiouw\+\)\(_]*)$',r'/\1',s,1)
                            accented_syllable = re.sub(r'(h|w|a_)i/',r'\1/i',accented_syllable,1)
                            accentedWord += accented_syllable
                    else:
                        if re.match(r'.*(h|w|au|eu|ou|ei|ui|\||ai|oi|_).*',s):
                            if re.match(r'\*[\)\(](?!r).*',s):
                                accentedWord += re.sub(r'^\*([\)\(])',r'\*\1=',s,1)
                            else:
                                accented_syllable = re.sub(r'([^aehiouw\+\)\(_]*)$',r'=\1',s,1)
                                accented_syllable = re.sub(r'(h|w|a_)i=',r'\1=i',accented_syllable,1)
                                accentedWord += accented_syllable
                        else:
                            if re.match(r'\*[\)\(](?!r).*',s):
                                accentedWord += re.sub(r'^\*([\)\(])',r'\*\1/',s,1)
                            else:
                                accented_syllable = re.sub(r'([^aehiouw\+\)\(_]*)$',r'/\1',s,1)
                                accented_syllable = re.sub(r'(h|w|a_)i/',r'\1/i',accented_syllable,1)
                                accentedWord += accented_syllable
                else:
                    accentedWord += s
            else:
                if i==len(splittedWord)-3:
                    if re.match(r'\*[\)\(](?!r).*',s):
                        accentedWord += re.sub(r'^\*([\)\(])',r'\*\1/',s,1)
                    else:
                        accented_syllable = re.sub(r'([^aehiouw\+\)\(_]*)$',r'/\1',s,1)
                        accented_syllable = re.sub(r'(h|w|a_)i/',r'\1/i',accented_syllable,1)
                        accentedWord += accented_syllable
                else:
                    accentedWord += s
        return accentedWord
    
    def split_syllable_rough(self,word):
        word = re.sub(r'([aehiouw]\|?\+?[_\^]?[\)\(]?)',r'\1€',word)
        word = re.sub(r'([aehouw][\^_]?)€i([_\^]?\|?[\)\(]?)€',r'\1i\2€',word)
        word = re.sub(r'([aehow][^_]?)€u([_\^]?[\)\(]?\|?)€',r'\1u\2€',word)
        word = re.sub(r'€([^aehiouw€]+)$',r'\1',word)
        word = re.sub('€$','',word)
        return word.split('€')
    
    def generate_stems(self,part,principalParts):
        presentStems = principalParts.get(part)
        if presentStems is None:
            print(f'{part} stem: not in corpus')
        else:
            presentStems = {k: v for k, v in sorted(presentStems.items(),key=lambda item: item[1],reverse=True)}
            stems_str = ', '.join([f'{k} ({v})' for k, v in presentStems.items()])
            print(f'{part} stem: {stems_str}')
    
    def find_verb_ending(self,lemma,pos,partType,stem):
        part_stem_infl_freq = self.lemma_stem_inflClass.get(lemma,{})
        stem_infl_freq = part_stem_infl_freq.get(partType,{})
        infl_freq = stem_infl_freq.get(stem,Counter())
        if len(infl_freq)>0:
            infl = max(infl_freq,key=infl_freq.get)
        else:
            infl = None
        if self.allow_voice_change:
            part_voice = self.voice_frequencies.get(lemma,{})
            stem_voice = part_voice.get(partType,{})
            infl_voice = stem_voice.get(stem,{})
            voice_freqs = infl_voice.get(infl,Counter())
            if self.most_frequent_voice:
                pos = pos[0:5]+max(voice_freqs,key=voice_freqs.get)+pos[6:]
            else:
                currentVoice = pos[5]
                if currentVoice not in voice_freqs:
                    newVoice = max(voice_freqs,key=voice_freqs.get)
                    if voice_freqs[newVoice] >= self.voice_change_threshold:
                        pos = pos[0:5]+newVoice+pos[6:]
        pos_endings = self.inflClass_pos_endings.get(infl,{})
        endings = pos_endings.get(pos)
        if endings is None:
            return None
        ending = max(endings,key=endings.get)
        return ending
    
    def find_noun_ending(self,lemma,pos,stem):
        stem_conjClass = self.nomLemma_stem_conjClass.get(lemma,{})
        conj_freq = stem_conjClass.get(stem,Counter())
        if len(conj_freq)>0:
            conj = max(conj_freq,key=conj_freq.get)
        else:
            conj = None
        pos_endings = self.conjClass_pos_endings.get(conj,None)
        endings = pos_endings.get(pos)
        if endings is None:
            return (None,None)
        ending = max(endings,key=endings.get)
        return (conj,ending)
    
    def get_most_frequent_noun_parts(self,lemma,pos):
        pos_freqs = self.form_frequencies.get(lemma)
        if pos_freqs is None:
            return (None,None,None,None)
        else:
            form_freqs = pos_freqs.get(pos)
            if form_freqs is not None:
                reduced_freqs = {k: v for k, v in form_freqs.items() if (not k.endswith('’') and ('͂' in k or '́' in k))}
                if len(reduced_freqs)>0:
                    form = max(reduced_freqs,key=reduced_freqs.get)
                else:
                    form = max(form_freqs,key=form_freqs.get)
                if not form.endswith('’'):
                    if form_freqs[form] >= self.min_form_freq:
                        return (form,None,None,None)
            stem = None
            accent_class = None
            stem_freqs = self.nomLemma_stems.get(lemma)
            if stem_freqs is None:
                return (None,None,None,None)
            ending = None
            for stem, _ in stem_freqs.most_common():
                conj_ending = self.find_noun_ending(lemma,pos,stem)
                ending = conj_ending[1]
                if ending is not None:
                    break
            if ending is None:
                return (None,None,None,None)
            stem_conjClass_accentClass = self.nomLemma_stem_conjClass_accentClass.get(lemma,{})
            conjClass_accentClass = stem_conjClass_accentClass.get(stem,{})
            accent_freqs = conjClass_accentClass.get(conj_ending[0])
            accent_class = max(accent_freqs,key=accent_freqs.get)
            return (stem,ending,accent_class,conj_ending[0])
    
    def get_most_frequent_verb_parts(self,lemma,pos):
        pos_freqs = self.form_frequencies.get(lemma)
        if pos_freqs is None:
            return (None,None,None,None)
        else:
            form_freqs = pos_freqs.get(pos)
            if form_freqs is not None:
                reduced_freqs = {k: v for k, v in form_freqs.items() if (not k.endswith('’') and ('͂' in k or '́' in k))}
                if len(reduced_freqs)>0:
                    form = max(reduced_freqs,key=reduced_freqs.get)
                else:
                    form = max(form_freqs,key=form_freqs.get)
                if not form.endswith('’'):
                    if form_freqs[form] >= self.min_form_freq:
                        return (None,None,form,None)
            stem = None
            partType = self.get_part_type(pos)
            part_stem_freq = self.lemma_stems.get(lemma)
            if part_stem_freq is None:
                return (None,None,None,None)
            else:
                stem_freq = part_stem_freq.get(partType)
                if stem_freq is not None:
                    stem = max(stem_freq,key=stem_freq.get)
            ending = None
            if stem is not None:
                for stem, _ in stem_freq.most_common():
                    ending = self.find_verb_ending(lemma,pos,partType,stem)
                    if ending is not None:
                        break
                if ending is None:
                    return (None,None,None,None)
            augment = None
            if stem is not None and pos[4] == 'i' and pos[3] in ['a','i','l']:
                stem_augments = self.lemma_stem_augments.get(lemma)
                if stem_augments is not None:
                    augments = stem_augments.get(stem)
                    if augments is not None:
                        augment = max(augments,key=augments.get)
                    else:
                        starting_sound = self.get_starting_sound(stem)
                        augment_freqs = self.augment_strategies.get(starting_sound)
                        if augment_freqs is None:
                            return(None,None,None,None)
                        augment = max(augment_freqs,key=augment_freqs.get)
                else:
                    starting_sound = self.get_starting_sound(stem)
                    augment_freqs = self.augment_strategies.get(starting_sound)
                    if augment_freqs is None:
                        return(None,None,None,None)
                    augment = max(augment_freqs,key=augment_freqs.get)
            prefix = None
            if stem is not None:
                stem_prefix = self.lemma_stem_prefix.get(lemma,{})
                prefix = stem_prefix.get(stem)
            return (prefix,augment,stem,ending)
    
    def get_part_type(self,pos):
        if pos[3] in ['p','i']:
            return 'present'
        elif pos[3] == 'a' and pos[5] in ['a','m']:
            return 'aorist'
        elif pos[3] in ['a','f'] and pos[5] == 'p':
            return 'aorist_future_passive'
        elif pos[3] == 'f' and pos[5] in ['a','m']:
            return 'future'
        elif pos[3] in ['r','l'] and pos[5] == 'a':
            return 'perfect'
        elif pos[3] in ['r','l'] and pos[5] in ['m','p']:
            # Error in GLAUx: some middle perfects are wrongly labeled as passive...
            return 'perfect_middle'
        elif pos[3] == 't':
            return 'future_perfect'
        else:
            return None
    
    def combine_prefices(self,prefices,stem):
        parts = prefices.copy()
        parts.append(stem)
        combined = ''
        for i, currentPart in enumerate(parts[:-1]):
            nextPart = parts[i+1]
            if currentPart in ['a)mfi/','ei)s','peri/','pro/','pro/s','u(pe/r']:
                currentPart = currentPart.replace('/','')
            elif currentPart in ['a)na/','dia/','para/']:
                if re.match('^[aehiouw].*',nextPart):
                    currentPart = currentPart[:-2]
                else:
                    currentPart = currentPart.replace('/','')
            elif currentPart in ['a)nti/','kata/','meta/']:
                if re.match('^[aehiouw].*',nextPart):
                    if '(' in nextPart:
                        currentPart = currentPart[:-3] + 'q'
                    else:
                        currentPart = currentPart[:-2]
                else:
                    currentPart = currentPart.replace('/','')
            elif currentPart in ['a)po/','e)pi/','u(po/']:
                if re.match('^[aehiouw].*',nextPart):
                    if '(' in nextPart:
                        currentPart = currentPart[:-3] + 'f'
                    else:
                        currentPart = currentPart[:-2]
                else:
                    currentPart = currentPart.replace('/','')
            elif currentPart == 'e)k':
                if re.match('^[aehiouw].*',nextPart):
                    currentPart = 'e)c'
            elif currentPart == 'e)n':
                if re.match('^[pbfy].*',nextPart):
                    currentPart = 'e)m'
                elif re.match('^[kgxc].*',nextPart):
                    currentPart = 'e)g'
            elif currentPart == 'su/n':
                if re.match('^[pbfym].*',nextPart):
                    currentPart = 'sum'
                elif re.match('^[kgxc].*',nextPart):
                    currentPart = 'sug'
                elif nextPart.startswith('l'):
                    currentPart = 'sul'
                elif nextPart.startswith('r'):
                    currentPart = 'sur'
                elif nextPart.startswith('s'):
                    currentPart = 'su'
                else:
                    currentPart = 'sun'
            if i != 0:
                currentPart = re.sub(r'[\(\)]','',currentPart)
            combined += currentPart
        stem = re.sub(r'[\(\)]','',stem)
        if re.match(r'^r[^r].*',stem) and re.match(r'.*[aehiouw]$',combined):
            stem = stem.replace('r','rr',1)
        combined += stem
        return combined
    
    def generate_noun_form(self,parts,pos):
        stem = parts[0]
        ending = parts[1]
        form = None
        if stem is None:
            return None
        elif ending is None:
            form = stem
            form = re.sub(r'[\^_]','',form)
            form = beta_code.beta_code_to_greek(form)
            if self.nfd:
                return ud.normalize(form,'NFD')
            else:
                return form
        else:
            if re.match('.*[/=].*',stem) or re.match('.*[/=].*',ending):
                form = stem + ending
                form = re.sub(r'[\^_]','',form)
                form = beta_code.beta_code_to_greek(form)
                if self.nfd:
                    return ud.normalize(form,'NFD')
                else:
                    return form
            else:
                forceCircumflex = False
                ncase = pos[7]
                number = pos[2]
                forceCircumflex = (
                    (ncase == 'g' and number in ['d','p']) or
                    (ncase == 'd' and number == 'd') or
                    (re.match('^(h_hs|a_hs|ah_ahs|eh_ehs|ehs_eou|os_h_on|eos_eh_eon|hs_ou|os_ou|os_on|oos_oon|oos_oou|oos_oh_oon)(_ier)?$',parts[3]) and ncase in['g','d']) or
                    (parts[3] in ['ah_ahs','eh_ehs','ehs_eou','eos_eou','hs_eos','klehs_kleous','oos_oh_oon','oos_oou','oos_oon']) or
                    (parts[3] in ['eus_ews','is_ews']) or
                    (parts[3] == 'aos_aou' and ncase=='d' and number=='s' and ending=='w') or
                    (parts[3] =='hs_es' and ending in ['a_','a_s','ei','ei+','eis','eus','h','ous']) or
                    (parts[3]=='us_eia_u' and ending in ['a_','ei','eis','h','hs','ous']) or
                    (parts[3]=='w_oos' and not (number=='s' and ncase in ['n','a'])) or
                    (parts[3]=='ws_oos' and not (number=='s' and ncase in ['n','v'])) or
                    (pos[0] == 'd'))
                # Sometimes these rules don't cover all cases. Still needs to be solved:
                # c_kos (e.g. γλαῦξ), n_nos (e.g. Σατᾶν), n_ntos (e.g. Κλειτοφῶν), r_ros (e.g. κῆρ), r_tos (e.g. στῆρ), s_dos (e.g. δαῖς) with a circumflex in the nominative
                # uLs_uos with a circumflex in several cases (e.g. ὀσφῦς)
                # ws_w_long with a circumflex in several cases (e.g. λαγῶς)
                # ws_w with a circumflex in several cases (e.g. Ἰεριχῶ)
                # In some dialects, the vocative on a_ or h of hs_ou has a circumflex (e.g. χρηστομαθῆ)
                accent = parts[2]
                conj = parts[3]
                if accent == 'suff_acc':
                    ending_split = self.split_syllable_rough(ending)
                    if ending == '':
                        ending_split = []
                    form = self.accent_syllable(len(ending_split)-1,stem+ending,False,forceCircumflex)
                elif accent == 'stem_acc':
                    if re.match('.*[aehiouw].*',ending):
                        ending_split = self.split_syllable_rough(ending)
                        if ending == '':
                            ending_split =  []
                        form = self.accent_syllable(len(ending_split),stem+ending,False,forceCircumflex)
                    else:
                        form = self.accent_syllable(0,stem+ending,False,forceCircumflex)
                else:
                    if conj in ['c_ggos','c_gos','c_kos','c_ktos','c_xos','y_bos','y_fos','y_pos'] and not re.match('.*[aehiouw].*',ending):
                        form = self.accent_syllable(1,stem+ending,False,forceCircumflex)
                    else:
                        form = self.accent_syllable(2,stem+ending,False,forceCircumflex)
                form = re.sub(r'[\^_]','',form)
                form = beta_code.beta_code_to_greek(form)
                if self.nfd:
                    return ud.normalize(form,'NFD')
                else:
                    return form
    
    def generate_verb_form(self,parts,pos):
        stem = parts[2]
        if stem is None:
            return None
        else:
            augment = parts[1]
            if augment is not None:
                if re.match('.*>.*',augment):
                    # Strange augment should be solved
                    if not augment == 'e(e>e(e':
                        augment_split = augment.split('>')
                        stem = augment_split[1] + stem[len(augment_split[0]):]
                else:
                    stem = augment + stem
            prefices = parts[0]
            if prefices is not None:
                stem = self.combine_prefices(prefices.split(','),stem)
            ending = parts[3]
            form = None
            if ending is not None:
                form = stem + ending
                if not re.match('.*[=/].*',form):
                    if pos[3] == 'a' and pos[4] == 'n' and pos[5] == 'a':
                        form = self.accent_syllable(1,form,False,False)
                    elif pos[3] == 'r' and pos[4] == 'n':
                        form = self.accent_syllable(1,form,False,False)
                    else:
                        form = self.accent_syllable(2,form,pos[4]=='o',False)
                form = form.replace('_','')
                form = beta_code.beta_code_to_greek(form)
                if self.nfd:
                    form = ud.normalize(form,'NFD')
            else:
                form = stem
        return form