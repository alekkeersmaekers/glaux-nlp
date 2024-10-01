# Code based on Muller et al.

from lemmatization.Counter import Counter
from lemmatization.Match import Match
from lemmatization.MatchNode import MatchNode
from lemmatization.ReplaceNode import ReplaceNode
import  re

class EditTreeBuilder:
    
    def __init__(self,max_depth):
        self.counter = Counter()
        self.max_depth = max_depth
    
    def clearCache(self):
        self.cache = {}
        
    def getCacheKey(self,input_start,input_end,output_start,output_end):
        s = ""
        s += str(hex(input_start))
        s += ' '
        s += str(hex(input_end))
        s += ' '
        s += str(hex(output_start))
        s += ' '
        s += str(hex(output_end))
        return s
    
    def retrieveFromCache(self,input_start,input_end,output_start,output_end):
        key = self.getCacheKey(input_start,input_end,output_start,output_end)
        return self.cache.get(key)
    
    def build_base(self,input,output):
        self.clearCache()
        return self.build(input,0,len(input),output,0,len(output),0)
    
    def build(self,input, input_start, input_end, output, output_start, output_end, depth):
        best_tree = self.retrieveFromCache(input_start,input_end,output_start,output_end)
        if best_tree is not None:
            return best_tree
        if self.max_depth < 0 or depth < self.max_depth:
            for match in self.longestMatches(input, input_start, input_end, output, output_start, output_end):
                
                left = None
                if input_start < match.input_start or output_start < match.output_start:
                    left = self.build(input,input_start,match.input_start,output,output_start,match.output_start, depth+1)
                
                right = None
                if match.input_end < input_end or match.output_end < output_end:
                    right = self.build(input,match.input_end,input_end,output,match.output_end,output_end,depth+1)
                    
                left_input_length = match.input_start - input_start
                right_input_length = input_end - match.input_end
                
                tree = MatchNode(left,right,left_input_length,right_input_length)
                
                if best_tree is None or tree.getCost(self) < best_tree.getCost(self):
                    best_tree = tree
        
        if best_tree is None:
            best_tree = ReplaceNode(input[input_start:input_end], output[output_start:output_end])
        
        self.addToCache(input_start, input_end, output_start, output_end, best_tree)
        return best_tree
    
    def addToCache(self, input_start, input_end, output_start, output_end, tree):
        key = self.getCacheKey(input_start, input_end, output_start, output_end)
        self.cache[key] = tree
    
    def longestMatches(self, input, input_start, input_end, output, output_start, output_end):
        longest_matches = []
        for m_input_start in range(input_start,input_end):
            for m_output_start in range(output_start,output_end):
                length = 0
                while True:
                    i = m_input_start + length
                    if i >= input_end:
                        break
                    o = m_output_start + length
                    if o >= output_end:
                        break
                    if input[i] != output[o]:
                        break
                    length+=1
                if length > 0:
                    if len(longest_matches) == 0 or longest_matches[0].length <= length:
                        if not len(longest_matches) == 0 and longest_matches[0].length < length:
                            longest_matches.clear()
                        longest_matches.append(Match(m_input_start,m_output_start,length))
        return longest_matches

def main2():
    builder = EditTreeBuilder(-1)
    input = r'C:\Users\u0111778\OneDrive - KU Leuven\Glaux\1.0\nlp\Lemming\data_training_norm.txt'
    #input = r'C:\Users\u0111778\OneDrive - KU Leuven\Colleges\Computerlinguistiek voor klassieke talen\Lemmatisering\\data_test_lemmatized.txt'
    output = r'C:\Users\u0111778\OneDrive - KU Leuven\Colleges\Computerlinguistiek voor klassieke talen\Lemmatisering\data_training_edittrees.txt'
    with open(input,'r',encoding='UTF-8') as infile:
        with open(output, 'w', encoding='UTF-8') as outfile:
            raw_text = infile.read().strip()
            raw_sents = re.split(r'\n\n', raw_text)
            for sent in raw_sents:
                words = re.split(r'\n', sent)
                for word in words:
                    split = re.split(r'\t',word)
                    form = split[1]
                    lemma = split[2]
                    tree = builder.build_base(form,lemma)
                    outfile.write(form+'\t'+lemma+'\t'+str(tree)+'\n')

def main():
    print('Got here')
    lexicon = {}
    file = open("C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/lexicon.txt")
    raw_text = file.read().strip()
    entries = re.split(r'\n', raw_text)
    for entry in entries:
        split = re.split('\t',entry)
        if split[2] == 'conjunction' or split[2] == 'coordinator' or split[2] == 'preposition' or split[2] == 'adverb' or split[2] == 'particle' or split[2] == 'interjection':
            split[2] = 'function_word'
        index = split[0] + split[2] + split[3]
        if index in lexicon:
            lemmas = lexicon[index]
            if not split[1] in lemmas:
                lemmas.append(split[1])
        else:
            lemmas = []
            lemmas.append(split[1])
            lexicon[index] = lemmas
    edit_trees = {}
    builder = EditTreeBuilder(-1)
    print(str(builder.build_base('ἔδοξεν','δοκέω')))
    #form_new = "χίλιοι";
    #print(tree)
    #print(tree.apply(form_new,0,len(form_new)))
    #count = 0
   # file = open("C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/LemmatizationData_Canonical_Reg.txt")   
    #raw_text = file.read().strip()
    #raw_sents = re.split(r'\n\n', raw_text)
    #for sent in raw_sents:
    #    words = re.split(r'\n', sent)
    #    for word in words:
    #        count+=1
    #        if count % 100000 == 0:
    #            print(count)
    #        split = re.split(r'\t',word)
    #        form = split[1]
    #        lemma = split[2]
    #        pos = split[4]
    #        morph = split[5]
    #        ref = split[8]
    #        index = form + pos + morph
    #        tree = builder.build_base(form,lemma)
    #        freq = 0
    #        if tree in edit_trees:
    #            freq = edit_trees[tree]
    #        freq+=1
    #        edit_trees[tree] = freq
    
    #print(len(edit_trees))
    #reduced_trees = set()
    #for tree,freq in edit_trees.items():
    #    if freq>=5:
    #        reduced_trees.add(tree)
    
    #print(len(reduced_trees))
    
    #lemmas = {}
    
    #file = open("C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/PosLemma_Freq.txt")
    #raw_text = file.read().strip()
    #raw_lemma_freqs = re.split(r'\n', raw_text)
    #for lemma_freq in raw_lemma_freqs:
    #    lemma_freq_s = re.split('\t',lemma_freq)
    #    lemmas[lemma_freq_s[0]] = int(lemma_freq_s[1])
    
    #unknown_forms = 0
    
    #file = open("C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/LemmatizationData_Canonical_Reg.txt")    
    #output = "C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/LemmaData_UnknownLemmas.txt"
    #with open(output, 'w', encoding='UTF-8') as outfile:
    #    raw_text = file.read().strip()
    #    raw_sents = re.split(r'\n\n', raw_text)
    #    for sent in raw_sents:
    #        words = re.split(r'\n', sent)
    #        for word in words:
    #            split = re.split(r'\t',word)
    #            form = split[1]
    #            lemma = split[2]
    #            pos = split[4]
    #            if pos == 'conjunction' or pos == 'coordinator' or pos == 'preposition' or pos == 'adverb' or pos == 'particle' or pos == 'interjection':
    #                pos = 'function_word'
    #            morph = split[5]
    #            ref = split[8]
    #            index = form + pos + morph
                
    #            len_lexicon = 0
    #            if index in lexicon:
    #                len_lexicon = len(lexicon[index])
                    
                #tree = builder.build_base(form,lemma)
    #            pred_lemma = '_'
    #            if len_lexicon == 0:
    #                unknown_forms+=1
    #                if unknown_forms % 1000==0:
    #                    print(unknown_forms)
    #                best_lemma_freq = 0
    #                best_lemma = ''
    #                for tree in edit_trees.keys():
    #                    poss_lemma = tree.apply(form,0,len(form))
    #                    if poss_lemma is not None and poss_lemma in lemmas:
    #                        freq = lemmas[poss_lemma]
    #                        if freq > best_lemma_freq:
    #                            best_lemma = poss_lemma
    #                            best_lemma_freq = freq
    #                pred_lemma = best_lemma
    #                outfile.write(form+'\t'+lemma+'\t'+pos+'\t'+morph+'\t'+ref+'\t'+pred_lemma+'\n')
    
    #lemmas = {}
    #file = open("C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/AllForms_Freq.txt")
    #raw_text = file.read().strip()
    #raw_form_freqs = re.split(r'\n', raw_text)
    #count = 0
    #for form_freq in raw_form_freqs:
    #    count+=1
    #    if count % 10000 == 0:
    #        print(count)
    #    form_freq_s = re.split('\t',form_freq)
    #    form = form_freq_s[0]
    #    freq = int(form_freq_s[1])
    #    for tree in reduced_trees:
    #        lemma = tree.apply(form,0,len(form))
    #        if lemma is not None:
    #            lemma_freq = 0
    #            if lemma in lemmas:
    #                lemma_freq = lemmas[lemma]
    #            lemma_freq += freq
    #            lemmas[lemma] = lemma_freq
    
    #output = "C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/PosLemma_Freq.txt"
    #with open(output, 'w', encoding='UTF-8') as outfile:
    #    for lemma,freq in lemmas.items():
    #        outfile.write(lemma+'\t'+str(freq)+'\n')
    
    #output = "C:/Users/u0111778/Documents/NLP/Tests_0223/Lemmatization/LemmaData.txt"
    #with open(output, 'w', encoding='UTF-8') as outfile:
    #    raw_text = file.read().strip()
    #    raw_sents = re.split(r'\n\n', raw_text)
    #    for sent in raw_sents:
    #        words = re.split(r'\n', sent)
    #        for word in words:
    #            split = re.split(r'\t',word)
    #            form = split[1]
    #            lemma = split[2]
    #            pos = split[4]
                #if pos == 'conjunction' or pos == 'coordinator' or pos == 'preposition' or pos == 'adverb' or pos == 'particle' or pos == 'interjection':
                #    pos = 'function_word'
    #            morph = split[5]
    #            ref = split[8]
    #            index = form + pos + morph
                
    #            len_lexicon = 0
    #            if index in lexicon:
    #                len_lexicon = len(lexicon[index])
                    
    #            tree = builder.build_base(form,lemma)
    #            outfile.write(form+'\t'+lemma+'\t'+pos+'\t'+morph+'\t'+str(tree)+'\t'+str(len_lexicon)+'\t'+ref+'\n')
        


    #tree = builder.build_base(form,lemma)
    #print(str(tree))
    
if __name__ == '__main__':
    main()