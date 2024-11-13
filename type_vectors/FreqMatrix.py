from treebanks.NodeOperations import children, coordinationHeadUD, highestCoordinateUD
from tqdm import tqdm
import math

class FreqMatrix():

    def __init__(self):
        self.matrix = {}
        self.wordFrequencies = {}
        self.contextFrequencies = {}
        self.contextFeats = []
    
    def reduceMatrix(self,min_wordFreq=0,min_contextFreq=0):
        newMatrix = FreqMatrix()
        newMatrix.wordFrequencies = self.wordFrequencies.copy()
        newMatrix.contextFrequencies = self.contextFrequencies.copy()
        for contextFeat in self.contextFeats:
            if self.contextFrequencies[contextFeat] >= min_contextFreq:
                newMatrix.contextFeats.append(contextFeat)
        for key, val in self.matrix.items():
            if self.wordFrequencies[key] >= min_wordFreq:
                freqs = {}
                for key2, val2 in val.items():
                    if key2 in newMatrix.contextFeats:
                        freqs[key2] = val2
                newMatrix.matrix[key] = freqs
        return newMatrix
    
    def generateLemmaFeat(self,token,relation=None,prefix=None,arc_direction=True,synt_relation=False,conjunct_relation='CO'):
        lemma = token['lemma']
        feat = lemma
        if arc_direction:
            if relation == conjunct_relation:
                prefix = 'Cd'
            if synt_relation:
                feat = prefix + '_' + relation + '_' + lemma
            else:
                feat = prefix + '_' + lemma
        return feat
    
    def addFeatToContextFeats(self,feat):
        contextFreq = self.contextFrequencies.get(feat,0)
        contextFreq += 1
        self.contextFrequencies[feat] = contextFreq
        if not feat in self.contextFeats:
            self.contextFeats.append(feat)
    
    def updateContexts(self,wordContexts,feat):
        self.addFeatToContextFeats(feat)
        contextFreq = wordContexts.get(feat,0)
        contextFreq += 1
        wordContexts[feat] = contextFreq
    
    def addSyntacticContexts(self,token,tokens,wordContexts,arc_direction=True,synt_relation=False,conjunct_relation='CO'):
        # Feature extraction: children
        for c in children(token,tokens):
            if not 'artificial' in c and c['morph']['pos'] != 'PUNCT' and c['morph']['pos'] != 'GAP' and len(c['lemma'])>0:
                feat = self.generateLemmaFeat(c,relation=c['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='C',conjunct_relation=conjunct_relation)
                self.updateContexts(wordContexts,feat)
                # If preposition or conjunction, add children of the preposition/conjunction also to the children
                if c['relation'] == 'AuxP' or c['relation'] == 'AuxC':
                    for c2 in children(c,tokens):
                        if not 'artificial' in c2 and c2['morph']['pos'] != 'PUNCT' and c2['morph']['pos'] != 'GAP':
                            feat = self.generateLemmaFeat(c2,relation=c2['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='C',conjunct_relation=conjunct_relation)
                            self.updateContexts(wordContexts,feat)
        
        # Feature extraction: head
        if token['head'] is not None and not 'artificial' in token['head'] and not token['head']['morph']['pos'] == 'PUNCT' and not token['head']['morph']['pos'] == 'GAP' and len(token['head']['lemma'])>0:
            # For prepositions and conjunctions, we add two features: e.g. H_AuxP_εἰς and H_OBJ_ἔρχομαι. The second one happens a little lower.
            if token['head']['relation'] == 'AuxP' or token['head']['relation'] == 'AuxC':
                feat = self.generateLemmaFeat(token['head'],relation=token['head']['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
            else:
                feat = self.generateLemmaFeat(token['head'],relation=token['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
            self.updateContexts(wordContexts,feat)
            # Coordination
            if token['relation'] == conjunct_relation:
                # This is the head of the first coordinate
                coHead = coordinationHeadUD(token,conjunct_relation)
                if coHead is not None and len(coHead['lemma']) > 0:
                    # This is the first coordinate
                    firstCo = highestCoordinateUD(token,conjunct_relation)
                    if coHead['relation'] == 'AuxP' or coHead['relation'] == 'AuxC':
                        # Again, we add two heads for prepositions/conjunctions (see above)
                        feat = self.generateLemmaFeat(coHead,relation=coHead['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                    else:
                        feat = self.generateLemmaFeat(coHead,relation=firstCo['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                    self.updateContexts(wordContexts,feat)
                    if coHead['relation'] == 'AuxP' or coHead['relation'] == 'AuxC':
                        realHead = coHead['head']
                        if realHead is not None and len(realHead['lemma']) > 0:
                            feat = self.generateLemmaFeat(realHead,relation=firstCo['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                            self.updateContexts(wordContexts,feat)
            # Here the second head feature gets added (see above)
            if token['head']['relation'] == 'AuxP' or token['head']['relation'] == 'AuxC':
                head2 = token['head']['head']
                if head2 is not None and len(head2['lemma']) > 0:
                    feat = self.generateLemmaFeat(head2,relation=token['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                    self.updateContexts(wordContexts,feat)
    
    def loadDependencies(self,treebank,wordclass=None,arc_direction=True,synt_relation=False,prefix=None):
        for sent in treebank:
            tokens = sent['tokens']
            for token in tokens:
                if not 'artificial' in token and (wordclass is None or token['morph']['pos']==wordclass) and len(token['lemma'])>0:
                    targetLemma = token['lemma']
                    if prefix is not None:
                        targetLemma = prefix + '_' + targetLemma
                    wordFreq = self.wordFrequencies.get(targetLemma,0)
                    wordFreq+=1
                    self.wordFrequencies[targetLemma] = wordFreq
                    wordContexts = self.matrix.get(targetLemma,{})
                    self.addSyntacticContexts(token,tokens,wordContexts,arc_direction=arc_direction,synt_relation=synt_relation)
                    self.matrix[targetLemma] = wordContexts
    
    def calcPPMI(self,smoothing=1):
        ppmiMatrix = FreqMatrix()
        ppmiMatrix.contextFeats = self.contextFeats.copy()
        ppmiMatrix.wordFrequencies = self.wordFrequencies.copy()
        ppmiMatrix.contextFrequencies = self.contextFrequencies.copy()
        wordFreq_total = 0
        for i in self.wordFrequencies.values():
            wordFreq_total += i
        for key, val in tqdm(self.matrix.items(),'Calculating PPMIs'):
            ppmis = {}
            for key2, val2 in val.items():
                observed = val2 / wordFreq_total
                expected = (self.wordFrequencies[key] / wordFreq_total) * pow(self.contextFrequencies[key2]/wordFreq_total,smoothing)
                ppmi = 0
                if observed > expected:
                    ppmi = math.log(observed/expected,2)
                ppmis[key2] = ppmi
            ppmiMatrix.matrix[key] = ppmis
        return ppmiMatrix
    