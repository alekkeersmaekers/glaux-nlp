from treebanks.NodeOperations import children, coordinationHeadUD, highestCoordinateUD
from tqdm import tqdm
import math
from collections import defaultdict, Counter
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import numpy as np
import pandas as pd

def generateLemmaFeat(token,relation=None,prefix=None,arc_direction=True,synt_relation=False,conjunct_relation='CO'):
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

def validToken(token):
    return (
        "artificial" not in token
        and token["morph"]["pos"] not in {"PUNCT", "GAP"}
        and len(token["lemma"]) > 0
    )

class FreqMatrix():

    def __init__(self):
        self.matrix = defaultdict(Counter)
        self.wordFrequencies = Counter()
        self.contextFrequencies = Counter()
        self.contextFeats = set()
    
    def reduceMatrix(self,min_wordFreq=0,min_contextFreq=0):
        newMatrix = FreqMatrix()
        newMatrix.wordFrequencies = self.wordFrequencies.copy()
        newMatrix.contextFrequencies = self.contextFrequencies.copy()
        for contextFeat in self.contextFeats:
            if self.contextFrequencies[contextFeat] >= min_contextFreq:
                newMatrix.contextFeats.add(contextFeat)
        for key, val in self.matrix.items():
            if self.wordFrequencies[key] >= min_wordFreq:
                freqs = {}
                for key2, val2 in val.items():
                    if key2 in newMatrix.contextFeats:
                        freqs[key2] = val2
                if len(freqs) > 0:
                    # We also remove words that don't have any context words (anymore) from the matrix, even if they exceed min_wordFreq
                    newMatrix.matrix[key] = freqs
        newMatrix.setNames()
        return newMatrix
    
    def addFeatToContextFeats(self,feat):
        self.contextFrequencies[feat] += 1
        self.contextFeats.add(feat)
    
    def updateContexts(self,wordContexts,feat):
        self.addFeatToContextFeats(feat)
        wordContexts[feat] += 1
            
    def addSyntacticContexts(self,token,tokens,wordContexts,arc_direction=True,synt_relation=False,conjunct_relation='CO'):
        # Feature extraction: children
        for c in children(token,tokens):
            if validToken(c):
                feat = generateLemmaFeat(c,relation=c['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='C',conjunct_relation=conjunct_relation)
                self.updateContexts(wordContexts,feat)
                # If preposition or conjunction, add children of the preposition/conjunction also to the children
                if c['relation'] == 'AuxP' or c['relation'] == 'AuxC':
                    for c2 in children(c,tokens):
                        if validToken(c2):
                            feat = generateLemmaFeat(c2,relation=c2['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='C',conjunct_relation=conjunct_relation)
                            self.updateContexts(wordContexts,feat)
        
        # Feature extraction: head
        head = token.get("head")
        if head is not None and validToken(head):
            # For prepositions and conjunctions, we add two features: e.g. H_AuxP_εἰς and H_OBJ_ἔρχομαι. The second one happens a little lower.
            relation = head["relation"] if head["relation"] in {"AuxP", "AuxC"} else token["relation"]
            feat = generateLemmaFeat(head,relation=relation,arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
            self.updateContexts(wordContexts,feat)
            # Coordination
            if token['relation'] == conjunct_relation:
                # This is the head of the first coordinate
                coHead = coordinationHeadUD(token,conjunct_relation)
                if coHead is not None and validToken(coHead):
                    # This is the first coordinate
                    firstCo = highestCoordinateUD(token,conjunct_relation)
                    relation = coHead["relation"] if coHead["relation"] in {"AuxP", "AuxC"} else firstCo["relation"]
                    feat = generateLemmaFeat(coHead,relation=relation,arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                    self.updateContexts(wordContexts,feat)
                    if coHead['relation'] == 'AuxP' or coHead['relation'] == 'AuxC':
                        realHead = coHead['head']
                        if realHead is not None and validToken(realHead):
                            feat = generateLemmaFeat(realHead,relation=firstCo['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                            self.updateContexts(wordContexts,feat)
            # Here the second head feature gets added (see above)
            if head['relation'] == 'AuxP' or head['relation'] == 'AuxC':
                head2 = head['head']
                if head2 is not None and validToken(head2):
                    feat = generateLemmaFeat(head2,relation=token['relation'],arc_direction=arc_direction,synt_relation=synt_relation,prefix='H',conjunct_relation=conjunct_relation)
                    self.updateContexts(wordContexts,feat)
    
    def loadDependencies(self,treebank,wordclass=None,arc_direction=True,synt_relation=False,prefix=None,show_progress=False):
        iterator = tqdm(treebank) if show_progress else treebank
        for sent in iterator:
            tokens = sent['tokens']
            for token in tokens:
                if not 'artificial' in token and (wordclass is None or token['morph']['pos']==wordclass) and len(token['lemma'])>0:
                    targetLemma = f"{prefix}_{token['lemma']}" if prefix else token["lemma"]
                    self.wordFrequencies[targetLemma] += 1
                    wordContexts = self.matrix[targetLemma]
                    self.addSyntacticContexts(token,tokens,wordContexts,arc_direction=arc_direction,synt_relation=synt_relation)
        self.setNames()
    
    def calcPPMI(self,smoothing=1):
        ppmiMatrix = FreqMatrix()
        ppmiMatrix.contextFeats = self.contextFeats.copy()
        ppmiMatrix.wordFrequencies = self.wordFrequencies.copy()
        ppmiMatrix.contextFrequencies = self.contextFrequencies.copy()
        wordFreq_total = sum(self.wordFrequencies.values())
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
        ppmiMatrix.rows = self.rows
        ppmiMatrix.cols = self.cols
        return ppmiMatrix
    
    def setNames(self):
        matrix_rows = self.matrix.keys()
        matrix_cols = {col for c in self.matrix.values() for col in c}
        # This makes sure that it is sorted by frequency, following the counter order. Not really necessary, but I just prefer it for the visual representation of the matrix.
        rows = [x for (x,_) in self.wordFrequencies.most_common() if x in matrix_rows]
        cols = [x for (x,_) in self.contextFrequencies.most_common() if x in matrix_cols]
        self.rows = rows
        self.cols = cols
    
    def toMatrix(self):
        row_idx = {r: i for i, r in enumerate(self.rows)}
        col_idx = {c: i for i, c in enumerate(self.cols)}
        data, row_ind, col_ind = [], [], []
        for word, contexts in self.matrix.items():
            for context, val in contexts.items():
                row_ind.append(row_idx[word])
                col_ind.append(col_idx[context])
                data.append(val)
        sparse_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(len(self.rows), len(self.cols)))
        return sparse_matrix
    
    def calcSVD(self,dim=300,sparse_matrix=None,return_help_matrices=False):
        if sparse_matrix is None:
            sparse_matrix = self.toMatrix()
        w, s, c = svds(sparse_matrix, k=dim)
        w = np.flip(w, axis=1)
        s = np.flip(s)
        c = np.flip(c, axis=0)
        reduced_matrix = csr_matrix(np.dot(w, np.diag(np.sqrt(s))))
        if return_help_matrices:
            return reduced_matrix, w, s, c
        else:
            return reduced_matrix
        
    def toDataFrame(self,sparse_matrix=None,is_reduced=False):
        if sparse_matrix is None:
            sparse_matrix = self.toMatrix()
        if is_reduced:
            # If SVD is applied, the columns don't have names
            df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,index=self.rows)
        else:
            df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix,index=self.rows,columns=self.cols)
        return df