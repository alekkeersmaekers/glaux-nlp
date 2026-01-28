class Scorer:
    
    def __init__(self, results, gold_sentences):
        self.results = results
        self.gold_sentences = gold_sentences
        
    def get_accuracy(self,score_threshold,pos=None,grc_pos_dict=None):
        total = 0
        correct = 0
        for result in self.results:
            if grc_pos_dict is not None:
                grc_pos = grc_pos_dict[f'{result[1]}_{result[3]}']
            if grc_pos_dict is None or pos is None or pos == grc_pos:
                total += 1
                if result[2] < score_threshold:
                    if result[8] == True:
                        correct +=1
                else:
                    if result[5] == 1:
                        correct +=1
        return correct / total, correct, total
    
    def get_standard_metrics(self,threshold):
        gold_alignments = []
        for sent in self.gold_sentences:
            sent_gold_alignments = []
            for grc, en in sent['alignment'].items():
                for en_index in en:
                    sent_gold_alignments.append((grc,en_index))
            gold_alignments.append(sent_gold_alignments)
        pred_alignments = []
        currentPred = []
        prevSent = -1
        for row_index, row in enumerate(self.results):
            if row[1] != prevSent and row_index != 0:
                pred_alignments.append(currentPred.copy())
                currentPred = []
            prevSent = row[1]
            currentIndex = row[3]
            if row[2] > threshold:
                for en_index in row[4]:
                    currentPred.append((currentIndex,en_index))
        pred_alignments.append(currentPred)
        true_positives = 0
        false_positives = 0
        for sent_no, sent in enumerate(pred_alignments):
            for alignment in sent:
                if alignment in gold_alignments[sent_no]:
                    true_positives += 1
                else:
                    false_positives += 1
        precision = (true_positives) / (true_positives + false_positives)
        true_positives = 0
        false_negatives = 0
        for sent_no, sent in enumerate(gold_alignments):
            for alignment in sent:
                if alignment in pred_alignments[sent_no]:
                    true_positives += 1
                else:
                    false_negatives += 1
        recall = (true_positives) / (true_positives + false_negatives)
        f1 = (2 * precision * recall) / (precision + recall)
        true_positives = 0
        total_alignments = 0
        for sent_no, sent in enumerate(pred_alignments):
            total_alignments += len(sent)
            total_alignments += len(gold_alignments[sent_no])
            for alignment in sent:
                if alignment in gold_alignments[sent_no]:
                    true_positives += 1
        aer = 1 - ((2*true_positives)/(total_alignments))
        return precision, recall, f1, aer