from stanza.models.common.doc import UPOS, XPOS, TEXT, FEATS, LEMMA
from stanza.models.lemma.data import DataLoader
from stanza.models.lemma.vocab import Vocab, MultiVocab
from stanza.models.common.vocab import DeltaVocab
import random

class LemmaDataLoader(DataLoader):

    def __init__(self, doc, batch_size, args, vocab=None, evaluation=False, conll_only=False, skip=None, expand_unk_vocab=False, pos_name=UPOS, data=None):
        self.batch_size = batch_size
        self.args = args
        self.eval = evaluation
        self.shuffled = not self.eval
        self.doc = doc

        if data is None:
            data = self.raw_data(pos_name)

        if conll_only: # only load conll file
            return

        if skip is not None:
            assert len(data) == len(skip)
            data = [x for x, y in zip(data, skip) if not y]

        # handle vocab
        if vocab is not None:
            if expand_unk_vocab:
                pos_vocab = vocab['pos']
                char_vocab = DeltaVocab(data, vocab['char'])
                self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})
            else:
                self.vocab = vocab
        else:
            self.vocab = dict()
            char_vocab, pos_vocab = self.init_vocab(data)
            self.vocab = MultiVocab({'char': char_vocab, 'pos': pos_vocab})

        # filter and sample data
        if args.get('sample_train', 1.0) < 1.0 and not self.eval:
            keep = int(args['sample_train'] * len(data))
            data = random.sample(data, keep)

        data = self.preprocess(data, self.vocab['char'], self.vocab['pos'], args)
        # shuffle for training
        if self.shuffled:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)

        # chunk into batches
        data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def raw_data(self,pos_name=UPOS):
        return self.load_doc(self.doc, self.args.get('caseless', False), self.args.get('skip_blank_lemmas', False), self.eval, pos_name)
    
    @staticmethod
    def combine_feats(data):
        new_data = []
        for row in data:
            new_row = [row[0]]
            if row[2] is not None and row[1] is not None:
                new_row.append(f'{row[1]}_{row[2]}')
            else:
                new_row.append(row[1])
            new_row.append(row[3])
            new_data.append(new_row)
        return new_data
    
    @staticmethod
    def load_doc(doc, caseless, skip_blank_lemmas, evaluation, pos_name):
        data = doc.get([TEXT, pos_name, FEATS, LEMMA])
        data = LemmaDataLoader.combine_feats(data)
        data = DataLoader.resolve_none(data)
        if not evaluation and skip_blank_lemmas:
            data = DataLoader.skip_blank_lemmas(data)
        if caseless:
            data = DataLoader.lowercase_data(data)
        return data