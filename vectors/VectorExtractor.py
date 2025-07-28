import torch
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
from data.CONLLReader import CONLLReader
import json
from tokenization import Tokenization
from data import Datasets

class VectorExtractor:

    def process_sentence(self,sentence):
        sentence.update(Tokenization.tokenize_sentence(sentence,tokenizer=self.tokenizer,return_tensors='pt'))
        sentence = self.align_wids_subwords(sentence)
        sentence = self.get_embeddings(sentence)
        return sentence
    
    def process_batch(self,batch):
        batch.update(Tokenization.tokenize_batch(batch, tokenizer=self.tokenizer, return_tensors='pt'))
        batch = self.align_wids_subwords_batch(batch)
        batch = self.get_embeddings_batch(batch)
        return batch
    
    def align_wids_subwords(self,sentence):
        wids = sentence['wids']
        subword_ids = sentence['subword_ids']
        wids_subwords = []
        id = -1
        previous_subword_id = -1
        for subword_id in subword_ids:
            if subword_id is None:
                wids_subwords.append('-1')
            else:
                if subword_id != previous_subword_id and subword_id is not None:
                    id+=1
                wids_subwords.append(wids[id])
                previous_subword_id = subword_id
        sentence['wids_subwords'] =  wids_subwords
        return sentence
    
    def align_wids_subwords_batch(self,batch):
        batch_wids_subwords = []
        for wids, subword_ids in zip(batch['wids'], batch['subword_ids']):
            wids_subwords = []
            id = -1
            previous_subword_id = -1
            for subword_id in subword_ids:
                if subword_id is None:
                    wids_subwords.append('-1')
                else:
                    if subword_id != previous_subword_id and subword_id is not None:
                        id += 1
                    wids_subwords.append(wids[id])
                    previous_subword_id = subword_id
            batch_wids_subwords.append(wids_subwords)
        batch['wids_subwords'] = batch_wids_subwords
        return batch
    
    def get_embeddings(self,sentence):
        with torch.no_grad():
            input_ids = sentence['input_ids'].to(self.model.device)
            attention_mask = sentence['attention_mask'].to(self.model.device)
            token_type_ids = sentence['token_type_ids'].to(self.model.device) if 'token_type_ids' in sentence else None
            output = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        hidden_states = output.hidden_states
        kept_states = torch.stack([hidden_states[i] for i in self.layers])
        embeddings = kept_states[:, 0, :, :].permute(1,0,2).cpu()
        if self.limit_wids is not None or self.limit_labels is not None or self.exclude_labels is not None:
            wids_subwords = sentence['wids_subwords']
            if self.limit_wids is not None:
                keep_wids = self.limit_wids
            elif self.limit_labels is not None or self.exclude_labels is not None:
                wids = sentence['wids']
                labels = sentence[self.label_name]
                if self.limit_labels is not None:
                    keep_wids = [wids[i] for i in range(0,len(wids)) if labels[i] in self.limit_labels]
                else:
                    keep_wids = [wids[i] for i in range(0,len(wids)) if labels[i] not in self.exclude_labels]
            reduced_states = [embeddings[i] for i in range(0,len(embeddings)) if wids_subwords[i] in keep_wids]
            vector_wids = [wids_subwords[i] for i in range(0,len(wids_subwords)) if wids_subwords[i] in keep_wids]
        else:
            wids_subwords = sentence['wids_subwords']
            reduced_states = [embeddings[i] for i in range(0,len(embeddings)) if wids_subwords[i] != '-1']
            vector_wids = [wids_subwords[i] for i in range(0,len(wids_subwords)) if wids_subwords[i] != '-1']
        sentence['embeddings'] = reduced_states
        sentence['vector_wids'] = vector_wids
        return sentence
    
    def get_embeddings_batch(self,batch):
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.model.device)
            attention_mask = batch['attention_mask'].to(self.model.device)
            token_type_ids = batch['token_type_ids'].to(self.model.device) if 'token_type_ids' in batch else None
            output = self.model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,output_hidden_states=True)
        hidden_states = output.hidden_states
        kept_states = torch.stack([hidden_states[i] for i in self.layers])
        kept_states = kept_states.permute(1,2,0,3).cpu()
        reduced_states = []
        vector_wids = []
        if self.limit_wids is not None or self.limit_labels is not None or self.exclude_labels is not None:
            for sent_no, sent_embeddings in enumerate(kept_states):
                wids_subwords = batch['wids_subwords'][sent_no]
                if self.limit_wids is not None:
                    keep_wids = self.limit_wids
                elif self.limit_labels is not None or self.exclude_labels is not None:
                    wids = batch['wids'][sent_no]
                    labels = batch[self.label_name][sent_no]
                    if self.limit_labels is not None:
                        keep_wids = [wids[i] for i in range(0,len(wids)) if labels[i] in self.limit_labels]
                    else:
                        keep_wids = [wids[i] for i in range(0,len(wids)) if labels[i] not in self.exclude_labels]
                reduced_states.append([sent_embeddings[i] for i in range(0,len(sent_embeddings)) if wids_subwords[i] in keep_wids])
                vector_wids.append([wids_subwords[i] for i in range(0,len(wids_subwords)) if wids_subwords[i] in keep_wids])
        else:
            for sent_no, sent_embeddings in enumerate(kept_states):
                wids_subwords = batch['wids_subwords'][sent_no]
                reduced_states.append([sent_embeddings[i] for i in range(0,len(sent_embeddings)) if wids_subwords[i] != '-1'])
                vector_wids.append([wids_subwords[i] for i in range(0,len(wids_subwords)) if wids_subwords[i] != '-1'])
        batch['embeddings'] = reduced_states
        batch['vector_wids'] = vector_wids
        return batch
    
    def get_vector(self,vectors):
        if self.subwords_combination_method == 'mean':
            if len(vectors) > 1:
                vector = torch.mean(torch.stack(vectors),dim=0)
            else:
                vector = vectors[0]
        elif self.subwords_combination_method == 'first':
            vector = vectors[0]
        elif self.subwords_combination_method == 'last':
            vector = vectors[-1]
        if self.layer_combination_method == 'concatenate':
            return vector.flatten().numpy()
        elif self.layer_combination_method == 'sum':
            return torch.sum(vector,dim=0).numpy()
    
    def extract_vectors(self,dataset,average_same_id=False):
        vectors = {}
        for sent in dataset:
            wid_embeddings = {}
            for wid_no, wid in enumerate(sent['vector_wids']):
                embeddings = wid_embeddings.get(wid,[])
                embeddings.append(sent['embeddings'][wid_no])
                wid_embeddings[wid] = embeddings
            for wid, embeddings in wid_embeddings.items():
                vector = self.get_vector(embeddings)
                if not average_same_id:
                    vectors[wid] = vector
                else:
                    wid_vecs = vectors.get(wid,[])
                    wid_vecs.append(vector)
                    vectors[wid] = wid_vecs
        if not average_same_id:
            return(vectors)
        else:
            averaged_vectors = {}
            for wid, all_vectors in vectors.items():
                length = len(all_vectors)
                transposed = list(zip(*all_vectors))
                averages = [sum(elements) / length for elements in transposed]
                averaged_vectors[wid] = averages
            return averaged_vectors
    
    def build_dataset(self,wids,tokens,labels=None,normalization_rule=None,batched=False,batch_size=1000):
        tokens_norm = tokens
        if normalization_rule is not None:
            tokens_norm = Tokenization.normalize_tokens(tokens, normalization_rule)
        if self.label_name is not None and labels is not None:
            dataset = Datasets.build_dataset(tokens_norm, {self.label_name:labels}, wids)
        else:
            dataset = Datasets.build_dataset(tokens_norm, None, wids)
        if not batched:
            dataset = dataset.map(self.process_sentence)
        else:
            dataset = dataset.map(self.process_batch,batched=True,batch_size=batch_size)
        dataset.set_format("pt", columns=["embeddings"], output_all_columns=True)
        return dataset
        
    def write_vectors(self,vectors,output_file,ids,tokens=None,labels=None,precision=5):
        with open(output_file, 'w', encoding='UTF-8') as outfile:
            for sentence_no, sentence in enumerate(ids):
                for word_no, word in enumerate(sentence):
                    if word in vectors:
                        outfile.write(word)
                        if tokens is not None:
                            outfile.write('\t'+tokens[sentence_no][word_no])
                        if labels is not None:
                            outfile.write('\t'+labels[sentence_no][word_no])
                        vector = vectors[word]
                        for element in vector:
                            outfile.write("\t"+("{:0."+str(precision)+"f}").format(element))
                        outfile.write('\n')

    def __init__(self,transformer_path,tokenizer_path=None,data_path=None,data_preset='CONLLU',feature_cols=None,tokenizer_add_prefix_space=False,layers=range(1,13),limit_wids=None,limit_labels=None,exclude_labels=None,label_name='labels',layer_combination_method='sum',subwords_combination_method='mean'):
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_path,add_prefix_space=tokenizer_add_prefix_space)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,add_prefix_space=tokenizer_add_prefix_space)
        self.reader = CONLLReader(data_preset,feature_cols)
        if data_path is not None:
            self.data = self.reader.parse_conll(data_path)
        self.model = AutoModel.from_pretrained(transformer_path,output_hidden_states=True)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        self.layers = layers
        self.limit_wids = limit_wids
        self.limit_labels = limit_labels
        self.exclude_labels = exclude_labels
        self.label_name = label_name
        self.layer_combination_method = layer_combination_method
        self.subwords_combination_method = subwords_combination_method

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('transformer_path',help='path to the transformer model')
    arg_parser.add_argument('data',help='path to the data')
    arg_parser.add_argument('output',help='output file')
    arg_parser.add_argument('--tokenizer_path',help='path to the tokenizer (defaults to the path of the transformer model)')
    arg_parser.add_argument('--data_preset',help='format of the data, defaults to CONLLU (other option: simple, where the data has columns ID, FORM, MISC)',default='CONLLU')
    arg_parser.add_argument('--feature_cols',help='define a custom format for the data, e.g. {"ID":0,"FORM":2,"MISC":3}')
    arg_parser.add_argument('--normalization_rule',help='normalize tokens during training/testing, normalization rules implemented are greek_glaux and standard NFD/NFKD/NFC/NFKC')
    arg_parser.add_argument('--label_column',help='column name that includes labels, if you want to add them to the output file (otherwise don\'t specify)')
    arg_parser.add_argument('--limit_wids',help='only generate vectors when word has one of these wids (separated by commas)')
    arg_parser.add_argument('--limit_labels',help='only generate vectors when word has one of these labels (separated by commas)')
    arg_parser.add_argument('--exclude_labels',help='opposite of limit labels, only generate vectors when word does not have one of these labels (separated by commas)')
    arg_parser.add_argument('--layers',help='layers you want to extract (separated by commas)',default='2')
    arg_parser.add_argument('--layer_combination_method',help='method to combine multiple layers (sum/concatenate)',default='sum')
    arg_parser.add_argument('--subwords_combination_method',help='method to combine vectors of multiple subwords (mean/first/last)',default='mean')
    args = arg_parser.parse_args()
    feature_cols = args.feature_cols
    if feature_cols is not None:
        # Only required for Eclipse
        feature_cols = json.loads(feature_cols.replace('\'','"'))
        # Other
        # feature_cols = json.loads(feature_cols)
    limit_labels = None
    if args.limit_labels is not None:
        limit_labels = args.limit_labels.split(',')
    limit_wids = None
    if args.limit_wids is not None:
        limit_wids = args.limit_wids.split(',')
    exclude_labels = None
    if args.exclude_labels is not None:
        exclude_labels = args.exclude_labels.split(',')
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(',')]
    
    extractor = VectorExtractor(transformer_path=args.transformer_path,tokenizer_path=args.tokenizer_path,data_path=args.data,data_preset=args.data_preset,feature_cols=feature_cols,layers=layers,limit_wids=limit_wids,limit_labels=limit_labels,exclude_labels=exclude_labels,label_name=args.label_column,layer_combination_method=args.layer_combination_method,subwords_combination_method=args.subwords_combination_method)
    if args.label_column is not None:
        wids, tokens, tags = extractor.reader.read_tokens(extractor.data,args.label_column,in_feats=False)
        dataset = extractor.build_dataset(wids,tokens,tags,args.label_column,args.normalization_rule)
    else:
        wids, tokens = extractor.reader.read_tokens(extractor.data, in_feats=False,return_tags=False)
        dataset = extractor.build_dataset(wids,tokens,None,None,args.normalization_rule)
    vectors = extractor.extract_vectors(dataset)
    extractor.write_vectors(vectors, args.output, wids, tokens, tags)
    