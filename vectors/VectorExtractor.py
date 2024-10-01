import torch
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
from data.CONLLReader import CONLLReader
import json
from tokenization import Tokenization
from data import Datasets

class VectorExtractor:

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
    
    def get_embeddings(self,sentence,model):
        with torch.no_grad():
            if 'token_type_ids' in sentence:
                output = model(input_ids=sentence['input_ids'],token_type_ids=sentence['token_type_ids'],attention_mask=sentence['attention_mask'])
            else:
                output = model(input_ids=sentence['input_ids'],attention_mask=sentence['attention_mask'])
        states = output.hidden_states
        embeddings = torch.stack(states).squeeze().permute(1,0,2)
        sentence['embeddings'] = embeddings
        return sentence
    
    def get_vector(self,sentence,wid,layers=[-2],layer_combination_method='sum',subwords_combination_method='mean'):
        ids = []
        for no, subword_id in enumerate(sentence['wids_subwords']):
            if subword_id == wid:
                ids.append(no)
        vectors = []
        embeddings = sentence['embeddings']
        for i in ids:
            vectors.append(embeddings[i])
        if subwords_combination_method == 'mean':
            vector = torch.mean(torch.stack(vectors),dim=0)
        elif subwords_combination_method == 'first':
            vector = vectors[0]
        elif subwords_combination_method == 'last':
            vector = vectors[-1]
        vector_layers = []
        for layer in layers:
            vector_layers.append(vector[layer])
        if layer_combination_method == 'concatenate':
            return torch.cat(vector_layers,dim=0).numpy()
        elif layer_combination_method == 'sum':
            return torch.sum(torch.stack(vector_layers),dim=0).numpy()
    
    def extract_vectors(self,dataset,limit_wids=None,limit_labels=None,exclude_labels=None,label_name='MISC',layers=[-2],layer_combination_method='sum',subwords_combination_method='mean',average_same_id=False):
        vectors = {}
        for sent in dataset:
            for word_no, wid in enumerate(sent['wids']):
                include = True
                if limit_wids is not None:
                    if not wid in limit_wids:
                        include = False
                elif limit_labels is not None:
                    label = sent[label_name][word_no]
                    if not label in limit_labels:
                        include = False
                elif exclude_labels is not None:
                    label = sent[label_name][word_no]
                    if label in exclude_labels:
                        include = False
                if include:
                    vector = self.get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                    if not average_same_id:
                        vectors[wid] = vector
                    else:
                        if wid in vectors:
                            vectors[wid].append(vector)
                        else:
                            vectors[wid] = [vector]
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
    
    def build_dataset(self,wids,tokens,labels=None,label_name=None,normalization_rule=None):
        tokens_norm = tokens
        if normalization_rule is not None:
            tokens_norm = Tokenization.normalize_tokens(tokens, normalization_rule)
        if label_name is not None and labels is not None:
            dataset = Datasets.build_dataset(tokens_norm, {label_name:labels}, wids)
        else:
            dataset = Datasets.build_dataset(tokens_norm, None, wids)
        dataset = dataset.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":self.tokenizer,"return_tensors":'pt'})
        dataset = dataset.map(self.align_wids_subwords)
        if "token_type_ids" in dataset:
            dataset.set_format("pt", columns=["input_ids","token_type_ids","attention_mask"], output_all_columns=True)
        else:
            dataset.set_format("pt", columns=["input_ids","attention_mask"], output_all_columns=True)
        dataset = dataset.map(self.get_embeddings,fn_kwargs={"model":self.model})
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

    def __init__(self,transformer_path,tokenizer_path=None,data_path=None,data_preset='CONLLU',feature_cols=None,tokenizer_add_prefix_space=False):
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_path,add_prefix_space=tokenizer_add_prefix_space)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,add_prefix_space=tokenizer_add_prefix_space)
        self.reader = CONLLReader(data_preset,feature_cols)
        if data_path is not None:
            self.data = self.reader.parse_conll(data_path)
        self.model = AutoModel.from_pretrained(transformer_path,output_hidden_states=True)
        self.model.eval()

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
    
    extractor = VectorExtractor(transformer_path=args.transformer_path,tokenizer_path=args.tokenizer_path,data_path=args.data,data_preset=args.data_preset,feature_cols=feature_cols)
    if args.label_column is not None:
        wids, tokens, tags = extractor.reader.read_tokens(extractor.data,args.label_column,in_feats=False)
        dataset = extractor.build_dataset(wids,tokens,tags,args.label_column,args.normalization_rule)
    else:
        wids, tokens = extractor.reader.read_tokens(extractor.data, in_feats=False,return_tags=False)
        dataset = extractor.build_dataset(wids,tokens,None,None,args.normalization_rule)
    vectors = extractor.extract_vectors(dataset,limit_wids=limit_wids,limit_labels=limit_labels,exclude_labels=exclude_labels,label_name=args.label_column,layers=layers,layer_combination_method=args.layer_combination_method,subwords_combination_method=args.subwords_combination_method)
    extractor.write_vectors(vectors, args.output, wids, tokens, tags)
    