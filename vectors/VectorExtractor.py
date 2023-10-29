import torch
from transformers import AutoTokenizer, AutoModel
from argparse import ArgumentParser
from data.CONLLReader import CONLLReader
import json
from tokenization import Tokenization
from data import Datasets

class VectorExtractor:

    def align_wids_subwords(self,sentence, prefix_subword_id=None):
        wids = sentence['wids']
        input_ids = sentence['input_ids'][0]
        offset = sentence['offset_mapping'][0]
        wids_subwords = []
        id = -1
        for i, current_offset in enumerate(offset):
            x = current_offset[0]
            y = current_offset[1]
            if(x==0 and y!=0):
                id+=1
                wids_subwords.append(wids[id])
            elif(x==0 and y==0):
                wids_subwords.append('-1')
            elif(x!=0):
                wids_subwords.append(wids[id])
            if prefix_subword_id is not None and input_ids[i] == prefix_subword_id:
                id-=1
        sentence['wids_subwords'] =  wids_subwords
        return sentence
    
    def get_embeddings(self,sentence,model):
        with torch.no_grad():
            output = model(input_ids=sentence['input_ids'],token_type_ids=sentence['token_type_ids'],attention_mask=sentence['attention_mask'])
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
    
    def extract_vectors(self,dataset,output_file,limit_wids=None,limit_labels=None,label_name='MISC',layers=[-2],layer_combination_method='sum',subwords_combination_method='mean'):
        with open(output_file, 'w', encoding='UTF-8') as outfile:
            for sent in dataset:
                if limit_wids is not None:
                    for word_no, wid in enumerate(sent['wids']):
                        if wid in limit_wids:
                            vector = self.get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                            if label_name is not None:
                                outfile.write(wid+'\t'+sent[label_name][word_no])
                            else:
                                outfile.write(wid)
                            for element in vector:
                                outfile.write("\t"+"{:0.5f}".format(element))
                            outfile.write('\n')
                elif limit_labels is not None:
                    for word_no, wid in enumerate(sent['wids']):
                        label = sent[label_name][word_no]
                        if label in limit_labels:
                            vector = self.get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                            if label_name is not None:
                                outfile.write(wid+'\t'+label)
                            else:
                                outfile.write(wid)
                            for element in vector:
                                outfile.write("\t"+"{:0.5f}".format(element))
                            outfile.write('\n')
                else:
                    for word_no, wid in enumerate(sent['wids']):
                        vector = self.get_vector(sent,wid,layers=layers,layer_combination_method=layer_combination_method,subwords_combination_method=subwords_combination_method)
                        if label_name is not None:
                            outfile.write(wid+'\t'+sent[label_name][word_no])
                        else:
                            outfile.write(wid)
                        for element in vector:
                            outfile.write("\t"+"{:0.5f}".format(element))
                        outfile.write('\n')

    def __init__(self,transformer_path,tokenizer_path,data_path=None,data_preset='CONLL',feature_cols=None):
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.prefix_subword_id = None
        if '▁' in self.tokenizer.vocab.keys():
            self.prefix_subword_id = self.tokenizer.convert_tokens_to_ids('▁')
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
    arg_parser.add_argument('--data_preset',help='format of the data, defaults to CONLL (other option: simple, where the data has columns ID, FORM, MISC)',default='CONLL')
    arg_parser.add_argument('--feature_cols',help='define a custom format for the data, e.g. {"ID":0,"FORM":2,"MISC":3}')
    arg_parser.add_argument('--normalization_rule',help='normalize tokens during training/testing, normalization rules implemented are greek_glaux and standard NFD/NFKD/NFC/NFKC')
    arg_parser.add_argument('--label_column',help='column name that includes labels, if you want to add them to the output file (otherwise don\'t specify)')
    arg_parser.add_argument('--limit_wids',help='only generate vectors when word has one of these wids (separated by commas)')
    arg_parser.add_argument('--limit_labels',help='only generate vectors when word has one of these labels (separated by commas)')
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
    if args.layers is not None:
        layers = [int(x) for x in args.layers.split(',')]
    extractor = VectorExtractor(transformer_path=args.transformer_path,tokenizer_path=args.tokenizer_path,data_path=args.data,data_preset=args.data_preset,feature_cols=feature_cols)
    if args.label_column is not None:
        wids, tokens, tags = extractor.reader.read_tags(args.label_column,extractor.data,in_feats=False)
        tokens_norm = tokens
        if args.normalization_rule is not None:
            tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
        dataset = Datasets.build_dataset(tokens_norm, {args.label_column:tags}, wids)
    else:
        wids, tokens = extractor.reader.read_tags(None,extractor.data,in_feats=False,return_tags=False)
        tokens_norm = tokens
        if args.normalization_rule is not None:
            tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
        dataset = Datasets.build_dataset(tokens_norm, None, wids)
    dataset = dataset.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":extractor.tokenizer,"return_tensors":'pt'})
    dataset = dataset.map(extractor.align_wids_subwords,fn_kwargs={"prefix_subword_id":extractor.prefix_subword_id})
    dataset.set_format("pt", columns=["input_ids","token_type_ids","attention_mask"], output_all_columns=True)
    dataset = dataset.map(extractor.get_embeddings,fn_kwargs={"model":extractor.model})
    extractor.extract_vectors(dataset,args.output,limit_wids=limit_wids,limit_labels=limit_labels,label_name=args.label_column,layers=layers,layer_combination_method=args.layer_combination_method,subwords_combination_method=args.subwords_combination_method)
    