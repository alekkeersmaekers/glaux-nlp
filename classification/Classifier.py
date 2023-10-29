import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorForTokenClassification
from data.CONLLReader import CONLLReader
import torch
from argparse import ArgumentParser
import json
from tokenization import Tokenization
from data import Datasets

class Classifier:
    
    def __init__(self,transformer_path,model_dir,tokenizer_path,training_data=None,test_data=None,ignore_label=None,unknown_label=None,data_preset='CONLL',feature_cols=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.prefix_subword_id = None
        if '▁' in self.tokenizer.vocab.keys():
            self.prefix_subword_id = self.tokenizer.convert_tokens_to_ids('▁')
        self.transformer_path = transformer_path
        self.model_dir = model_dir
        self.reader = CONLLReader(data_preset,feature_cols)
        if training_data is not None:
            self.training_data = self.reader.parse_conll(training_data)
        else:
            self.training_data = None
        if test_data is not None:
            self.test_data = self.reader.parse_conll(test_data)
        else:
            self.test_data = None
        self.ignore_label = ignore_label
        self.unknown_label = unknown_label

    
    def id_label_mappings(self,tags):
        unique_tags = set(tag for doc in tags for tag in doc)
        if self.ignore_label is not None:
            unique_tags.remove(self.ignore_label)
        tag2id = {tag: s_id for s_id, tag in enumerate(sorted(unique_tags))}
        id2tag = {s_id: tag for tag, s_id in tag2id.items()}
        return tag2id, id2tag
    
    def get_valid_subwords(self, offset, input_ids, last_subword=True, prefix_subword_id=None):
        valid_subwords = []
        for i, current_offset in enumerate(offset):
            x = current_offset[0]
            y = current_offset[1]
            if last_subword:
                if i == len(offset) - 1:
                    if (x, y) != (0, 0): # Catches the edge case in which there are 512 subwords.
                        valid_subwords.append(True)
                    else:
                        valid_subwords.append(False)
                else:
                    # Each new word starts with x = 0. Subsequent subwords follow the pattern y = next_x
                    # For example (0, 1) (1, 4) (4, 6)
                    # If a sentence does not need the full 512 subwords (most cases): the remaining ones are filled with (0,0): see edge case supra
                    next_offset = offset[i + 1]
                    next_x = next_offset[0]
                        
                    # Necessary for SentencePiece: if a prefix has been tokenized in a separate token, it has offset mapping (0,1) while the next one still has (0,6). As a consequence, y != next_x while the token still does not need a label.
                    if prefix_subword_id is not None and input_ids[i] == prefix_subword_id:
                        valid_subwords.append(False)
        
                    elif y != next_x and next_x == 0:
                        valid_subwords.append(True)
                    else:
                        valid_subwords.append(False)
            else:
                # First subword strategy is substantially easier: all subwords at the beginning have offset mapping 0, !0
                if x == 0 and y != 0:
                    # One exception: the prefix subword of SentencePiece also has 0, !0 (i.e. 0,1)
                    if prefix_subword_id is not None and input_ids[i] == prefix_subword_id:
                        valid_subwords.append(False)
                    else:
                        valid_subwords.append(True)
                else:
                    valid_subwords.append(False)
        return valid_subwords
    
    def align_labels(self, sentence, tag2id, last_subword=True, prefix_subword_id=None, labelname='MISC', is_tensor=False):
        labels = sentence[labelname]
        offset = sentence['offset_mapping']
        input_ids = sentence['input_ids']
        
        if is_tensor:
            offset = offset[0]
            input_ids = input_ids[0]
        
        valid_subwords = self.get_valid_subwords(offset,input_ids,last_subword=last_subword,prefix_subword_id=prefix_subword_id)
        
        enc_labels = np.ones(len(valid_subwords),dtype=int) * - 100
        
        label_match_id = -1
        
        for n_subword, valid_subword in enumerate(valid_subwords):
            if valid_subword:
                label_match_id += 1
                label = labels[label_match_id]
                if not (self.ignore_label is not None and self.ignore_label==label):
                    enc_labels[n_subword] = tag2id[label]
        
        sentence['labels'] = enc_labels
        return sentence
        
    def train_classifier(self,output_model,train_dataset,tag2id,id2tag,epochs=3,batch_size=16,learning_rate=5e-5):        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)        
        self.config = AutoConfig.from_pretrained(self.transformer_path, num_labels=len(tag2id), id2label=id2tag, label2id=tag2id)
        self.classifier_model = AutoModelForTokenClassification.from_pretrained(self.transformer_path,config=self.config)
        
        training_args = TrainingArguments(output_dir=output_model,num_train_epochs=epochs,per_device_train_batch_size=batch_size,learning_rate=learning_rate,save_strategy='no')
        trainer = Trainer(model=self.classifier_model,args=training_args,train_dataset=train_dataset,tokenizer=self.tokenizer,data_collator=data_collator)
        trainer.train()
        trainer.model.save_pretrained(save_directory=trainer.args.output_dir)
            
    def predict(self,test_data,model_dir=None,batch_size=16,labelname='MISC'):        
        ##Only works when padding is set to the right!!! See below
        self.classifier_model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.config = AutoConfig.from_pretrained(model_dir)
        id2tag = self.config.id2label
        tag2id = self.config.label2id
        
        if self.unknown_label is not None:
            tag2id[self.unknown_label] = 0
        
        test_data = test_data.map(self.align_labels,fn_kwargs={"prefix_subword_id":self.prefix_subword_id,"tag2id":tag2id,"labelname":labelname})
        
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        training_args = TrainingArguments(output_dir=self.model_dir,per_device_eval_batch_size=batch_size)
        trainer = Trainer(model=self.classifier_model,args=training_args,tokenizer=self.tokenizer,data_collator=data_collator)
        
        predictions = trainer.predict(test_data)
        softmaxed_predictions = torch.nn.functional.softmax(torch.from_numpy(predictions.predictions),dim=-1).tolist()
        
        # This only works when padding is set to the right, since the padded predictions will be longer than valid_subword
        all_preds = []
        for sent_no, sent in enumerate(test_data):
            valid_subwords = self.get_valid_subwords(sent['offset_mapping'],sent['input_ids'],prefix_subword_id=self.prefix_subword_id)
            for subword_no, valid_subword in enumerate(valid_subwords):
                if valid_subword:
                    # If a (sub)word has the label defined by ignore_label, it is also set to -100 with classifier.align_labels, even though it is counted as a valid subword
                    if not ('labels' in sent and sent['labels'][subword_no] == -100):
                        preds = {}
                        for prob_n, prob in enumerate(softmaxed_predictions[sent_no][subword_no]):
                            classname = id2tag[prob_n]
                            preds[classname] = prob
                        all_preds.append(preds)
        
        return all_preds
        
    
    def write_prediction(self,wids,tokens,tags,preds,output_file,output_format,output_sentence=True):
        with open(output_file, 'w', encoding='UTF-8') as outfile:
            if output_format == 'tab':
                outfile.write("id\ttoken\tgold\tprediction\tprobability")
                if output_sentence:
                    outfile.write("\tsentence")
                outfile.write('\n')
            word_no = -1
            for sent_id, sent in enumerate(tokens):
                for word_id, word in enumerate(sent):
                    wid = wids[sent_id][word_id]
                    tag = tags[sent_id][word_id]
                    if self.ignore_label is not None and tag == self.ignore_label:
                        if output_format == 'CONLL':
                            outfile.write(wid + "\t" + word + "\t_\t_\t_\t_\t_\t_\t_\t_\n")
                        elif output_format == 'simple':
                            outfile.write(wid + "\t" + word + "\t_\n")
                    else:
                        word_no += 1
                        top_prediction = sorted(preds[word_no].items(), reverse=True, key=lambda x: x[1])[0]
                        if output_format == 'CONLL':
                            outfile.write(wid + "\t" + word + "\t_\t_\t_\t_\t_\t_\t_\t"+top_prediction[0]+"\n")
                        elif output_format == 'simple':
                            outfile.write(wid + "\t" + word + "\t"+top_prediction[0]+"\n")
                        elif output_format == 'tab':
                            outfile.write(wid + "\t" + word + "\t" + tag +"\t" + top_prediction[0] + "\t" + f"{top_prediction[1]:.5f}")
                            if output_sentence:
                                sent_str = ''
                                for word_id_2, word_2 in enumerate(sent):
                                    if word_id_2 == word_id:
                                        sent_str+='['
                                    sent_str+= word_2
                                    if word_id_2 == word_id:
                                        sent_str+=']'
                                    sent_str+= ' '
                                outfile.write('\t'+sent_str.strip())
                            outfile.write('\n')
                if output_format == 'CONLL' or output_format == 'simple':
                    outfile.write('\n')
                        
if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('mode',help='train/test')
    arg_parser.add_argument('transformer_path',help='path to the transformer model')
    arg_parser.add_argument('model_dir',help='path of classifier model')
    arg_parser.add_argument('--tokenizer_path',help='path to the tokenizer (defaults to the path of the transformer model)')
    arg_parser.add_argument('--training_data',help='classifier training data')
    arg_parser.add_argument('--test_data',help='classifier test data')
    arg_parser.add_argument('--output_file',help='classified data')
    arg_parser.add_argument('--output_format',help='format of the output data: CONLL (standard CONLL, with prediction in MISC), simple (CONLL-style with only columns ID, FORM and MISC=prediction) or tab (tabular format, with prediction probabilities, no sentence boundaries and without irrelevant tokens)',default='CONLL')
    arg_parser.add_argument('--ignore_label',help='all tokens with this tag will be ignored during training or classification')
    arg_parser.add_argument('--unknown_label',help='tag in the test data for tokens for which we do not know the label beforehand')
    arg_parser.add_argument('--data_preset',help='format of the data, defaults to CONLL (other option: simple, where the data has columns ID, FORM, MISC)',default='CONLL')
    arg_parser.add_argument('--feature_cols',help='define a custom format for the data, e.g. {"ID":0,"FORM":2,"MISC":3}')
    arg_parser.add_argument('--normalization_rule',help='normalize tokens during training/testing, normalization rules implemented are greek_glaux and standard NFD/NFKD/NFC/NFKC')
    arg_parser.add_argument('--epochs',help='number of epochs for training, defaults to 3',type=int,default=3)
    arg_parser.add_argument('--batch_size',help='batch size for training/testing, defaults to 16',type=int,default=16)
    arg_parser.add_argument('--learning_rate',help='learning rate for training, defaults to 5e-5',type=float,default=5e-5)
    args = arg_parser.parse_args()
    feature_cols = args.feature_cols
    if feature_cols is not None:
        # Only required for Eclipse
        feature_cols = json.loads(feature_cols.replace('\'','"'))
        # Other
        # feature_cols = json.loads(feature_cols)
    if args.mode == 'train':
        if args.training_data == None:
            print('Training data is missing')
        else:
            classifier = Classifier(args.transformer_path,args.model_dir,args.tokenizer_path,args.training_data,args.test_data,args.ignore_label,args.unknown_label,args.data_preset,feature_cols)
            tokens, tags = classifier.reader.read_tags('MISC', classifier.training_data, in_feats=False,return_wids=False)
            tag_dict = {'MISC':tags}
            tokens_norm = tokens
            if args.normalization_rule is not None:
                tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
            tag2id, id2tag = classifier.id_label_mappings(tags)
            training_data = Datasets.build_dataset(tokens_norm,tag_dict)
            training_data = training_data.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":classifier.tokenizer})
            training_data = training_data.map(classifier.align_labels,fn_kwargs={"prefix_subword_id":classifier.prefix_subword_id,"tag2id":tag2id})
            classifier.train_classifier(classifier.model_dir,training_data,tag2id=tag2id,id2tag=id2tag,epochs=args.epochs,batch_size=args.batch_size,learning_rate=args.learning_rate)
    elif args.mode == 'test':
        if args.test_data == None:
            print('Test data is missing')
        else:
            classifier = Classifier(args.transformer_path,args.model_dir,args.tokenizer_path,args.training_data,args.test_data,args.ignore_label,args.unknown_label,args.data_preset,feature_cols)
            wids, tokens, tags = classifier.reader.read_tags('MISC', classifier.test_data, False)
            tags_dict = {'MISC':tags}
            tokens_norm = tokens
            if args.normalization_rule is not None:
                tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
            test_data = Datasets.build_dataset(tokens,tags_dict)
            test_data = test_data.map(Tokenization.tokenize_sentence,fn_kwargs={"tokenizer":classifier.tokenizer})
            prediction = classifier.predict(test_data,model_dir=classifier.model_dir,batch_size=args.batch_size)
            if args.output_file is not None:
                classifier.write_prediction(wids,tokens,tags,prediction,args.output_file,args.output_format)
                