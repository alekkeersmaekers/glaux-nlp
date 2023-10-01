import numpy as np
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig, AdamW, TrainingArguments, Trainer
from data.CONLLReader import CONLLReader
from data.ClassificationDataset import ClassificationDataset
from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser
import json
from tokenization import Tokenization

class Classifier:
    
    def __init__(self,transformer_path,model_dir,tokenizer_path,training_data=None,test_data=None,ignore_label=None,unknown_label=None,data_preset='CONLL',feature_cols=None):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if tokenizer_path is None:
            self.tokenizer = AutoTokenizer.from_pretrained(transformer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
        
    def encode_tags(self, tags, encodings, tag2id, print_output):
        
        if self.ignore_label is not None:
            tag2id[self.ignore_label] = -100
        
        if self.unknown_label is not None and not self.unknown_label in tag2id:
            tag2id[self.unknown_label] = 0
        
        # SentencePiece uses a special character, (U+2581) 'lower one eighth block' to reconstruct spaces. Sometimes this character gets tokenized into its own subword, needing special control behavior.
        # Below, we set all seperately tokenized U+2581 to -100.
        prefix_subword_id = None
        if '▁' in self.tokenizer.vocab.keys():
            prefix_subword_id = self.tokenizer.convert_tokens_to_ids('▁')
        
        # Give tags to words instead of subwords
        labels = [[tag2id[tag] for tag in doc] for doc in tags]  # corresponding numbers of the different labels in the data set
        encoded_labels = []
        idx = 0

        # We use a subword tokenizer, so there are more tokens than labels. We need the offset mapping to link each label to the last subword of each original word.
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            initial_length = len(doc_labels)
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100  # we initialize an array with a length equal to the number of subwords, and fill it with -100, an unreachable number to avoid confusion.

            current_x, current_y, label_match_id = -1, -1, -1
            subword_counter = 0
            prefix_counter = 0
            if idx <= 5 and print_output:
                print(idx)
            
            if len(doc_offset) > 512:
                print("Offset wrong")
            
            for i, (x, y) in enumerate(doc_offset):  # We loop through the offset of each document to match the word endings.
                if i == len(doc_offset) - 1:  # Catches the edge case in which there are 512 subwords.
                    if (x, y) != (0, 0):
                        label_match_id += 1
                        doc_enc_labels[i] = doc_labels[label_match_id]
                        if idx <= 5 and print_output:
                            print(i, label_match_id, x, y)
                        subword_counter = 0
                else:
                    next_x, next_y = doc_offset[i + 1]

                    # Each new word starts with x = 0. Subsequent subwords follow the pattern y = next_x
                    # For example (0, 1) (1, 4) (4, 6)
                    # If a sentence does not need the full 512 subwords (most cases): the remaining ones are filled with (0,0): see edge case supra
                    
                    # Necessary for SentencePiece, see above.                    
                    if prefix_subword_id is not None and encodings[idx].ids[i] == prefix_subword_id:
                        doc_enc_labels[i] = -100
                        prefix_counter += 1

                    elif y != next_x and next_x == 0:
                        label_match_id += 1
                        doc_enc_labels[i] = doc_labels[label_match_id]  # Switches the initial -100 to the correct label.
                        if idx <= 5 and print_output:
                            print(i, label_match_id,self.tokenizer.decode(encodings.encodings[idx].ids[i - subword_counter:i + 1]),self.id2tag[doc_labels[label_match_id]])
                        subword_counter = 0

                    else:
                        subword_counter += 1

            result = 0
            for number in doc_enc_labels:  # Sanity check: the number of labels should be equal to the number of words at the end of sentence.
                if number != -100:
                    result += 1

            if initial_length != result:
                logging.log(0, f"Result doesn't match length at {idx}")

            encoded_labels.append(doc_enc_labels.tolist())

            idx += 1

        return encoded_labels
    
    def train_classifier(self,wids,tokens,tags,output_model,epochs=3,batch_size=16):
        unique_tags = set(tag for doc in tags for tag in doc)
        if self.ignore_label is not None:
            unique_tags.remove(self.ignore_label)
        tag2id = {tag: s_id for s_id, tag in enumerate(sorted(unique_tags))}
        id2tag = {s_id: tag for tag, s_id in tag2id.items()}
        encodings = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=512, is_split_into_words=True, return_offsets_mapping=True)
        labels = self.encode_tags(tags, encodings, tag2id, print_output=False)
        encodings.pop("offset_mapping")
        train_dataset = ClassificationDataset(encodings,labels,wids)
                
        self.config = AutoConfig.from_pretrained(self.transformer_path, num_labels=len(unique_tags), id2label=id2tag, label2id=tag2id)
        self.classifier_model = AutoModelForTokenClassification.from_pretrained(self.transformer_path,config=self.config)
        
        self.classifier_model.to(self.device)
        self.classifier_model.train()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optim = AdamW(self.classifier_model.parameters(), lr=5e-5)
        
        for epoch in range(epochs):
        
            for idx, batch in enumerate(train_loader):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.classifier_model(input_ids, attention_mask=attention_mask, labels=labels)
        
                loss = outputs[0]
                loss.backward()
                print("| epoch {:3d} | {:5d}/{:5d} batches")

        self.classifier_model.save_pretrained(output_model)
    
    def predict(self,tokens,tags,wids,model_dir=None,batch_size=16):
        if model_dir is not None:
            self.classifier_model = AutoModelForTokenClassification.from_pretrained(model_dir).to(self.device)
            self.config = AutoConfig.from_pretrained(model_dir)
        
        tag2id = self.config.label2id
        id2tag = self.config.id2label
        
        encodings = self.tokenizer(tokens, padding='max_length', truncation=True, max_length=512, is_split_into_words=True, return_offsets_mapping=True)        
        labels = self.encode_tags(tags, encodings, tag2id, print_output=False)
        encodings.pop("offset_mapping")
        dataset = ClassificationDataset(encodings, labels, wids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        preds_total = []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.classifier_model(input_ids, attention_mask=attention_mask, labels=labels)
                for batch_idx, batch_piece in enumerate(labels):
                    for label_idx, label in enumerate(batch_piece):
                        if label != -100:
                            softmaxed_predictions = torch.nn.functional.softmax(outputs[1][batch_idx][label_idx],
                                                                                -1).tolist()
                            preds = {}
                            for index, pred in enumerate(softmaxed_predictions):
                                preds[id2tag[index]] = pred
                            preds_total.append(preds)
        return preds_total
    
    def write_prediction(self,wids,tokens,tags,preds,output_file,output_format):
        with open(output_file, 'w', encoding='UTF-8') as outfile:
            if output_format == 'tab':
                outfile.write("id\ttoken\tgold\tprediction\tprobability\n")
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
                            outfile.write(wid + "\t" + word + "\t" + tag +"\t" + top_prediction[0] + "\t" + f"{top_prediction[1]:.5f}"+"\n")
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
            wids, tokens, tags = classifier.reader.read_tags('MISC', classifier.training_data, False)
            tokens_norm = tokens
            if args.normalization_rule is not None:
                tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
            classifier.train_classifier(wids,tokens_norm,tags,classifier.model_dir,args.epochs,args.batch_size)
    elif args.mode == 'test':
        if args.test_data == None:
            print('Test data is missing')
        else:
            classifier = Classifier(args.transformer_path,args.model_dir,args.tokenizer_path,args.training_data,args.test_data,args.ignore_label,args.unknown_label,args.data_preset,feature_cols)
            wids, tokens, tags = classifier.reader.read_tags('MISC', classifier.test_data, False)
            tokens_norm = tokens
            if args.normalization_rule is not None:
                tokens_norm = Tokenization.normalize_tokens(tokens, args.normalization_rule)
            prediction = classifier.predict(tokens_norm,tags,wids,model_dir=classifier.model_dir,batch_size=args.batch_size)
            if args.output_file is not None:
                classifier.write_prediction(wids,tokens,tags,prediction,args.output_file,args.output_format)
                