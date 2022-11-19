# -*- coding: UTF-8 -*-

from data.CONLLReader import CONLLReader
from data.AGPoSDataset import AGPoSDataset
from torch.utils.data import DataLoader
import numpy as np
import logging
import torch
import re
from transformers import AdamW, ElectraForTokenClassification
from tokenization.Tokenization import fix_accents
import unicodedata as ud

class Tagger:

    def encode_tags(self, tags, encodings, tag2id, print_output):
        # Give tags to words instead of subwords

        labels = [[tag2id[tag] for tag in doc] for doc in
                  tags]  # corresponding numbers of the different labels in the data set
        encoded_labels = []
        idx = 0

        # The WordPiece tokenizer uses subwords, so there are more tokens than labels. We need the offset mapping to link each label to the last subword of each original word.

        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            initial_length = len(doc_labels)
            doc_enc_labels = np.ones(len(doc_offset),
                                     dtype=int) * -100  # we initialize an array with a length equal to the number of subwords, and fill it with -100, an unreachable number to avoid confusion.

            current_x, current_y, label_match_id = -1, -1, -1
            subword_counter = 0
            if idx <= 5 and print_output:
                print(idx)

            for i, (x, y) in enumerate(
                    doc_offset):  # We loop through the offset of each document to match the word endings.
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

                    if y != next_x and next_x == 0:
                        label_match_id += 1
                        doc_enc_labels[i] = doc_labels[
                            label_match_id]  # Switches the initial -100 to the correct label.
                        if idx <= 5 and print_output:
                            print(i, label_match_id,
                                  self.tokenizer.decode(encodings.encodings[idx].ids[i - subword_counter:i + 1]),
                                  self.id2tag[doc_labels[label_match_id]])
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

    def save(self, model, optimizer, output_model):
        torch.save({
            'model_state_dict': model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, output_model)

    def __init__(self, training_data, transformer_model, include_upos, include_xpos, model_dir):
        self.reader = CONLLReader(training_data)
        self.training_data = self.reader.parse_conll()
        self.transformer_model = transformer_model
        self.include_upos = include_upos
        self.include_xpos = include_xpos
        self.feature_dict = self.build_feature_dict()
        self.model_dir = model_dir

    def build_feature_dict(self):
        # Builds a dictionary with all tagging features and their possible values, based on the training data
        feature_dict = dict()
        for sent in self.training_data:
            for line in sent.split('\n'):
                split = line.rstrip().split('\t')
                if self.include_upos:
                    current_upos = split[self.reader.col_upos]
                    if 'UPOS' in feature_dict:
                        upos_values = feature_dict['UPOS']
                        upos_values.add(current_upos)
                    else:
                        upos_values = set()
                        upos_values.add(current_upos)
                        feature_dict['UPOS'] = upos_values
                if self.include_xpos:
                    current_xpos = split[self.reader.col_xpos]
                    if 'XPOS' in feature_dict:
                        xpos_values = feature_dict['XPOS']
                        xpos_values.add(current_xpos)
                    else:
                        xpos_values = set()
                        xpos_values.add(current_xpos)
                        feature_dict['XPOS'] = xpos_values
                if split[self.reader.col_morph] != '_':
                    morph = split[self.reader.col_morph].split('|')
                    for feat in morph:
                        split_feat = feat.split('=')
                        if split_feat[0] in feature_dict:
                            feat_values = feature_dict[split_feat[0]]
                            feat_values.add(split_feat[1])
                        else:
                            feat_values = set()
                            feat_values.add(split_feat[1])
                            feature_dict[split_feat[0]] = feat_values
        return feature_dict

    def read_tags(self, feature, data, return_tags=True, return_words=True):
        # Reads the values for a given feature (UPOS, XPOS or morphological) from the training set, as well as the wids and forms
        wid_sents = []
        token_sents = []
        tag_sents = []

        for sent in data:
            wids = []
            tokens = []
            tags = []
            for line in sent.split("\n"):
                split = line.split("\t")
                if return_words:
                    wids.append(split[self.reader.col_id])
                    tokens.append(split[self.reader.col_token])
                if return_tags:
                    if feature == 'UPOS':
                        tags.append(split[self.reader.col_upos])
                    elif feature == 'XPOS':
                        tags.append(split[self.reader.col_xpos])
                    else:
                        if split[self.reader.col_morph] == '_':
                            feature_values = self.feature_dict[feature]
                            feature_values.add('_')
                            tags.append('_')
                        else:
                            morph = split[self.reader.col_morph].split('|')
                            in_conll = False
                            for feature_value in morph:
                                feat_split = feature_value.split('=')
                                if feat_split[0] == feature:
                                    tags.append(feat_split[1])
                                    in_conll = True
                                    break
                            if not in_conll:
                                feature_values = self.feature_dict[feature]
                                feature_values.add('_')
                                tags.append('_')
            wid_sents.append(wids)
            token_sents.append(tokens)
            tag_sents.append(tags)

        if return_tags and return_words:
            return wid_sents, token_sents, tag_sents
        elif return_tags:
            return tag_sents
        else:
            return wid_sents, token_sents

    def train_model(self, wids, tokens, tags, output_model):
        # Trains a model for a given feature
        unique_tags = set(tag for doc in tags for tag in doc)
        tag2id = {tag: s_id for s_id, tag in enumerate(sorted(unique_tags))}
        # tag2id is saved to the model dir to be able to retrieve the labels during tagging
        with open(f'{output_model}/tag2id', 'w', encoding='UTF-8') as outfile:
            for key in tag2id:
                outfile.write(key + "\t" + str(tag2id[key]) + "\n")
        id2tag = {s_id: tag for tag, s_id in tag2id.items()}
        encodings = self.tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                   truncation=True)
        labels = self.encode_tags(tags, encodings, tag2id, print_output=False)
        encodings.pop("offset_mapping")
        dataset = AGPoSDataset(encodings, labels, wids)
        total_acc, total_count = 0, 0
        model = ElectraForTokenClassification.from_pretrained(self.transformer_model, num_labels=len(unique_tags))
        model.to(self.device)
        model.train()
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        optim = AdamW(model.parameters(), lr=5e-5)

        print(output_model)
        for epoch in range(10):
            train_gold = []
            train_predictions = []
            for idx, batch in enumerate(loader):
                optim.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                for batch_idx, batch_piece in enumerate(labels):
                    for label_idx, label in enumerate(batch_piece):
                        if label != -100:
                            predictions = list(outputs[1][batch_idx][label_idx])
                            current_pred = predictions.index(max(predictions))
                            if label == current_pred:
                                total_acc += 1
                            total_count += 1
                            train_gold.append(id2tag[label.item()])
                            train_predictions.append(id2tag[current_pred])
                loss = outputs[0]
                loss.backward()
                optim.step()
                print("| epoch {:3d} | {:5d}/{:5d} batches "
                      "| accuracy {:8.3f}".format(epoch, idx, len(loader), total_acc / total_count))
                total_acc, total_count = 0, 0
        model.save_pretrained(output_model)

    def calc_tag_probs(self,possible_tags,preds,word_no):
        tag_probs = {}
        for tag in possible_tags:
            tag_prob = 1
            for feat in tag:
                try:
                    prob_attr = preds[feat[0]][word_no][feat[1]]
                except KeyError:
                    print('Feature not found (probably mismatch with lexicon): '+feat[0]+' '+feat[1])
                except IndexError:
                    print(word_no)
                    print(len(preds[feat[0]]))
                tag_prob *= prob_attr
            tag_probs[tag] = tag_prob
        tag_probs = sorted(tag_probs.items(), reverse=True, key=lambda x: x[1])
        return tag_probs

    def tag_data(self, wids, tokens, preds, output_data, output_format='CONLL'):
        # Combines predictions, restricts outputs with lexicon and writes the output to CONLL file
        with open(output_data, 'w', encoding='UTF-8') as outfile:
            if output_format == 'tab':
                outfile.write("id\ttoken\tprobability\tpossibilities\tin_lexicon")
                for feat in self.feature_dict:
                    outfile.write("\t"+feat)
                outfile.write('\n')
            word_no = -1
            for sent_id, sent in enumerate(tokens):
                if sent_id % 100 == 0:
                    print("Tagging sentence "+str(sent_id))
                for word_id, word in enumerate(sent):
                    wid = wids[sent_id][word_id]
                    word_no += 1
                    possible_tags = self.possible_tags
                    if word in self.lexicon:
                        possible_tags = self.lexicon[word]
                    tag_probs = self.calc_tag_probs(possible_tags,preds,word_no)
                    top_prediction = tag_probs[0]
                    tag = dict(top_prediction[0])
                    upos = "_"
                    xpos = "_"
                    if self.include_upos:
                        upos = tag["UPOS"]
                    if self.include_xpos:
                        xpos = tag["XPOS"]
                    if output_format == 'CONLL':
                        morph = ""
                        for feat, val in tag.items():
                            if feat != 'UPOS' and feat != 'XPOS' and val != '_':
                                morph += feat + "=" + val + "|"
                        if morph == "":
                            morph = '_'
                        else:
                            morph = morph[:-1]
                        outfile.write(wid + "\t" + word + "\t_\t" + upos + "\t" + xpos + "\t" + morph + "\t_\t_\t_\t_\n")
                    elif output_format == 'tab':
                        outfile.write(wid +"\t" + word + "\t" + str(top_prediction[1]) +'\t' + str(len(possible_tags)) + "\t" + str(word in self.lexicon))
                        
                        for feat in self.feature_dict:
                            val = '_'
                            if feat in tag:
                                val = tag[feat]
                            outfile.write("\t"+val)
                        outfile.write('\n')
                if output_format == 'CONLL':
                    outfile.write('\n')

    def read_lexicon(self, file):
        lexicon = {}
        file = open(file, encoding='utf-8')
        raw_text = file.read().strip()
        lines = raw_text.split('\n')
        header = lines.pop(0).split('\t')
        feat_col = {}
        for col, feat in enumerate(header):
            feat_col[feat] = col
        for line in lines:
            entry = line.split('\t')
            form = entry[0]
            tag = []
            for feat in self.feature_dict:
                tag.append((feat, entry[feat_col[feat]]))
            tag = tuple(tag)
            if form in lexicon:
                tags = lexicon[form]
                if tag not in tags:
                    tags.append(tag)
            else:
                tags = []
                tags.append(tag)
                lexicon[form] = tags
        return lexicon
    
    def trim_lexicon(self):
        # Removes unnecessary information and sorts the tag in the order of the feature dict
        lexicon_new = {}
        for form, tags in self.lexicon.items():
            new_tags = []
            for tag in tags:
                new_tag = []
                for feat in self.feature_dict:
                    for tag_feat in tag:
                        if tag_feat[0] == feat:
                            new_tag.append(tag_feat)
                new_tag = tuple(new_tag)
                new_tags.append(new_tag)
            lexicon_new[form] = new_tags
        self.lexicon = lexicon_new           
    
    def add_training_data_to_lexicon(self):
        for sent in self.training_data:
            for line in sent.split("\n"):
                split = line.split("\t")
                morph = split[self.reader.col_morph]
                morph_dict = {}
                if morph != '_':
                    for feat_val in morph.split('|'):
                        feat, val = feat_val.split('=')
                        morph_dict[feat] = val
                tag = []
                for feat in self.feature_dict:
                    if feat == 'UPOS':
                        tag.append((feat, split[self.reader.col_upos]))
                    elif feat == 'XPOS':
                        tag.append((feat, split[self.reader.col_xpos]))
                    else:
                        if feat in morph_dict:
                            tag.append((feat, morph_dict[feat]))
                        else:
                            tag.append((feat, '_'))
                tag = tuple(tag)
                form = split[self.reader.col_token]
                if form in self.lexicon:
                    tags = self.lexicon[form]
                    if tag not in tags:
                        tags.append(tag)
                else:
                    tags = []
                    tags.append(tag)
                    self.lexicon[form] = tags

    def predict_feature(self, model_dir, wids, tokens, tags):
        # Makes prediction for a given feature
        file = open(model_dir + "/tag2id", encoding='utf-8')
        raw_text = file.read().strip()
        tag2id = {}
        id2tag = {}
        for line in raw_text.split("\n"):
            split = line.split("\t")
            tag2id[split[0]] = int(split[1])
            id2tag[int(split[1])] = split[0]
        encodings = self.tokenizer(tokens, is_split_into_words=True, return_offsets_mapping=True, padding=True,
                                   truncation=True)
        labels = self.encode_tags(tags, encodings, tag2id, print_output=False)
        encodings.pop("offset_mapping")
        dataset = AGPoSDataset(encodings, labels, wids)
        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        model = ElectraForTokenClassification.from_pretrained(model_dir).to(self.device)
        preds_total = []
        with torch.no_grad():
            for idx, batch in enumerate(loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
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

    def read_possible_tags(self, file):
        possible_tags = []
        file = open(file, encoding='utf-8')
        raw_text = file.read().strip()
        lines = raw_text.split('\n')
        header = lines.pop(0).split('\t')
        feat_col = {}
        for col, feat in enumerate(header):
            feat_col[feat] = col
        for line in lines:
            tag = []
            vals = line.split('\t')
            for feat in self.feature_dict:
                tag.append((feat, vals[feat_col[feat]]))
            tag = tuple(tag)
            possible_tags.append(tag)
        return possible_tags
    
    def string_to_tokens(self,string):
        string = re.sub(r'([\.,])',r' \1',string)
        string = re.sub(r'[᾽\'ʼ\\u0313´]', '’',string);
        string = re.sub(r'[‑—]', ' — ',string);
        string = re.sub('--', ' — ',string);
        string = re.sub(r'[“”„‘«»ʽ"]', ' " ',string);
        string = re.sub(r'[:··•˙]', ' ·',string)
        string = re.sub(';', ';',string);
        string = re.sub(';', ' ;',string)
        string = re.sub(r'[（\(]', r'\( ',string);
        string = re.sub(r'[）\)]', r' \)',string);
        string = re.sub(r'\s+',' ',string)
        string = re.sub(r'^ ', '',string);
        string = re.sub(r' $', '',string);
        tokens_str = string.split(' ')
        return tokens_str
    
    def tag_string(self, string):
        tokens_str = self.string_to_tokens(string)
        tokens = []
        current_sent = []
        new_sent = False
        for token in tokens_str:
            token = ud.normalize("NFKD",token)
            token = fix_accents(token)
            if new_sent:
                tokens.append(current_sent)
                current_sent = []
                new_sent = False
            current_sent.append(token)
            if token=='.' or token==';' or token == '·':
                new_sent = True
        tokens.append(current_sent)
        wids = []
        for sent in tokens:
            wids_sent = []
            for id, token in enumerate(sent):
                wids_sent.append(str(id+1))
            wids.append(wids_sent)
        all_preds = {}
        wids = []
        wids.append(wids_sent)
        for feat in self.feature_dict:
            tags = []
            for sent in tokens:
                tag_sent = []
                for token in sent:
                    if feat == 'XPOS':
                        tag_sent.append('PUNCT')
                    else:
                        tag_sent.append('_')
                tags.append(tag_sent)
            preds = self.predict_feature(f"{self.model_dir}/{feat}", wids, tokens, tags)
            all_preds[feat] = preds
        word_no = -1
        output = ""
        for sent_id, sent in enumerate(tokens):
            for word_id, word in enumerate(sent):
                    wid = wids[sent_id][word_id]
                    word_no += 1
                    possible_tags = self.possible_tags
                    if word in self.lexicon:
                        possible_tags = self.lexicon[word]
                    tag_probs = self.calc_tag_probs(possible_tags,all_preds,word_no)
                    top_prediction = tag_probs[0]
                    tag = dict(top_prediction[0])
                    second_tag = {}
                    if len(tag_probs)>1:
                        second_prediction = tag_probs[1]
                        second_tag = dict(second_prediction[0])
                    output+=('{0:.3f}'.format(top_prediction[1])+'&nbsp;&nbsp;&nbsp;&nbsp;'+'<b><font style="'+self.color_by_prob(top_prediction[1])+'">'+word+'</font></b>'+'&nbsp;&nbsp;&nbsp;&nbsp;'+str(word in self.lexicon)+'&nbsp;&nbsp;&nbsp;&nbsp;'+str(tag)+'&nbsp;&nbsp;&nbsp;&nbsp;'+str(second_tag)+'<br>')
        return output

    def color_by_prob(self,prob):
        green = 200*prob
        red = 200*(1-prob)
        return "color: rgb("+f'{red:.0f}'+","+f'{green:.0f}'+",0)";