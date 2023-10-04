from datasets import Dataset

def build_dataset(tokens,tag_dict=None,wids=None):
    dataset = []
    for sent_id, sent in enumerate(tokens):
        sent_dict = dict()
        if wids is not None:
            sent_dict['wids'] = wids[sent_id]
        sent_dict['tokens'] = tokens[sent_id]
        if tag_dict is not None:
            for tag_class in tag_dict:
                sent_dict[tag_class] = tag_dict[tag_class][sent_id]
        dataset.append(sent_dict)
    return Dataset.from_list(dataset)        
