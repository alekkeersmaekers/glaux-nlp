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

def split_n_fold(dataset,n_fold):
    fold_size = int(len(dataset) / n_fold)
    folds = []
    index = 0
    for i in range(n_fold):
        data = {}
        start = index
        end = index + fold_size
        if i+1 == n_fold or end > (len(dataset)):
            end = len(dataset)
        rows = range(start,end)
        data['test'] = dataset.select(i for i in rows)
        data['train'] = dataset.select(i for i in range(len(dataset)) if i not in rows)
        folds.append(data)
        index = index + fold_size
    return folds