# GLAUx NLP

Code used in the [GLAUx project](https://github.com/alekkeersmaekers/glaux). While it has been primarily developed to analyze Ancient Greek, many of the modules are general enough to be used to handle other highly inflectional languages as well (see 'how to use').

# How to use

## Morphological tagging

The easiest way to use the tagger is through the command line interface:
`python Tagger.py mode transformer_path model_dir [additional arguments]`

'mode' is either train or test, 'transformer_path' is the path of the pretrained transformer model (can be a huggingface path, e.g. mercelisw/electra-grc for https://huggingface.co/mercelisw/electra-grc) and 'model_dir' is the directory of the tagger model (or the directory where you want it to be saved in case mode=train).

Additional arguments can be specified (see [this paper](https://aclanthology.org/2024.ml4al-1.17/) for more details):
- --training_data: the file containing the training data for the tagger. Note that if mode=test and training data is specified, by default the possible tags that can be predicted are constrained to the ones occuring in the training data (if an external list of possible tags is also provided, the two lists are combined) and if a lexicon is specified, all the words and their tags that are present in the training data are added to this lexicon as well. If you do not want the former behavior, use the flag --no-add_td_to_possible_tags (see below). If you do not want either behavior, simply do not specify the training data.
- --test_data: the prediction data for the tagger.
- --feats: the columns in your data that contain the classes that you want to predict (by default UPOS,XPOS,FEATS). Typically it makes more sense to change this to either UPOS,FEATS or XPOS,FEATS, depending on the format of your data. It is also possible to specify values in the FEATS column, e.g. if --feats=gender only gender is predicted.
- --tokenizer_path: path to the tokenizer. Should only be specified if it is different form the transformer_path.
- --output_file: for mode=test, the name of the output file that will contain the tagger's prediction.
- --output_format: format of the output data, CONLLU or tab (tabular format, with tag probability, number possible tags, whether the tag occurs in the lexicon, and without sentence boundaries).
- --data_preset: format of the training data, use CONLLU if your file is a strict CONLLU, you can also use simple_tagger if your data only has the columns ID, FORM, UPOS, XPOS and FEATS (in this order). If you want to define your own format, use the argument feature_cols.
- --feature_cols: to define a custom format for your training data. For example, if you want a 4 column format with columns ID, FORM, UPOS and FEATS, set feature_cols to {"ID":0,"FORM":1,"UPOS":2,"FEATS":3}.
- --unknown_label: by the default behavior of the tagger, all tokens in the test data need a tag. If you're running the tagger on unlabeled data, you can simply assign a dummy label to the tokens (e.g. '_'), and set unknown_label to this value.
- --is_multitask: enables multitask learning. Should also be specified if a multitask model is used during mode=test.
- --possible_tags_file: constrains the feature combinations that can be predicted to the ones occurring in this file (see 'file formatting' below for the format).
- --no-add_td_to_possible_tags: do not constrain the feature combinations that can be predicted to the ones occurring in the training data. If a possible tags file is specified, the feature combinations that are predicted are constrained to only the ones occurring in this file (and not the training data). If no such file is specified, simply all feature combinations are possible.
- --lexicon: constrains the tags that can be predicted for a given form to the ones occuring in this lexicon for this form (see 'file formatting' below for the format).
- --normalization_rule: normalizes the tokens, necessary for some pretrained language models. Typically you would use NFC. If you use electra-grc for Greek, use greek_glaux.
- --epochs: number of training epochs (default 3)
- --learning_rate: training learning rate (default 2e-5)
- --batch_size: training/testing batch size (default 16)
- --tokenizer_add_prefix_space: should be enabled when your tokenizer is a RoBERTa tokenizer.

So what options should I use?
- As described in the above paper, whether a lexicon is beneficial may depend from language to language as well as how strictly the tags present in the lexicon match the ones in the training data: in our experiments we found a clear positive effect for Ancient Greek, but not so much for Latin.
- If you can easily generate a possible tags file, it could be a good idea to provide this, in order to avoid that feature combinations that do not make sense linguistically can be predicted (e.g. noun+voice). Otherwise, it is probably a good idera to use the flag no-add_td_to_possible_tags if you're expecting that your training data does not cover all possible feature combinations for your language.
- We did not find any positive effect of using multitask learning for the 2 languages that we tested, even a slight negative one for Ancient Greek (although it makes a big difference in file size).
- In the paper, instead of the defaults specified above we trained for 10 epochs with a learning rate of 5e-5, but we did not test whether different hyperparameter settings would yield different results.

For reference, the best-performing models described in the paper mentioned above were trained and tested with the following settings (the files are available on https://github.com/alekkeersmaekers/transformer-tagging - except for the Latin training/test data, which are directly taken from the Universal Dependencies treebanks - while the models can all be found on https://huggingface.co/alekkeersmaekers):

a) Greek

- `python -m tagger.Tagger train mercelisw/electra-grc models/greek/greek-tagging-no-multitask --training_data files/greek/data_training.txt --feats XPOS,FEATS --normalization_rule greek_glaux --epochs 10 --learning_rate 5e-5`
- `python -m tagger.Tagger test mercelisw/electra-grc models/greek/greek-tagging-no-multitask --training_data files/greek/data_training.txt --test_data files/greek/data_test.txt --output_file files/greek/data_test_tagged.txt --possible_tags_file files/greek/possible_tags.txt --lexicon files/greek/lexicon.txt --feats XPOS,FEATS --normalization_rule greek_glaux`

b) Latin

- `python -m tagger.Tagger train bowphs/LaBerta models/latin/latin-tagging-no-multitask --training_data files/latin/la_proiel-ud-train.conllu --feats UPOS,FEATS --epochs 10 --learning_rate 5e-5 --tokenizer_add_prefix_space`
- `python -m tagger.Tagger test bowphs/LaBerta models/latin/latin-tagging-no-multitask --training_data files/latin/la_proiel-ud-train.conllu --test_data files/latin/la_proiel-ud-test.conllu --output_file files/latin/data_test_tagged.txt --possible_tags_file files/latin/possible_tags.txt --feats UPOS,FEATS --tokenizer_add_prefix_space`

### File formatting

See https://github.com/alekkeersmaekers/transformer-tagging for some examples. The lexicon has a header specifying the feature names, with each feature separated by a tab, e.g.:
| form | XPOS | person | number | tense | mood | voice | gender | case | degree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ἀρχόμενος | participle | _ | sg | pres | _ | mid | masc | nom | _ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

The format of the possible tags list is similar, e.g. :
| XPOS | person | number | tense | mood | voice | gender | case | degree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adjective | _ | sg | _ | _ | _ | fem | acc | pos |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

In the future, we will add code to this repository that can generate such a possible tags list based on a number of user-defined constraints.

# How to cite

Please cite the paper(s) that discuss(es) the subpart of the code base that you are using:
- In case you are making use of the morphological tagger, please cite:
`Alek Keersmaekers and Wouter Mercelis. 2024. Adapting transformer models to morphological tagging of two highly inflectional languages: a case study on Ancient Greek and Latin. In Proceedings of the 1st Workshop on Machine Learning# How to use

## Morphological tagging

The easiest way to use the tagger is through the command line interface:
`python Tagger.py mode transformer_path model_dir [additional arguments]`

'mode' is either train or test, 'transformer_path' is the path of the pretrained transformer model (can be a huggingface path, e.g. mercelisw/electra-grc for https://huggingface.co/mercelisw/electra-grc) and 'model_dir' is the directory of the tagger model (or the directory where you want it to be saved in case mode=train).

Additional arguments can be specified (see [this paper](https://aclanthology.org/2024.ml4al-1.17/) for more details):
- --training_data: the file containing the training data for the tagger. Note that if mode=test and training data is specified, by default the possible tags that can be predicted are constrained to the ones occuring in the training data (if an external list of possible tags is also provided, the two lists are combined) and if a lexicon is specified, all the words and their tags that are present in the training data are added to this lexicon as well. If you do not want the former behavior, use the flag --no-add_td_to_possible_tags (see below). If you do not want either behavior, simply do not specify the training data.
- --test_data: the prediction data for the tagger.
- --feats: the columns in your data that contain the classes that you want to predict (by default UPOS,XPOS,FEATS). Typically it makes more sense to change this to either UPOS,FEATS or XPOS,FEATS, depending on the format of your data. It is also possible to specify values in the FEATS column, e.g. if --feats=gender only gender is predicted.
- --tokenizer_path: path to the tokenizer. Should only be specified if it is different form the transformer_path.
- --output_file: for mode=test, the name of the output file that will contain the tagger's prediction.
- --output_format: format of the output data, CONLLU or tab (tabular format, with tag probability, number possible tags, whether the tag occurs in the lexicon, and without sentence boundaries).
- --data_preset: format of the training data, use CONLLU if your file is a strict CONLLU, you can also use simple_tagger if your data only has the columns ID, FORM, UPOS, XPOS and FEATS (in this order). If you want to define your own format, use the argument feature_cols.
- --feature_cols: to define a custom format for your training data. For example, if you want a 4 column format with columns ID, FORM, UPOS and FEATS, set feature_cols to {"ID":0,"FORM":1,"UPOS":2,"FEATS":3}.
- --unknown_label: by the default behavior of the tagger, all tokens in the test data need a tag. If you're running the tagger on unlabeled data, you can simply assign a dummy label to the tokens (e.g. '_'), and set unknown_label to this value.
- --is_multitask: enables multitask learning. Should also be specified if a multitask model is used during mode=test.
- --possible_tags_file: constrains the feature combinations that can be predicted to the ones occurring in this file (see 'file formatting' below for the format).
- --no-add_td_to_possible_tags: do not constrain the feature combinations that can be predicted to the ones occurring in the training data. If a possible tags file is specified, the feature combinations that are predicted are constrained to only the ones occurring in this file (and not the training data). If no such file is specified, simply all feature combinations are possible.
- --lexicon: constrains the tags that can be predicted for a given form to the ones occuring in this lexicon for this form (see 'file formatting' below for the format).
- --normalization_rule: normalizes the tokens, necessary for some pretrained language models. Typically you would use NFC. If you use electra-grc for Greek, use greek_glaux.
- --epochs: number of training epochs (default 3)
- --learning_rate: training learning rate (default 2e-5)
- --batch_size: training/testing batch size (default 16)
- --tokenizer_add_prefix_space: should be enabled when your tokenizer is a RoBERTa tokenizer.

So what options should I use?
- As described in the above paper, whether a lexicon is beneficial may depend from language to language as well as how strictly the tags present in the lexicon match the ones in the training data: in our experiments we found a clear positive effect for Ancient Greek, but not so much for Latin.
- If you can easily generate a possible tags file, it could be a good idea to provide this, in order to avoid that feature combinations that do not make sense linguistically can be predicted (e.g. noun+voice). Otherwise, it is probably a good idera to use the flag no-add_td_to_possible_tags if you're expecting that your training data does not cover all possible feature combinations for your language.
- We did not find any positive effect of using multitask learning for the 2 languages that we tested, even a slight negative one for Ancient Greek (although it makes a big difference in file size).
- In the paper, instead of the defaults specified above we trained for 10 epochs with a learning rate of 5e-5, but we did not test whether different hyperparameter settings would yield different results.

For reference, the best-performing models described in the paper mentioned above were trained and tested with the following settings (the files are available on https://github.com/alekkeersmaekers/transformer-tagging - except for the Latin training/test data, which are directly taken from the Universal Dependencies treebanks - while the models can all be found on https://huggingface.co/alekkeersmaekers):

a) Greek

- `python -m tagger.Tagger train mercelisw/electra-grc models/greek/greek-tagging-no-multitask --training_data files/greek/data_training.txt --feats XPOS,FEATS --normalization_rule greek_glaux --epochs 10 --learning_rate 5e-5`
- `python -m tagger.Tagger test mercelisw/electra-grc models/greek/greek-tagging-no-multitask --training_data files/greek/data_training.txt --test_data files/greek/data_test.txt --output_file files/greek/data_test_tagged.txt --possible_tags_file files/greek/possible_tags.txt --lexicon files/greek/lexicon.txt --feats XPOS,FEATS --normalization_rule greek_glaux`

b) Latin

- `python -m tagger.Tagger train bowphs/LaBerta models/latin/latin-tagging-no-multitask --training_data files/latin/la_proiel-ud-train.conllu --feats UPOS,FEATS --epochs 10 --learning_rate 5e-5 --tokenizer_add_prefix_space`
- `python -m tagger.Tagger test bowphs/LaBerta models/latin/latin-tagging-no-multitask --training_data files/latin/la_proiel-ud-train.conllu --test_data files/latin/la_proiel-ud-test.conllu --output_file files/latin/data_test_tagged.txt --possible_tags_file files/latin/possible_tags.txt --feats UPOS,FEATS --tokenizer_add_prefix_space`

### File formatting

See https://github.com/alekkeersmaekers/transformer-tagging for some examples. The lexicon has a header specifying the feature names, with each feature separated by a tab, e.g.:
| form | XPOS | person | number | tense | mood | voice | gender | case | degree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ἀρχόμενος | participle | _ | sg | pres | _ | mid | masc | nom | _ |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

The format of the possible tags list is similar, e.g. :
| XPOS | person | number | tense | mood | voice | gender | case | degree |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adjective | _ | sg | _ | _ | _ | fem | acc | pos |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

In the future, we will add code to this repository that can generate such a possible tags list based on a number of user-defined constraints.

# How to cite

Please cite the paper(s) that discuss(es) the subpart of the code base that you are using:
- In case you are making use of the morphological tagger, please cite:

> Alek Keersmaekers and Wouter Mercelis. 2024. Adapting transformer models to morphological tagging of two highly inflectional languages: a case study on Ancient Greek and Latin. In *Proceedings of the 1st Workshop on Machine Learning for Ancient Languages (ML4AL 2024)*, pages 165–176. DOI:[10.18653/v1/2024.ml4al-1.17](https://doi.org/10.18653/v1/2024.ml4al-1.17).
