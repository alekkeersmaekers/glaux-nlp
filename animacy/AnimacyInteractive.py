from tokenization import Tokenization
from treebanks import Tagsets
from datasets import disable_progress_bar
import pandas as pd
import unicodedata as ud
from data import TabularDatasets
import xgboost as xgb
import shap
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tagger.Tagger import Tagger
from lemmatization.DictionaryLemmatizer import DictionaryLemmatizer
from vectors.VectorExtractor import VectorExtractor
import pickle

def enter_sent():
    example_sentences = {
    '1': 'ἡ γὰρ πόλις ἥδε, καὶ εἰ ἔρχονται Ἀθηναῖοι, ἀμυνεῖται αὐτοὺς ἀξίως αὑτῆς, καὶ στρατηγοί εἰσιν ἡμῖν οἳ σκέψονται αὐτά·',
    '2': 'οἱ δὲ περὶ τὸν Ἱερώνυμον ἀπὸ τούτων τῶν καιρῶν ἐνήργουν τὰ τοῦ πολέμου, καὶ τάς τε δυνάμεις ἥθροιζον καὶ καθώπλιζον τάς τε λοιπὰς χορηγίας ἡτοίμαζον.',
    '3': 'μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος οὐλομένην, ἣ μυρί᾽ Ἀχαιοῖς ἄλγε᾽ ἔθηκε, πολλὰς δ᾽ ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν οἰωνοῖσί τε πᾶσι, Διὸς δ᾽ ἐτελείετο βουλή, ἐξ οὗ δὴ τὰ πρῶτα διαστήτην ἐρίσαντε Ἀτρεΐδης τε ἄναξ ἀνδρῶν καὶ δῖος Ἀχιλλεύς.',
    '4': 'κατέβην χθὲς εἰς Πειραιᾶ μετὰ Γλαύκωνος τοῦ Ἀρίστωνος προσευξόμενός τε τῇ θεῷ καὶ ἅμα τὴν ἑορτὴν βουλόμενος θεάσασθαι τίνα τρόπον ποιήσουσιν ἅτε νῦν πρῶτον ἄγοντες.',
    '5': 'τῆς Ἀττικῆς νομίζετ᾿ εἶναι τὸν τόπον, Φυλήν, τὸ νυμφαῖον δ᾿ ὅθεν προέρχομαι Φυλασίων καὶ τῶν δυναμένων τὰς πέτρας ἐνθάδε γεωργεῖν, ἱερὸν ἐπιφανὲς πάνυ.'
    }  
    sent = input('Enter a Greek sentence:\n')
    if sent in ['1','2','3','4','5']:
        sent = example_sentences[sent]
    return sent

def analyze_sent(sent,tagger,lemmatizer,classifier,extractor,static_vectors,alignment_vectors,person_gazetteer,place_gazetteer):
    disable_progress_bar()
    tokens = [Tokenization.greek_glaux_to_tokens(sent)]
    wids_sent = []
    for word_no, _ in enumerate(tokens[0]):
        wids_sent.append(str(word_no))
    wids = [wids_sent]
    tokens_norm = Tokenization.normalize_tokens(tokens, 'greek_glaux')
    all_preds = tagger.tag_seperately(tokens_norm, print_prediction=False)
    best_tags, num_poss = tagger.tag_data(tokens_norm, all_preds, False, True, disable_progress_bar=True)
    labels_sent = []
    test_data = []
    for tag_no, tag in enumerate(best_tags):
        tag_dict = dict(tag[0])
        tag_dict['pos'] = tag_dict.pop('XPOS')
        if tag_dict['pos'] == 'noun':
            labels_sent.append('NA')
            perseus_tag = Tagsets.feats_to_perseus(tag_dict)
            token = tokens[0][tag_no]
            lemma = ud.normalize('NFC',lemmatizer.lemmatize(token,perseus_tag))
            test_data.append([tag_no,token,lemma])
        else:
            labels_sent.append('_')
    labels = [labels_sent]
    classifier.test_data = pd.DataFrame(test_data,columns=['ID','FORM','LEMMA'])
    dataset = extractor.build_dataset(wids,tokens,labels,"MISC",'NFC')
    vectors_test = extractor.extract_vectors(dataset,exclude_labels='_',label_name='MISC',layers=range(1,13))
    for key, vector in vectors_test.items():
        vectors_test[key] = [round(val,3) for val in vectors_test[key]]
    features = [
           ['static_embedding',{'vector_file':static_vectors,'feature_name':'Type','fill_nas':None,'normalize':True}],
           ['transformer_embedding',{'feature_name':'Token','transformer_embeddings':vectors_test,'normalize':True}],
           ['capital'],
           ['static_embedding',{'vector_file':alignment_vectors,'feature_name':'Eng','fill_nas':None,'header':None,'normalize':True}],
           ['gazetteer',{'gazetteer_file':person_gazetteer,'gazetteer_name':'Person'}],
           ['gazetteer',{'gazetteer_file':place_gazetteer,'gazetteer_name':'Place'}],
           ]
    classifier.test_data = TabularDatasets.add_features(classifier.test_data, features)
    test = classifier.test_data.copy()
    test.drop(columns=['ID','FORM','LEMMA'],inplace=True)
    test_matrix = xgb.DMatrix(data=test,enable_categorical=True)
    predictions = classifier.models[0].predict(test_matrix)
    label_encoder = classifier.label_encoders[0]
    classifier.set_predictions(label_encoder[predictions.astype(int)])
    results = pd.DataFrame(test_data,columns=['ID','FORM','LEMMA'])
    results.drop(columns=['ID','LEMMA'],inplace=True)
    results['PREDICTION'] = results.index.map(classifier.predictions)
    explainer = shap.TreeExplainer(classifier.models[0])
    shap_values = explainer(test_matrix)
    if shap_values.feature_names is None:
        shap_values.feature_names = list(test.columns)
        shap_values.data = np.array(test)
    classifier.merge_shap_values(shap_values,'Type')
    classifier.merge_shap_values(shap_values,'Token')
    classifier.merge_shap_values(shap_values,'Eng')
    shap_values.feature_names = ['General meaning of the word','Sentence context','Is the word capitalized?','English translations of the word','Does the word occur in a list of people?','Does the word occur in a list of places?']
    return tokens, results, shap_values

def get_prediction_graph_info(classifier,token_index,tokens):
    type_vectors = classifier.training_data.iloc[:,5:305].copy()
    type_vectors.index = classifier.training_data['LEMMA']
    vector = pd.DataFrame(classifier.test_data.iloc[token_index,3:303].copy()).T
    if not vector.isnull().values.any():
        instance_lemma = classifier.test_data.iloc[token_index,2]
        vector.index = [instance_lemma]
        type_vectors = pd.concat([type_vectors,vector])
        type_vectors.drop_duplicates(inplace=True)
        type_vectors = type_vectors[type_vectors['Type1']!=0].copy()
        cosine_sim = cosine_similarity(type_vectors)
        cosine_matrix = pd.DataFrame(cosine_sim,index=type_vectors.index,columns=type_vectors.index)
        neighbors = cosine_matrix[instance_lemma].sort_values(ascending=False)
        closest = neighbors[1:4]
        type_feat_str = f"Similar to {','.join(closest.keys())}, ..."
    else:
        type_feat_str = f'This word is not recognized (probably it is infrequent)'
    word_no = classifier.test_data.iloc[token_index,0]
    context = []
    sent = tokens[0]
    if word_no-5 > 0:
        context.append('...')
        if word_no+5 < len(sent)-1:
            context.extend(sent[word_no-5:word_no+6])
            context.append('...')
        else:
            context.extend(sent[word_no-5:len(sent)])
    else:
        if word_no+5 < len(sent)-1:
            context.extend(sent[0:word_no+6])
            context.append('...')
        else:
            context = sent
    token_str = ' '.join(context)
    return type_feat_str, token_str

def explain_prediction(pred_class,classifier,shap_values,token_index,type_feat_str,token_str,alignment_lexicon,positive=True):
    instance_lemma = classifier.test_data.iloc[token_index,2]
    class_index = list(classifier.label_encoders[0]).index(pred_class)
    shap_values.data[token_index,0] = type_feat_str
    shap_values.data[token_index,1] = token_str
    shap_values.data[token_index,3] = ', '.join(alignment_lexicon.get(instance_lemma,['No translation for the word available (probably it is infrequent)']))
    fig = shap.plots.waterfall(shap_values[token_index,:,class_index],show=False)
    if positive:
        title = f"Why is {classifier.test_data.iloc[token_index,1]} predicted as '{pred_class}'?"
    else:
        title = f"Why is {classifier.test_data.iloc[token_index,1]} NOT predicted as '{pred_class}'?"
    plt.title(title)
    plt.show()

def setup(tagger_path,lemmatizer_path,classifier_path,alignment_lexicon_path):
    tagger = Tagger(transformer_path='mercelisw/electra-grc',model_dir=tagger_path)
    lemmatizer = DictionaryLemmatizer(lemmatizer_path)
    with open(classifier_path,'rb') as infile:
        classifier = pickle.load(infile)
    extractor = VectorExtractor(transformer_path='bowphs/greberta',data_preset='simple',tokenizer_add_prefix_space=True)
    alignment_lexicon = {}
    with open(alignment_lexicon_path,encoding='utf8') as infile:
        count = 0
        for line in infile.readlines():
            sl = line.strip().split('\t')
            key = ud.normalize('NFC',sl[0])
            val = []
            for transl in sl[1].split('|'):
                val.append(transl.split('=')[0])
            alignment_lexicon[key] = val
    return tagger, lemmatizer, classifier, extractor, alignment_lexicon