from lexicon.MorpheusProcessor import MorpheusProcessor
from lexicon.LexiconProcessor import LexiconProcessor

if __name__ == '__main__':
    #file = open(r'C:\Users\u0111778\OneDrive - KU Leuven\Glaux\1.0\nlp\Morpheus\AllForms_Treebanks.morph',encoding='utf-8')
    file = open(r'C:\Users\u0111778\Documents\Lexica\Woodhouse_Unknown_Lemmata.morph',encoding='utf-8')
    morpheus_output = file.read().strip()
    mp = MorpheusProcessor()
    lexicon = mp.convert_morpheus_output(morpheus_output, False)
    lp = LexiconProcessor(lexicon)
    lp.write_lexicon(r'C:\Users\u0111778\Documents\Lexica\woodhouse_unknown_lemmata.txt','CONLL',['number','gender','case','degree','person','tense','mood','voice'])
    #lp.write_lexicon(r'C:\Users\u0111778\OneDrive - KU Leuven\Glaux\1.0\nlp\Morpheus\lexicon_treebanks_conll.txt','CONLL',['number','gender','case','degree','person','tense','mood','voice'])
    