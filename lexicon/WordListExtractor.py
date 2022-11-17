import beta_code
import re
import xml.etree.ElementTree as ET

class WordListExtractor:
    
    def read_xml(self,in_file):
        wordlist = []
        tree = ET.parse(in_file)
        for word in tree.findall('.//word'):
            if 'form' in word.attrib and not 'artificial' in word.attrib:
                form = word.attrib['form']
                if not form in wordlist:
                    wordlist.append(form)
        return wordlist
    
    def convert_beta_code(self,wordlist,normalizations):
        wordlist_beta = []
        for form_uni in wordlist:
            if form_uni != '' and form_uni != '...' and not re.match('.*[G–⏔⏕⏑x].*',form_uni) and not re.match('[\\.,·;:\\(\\)—†\"]',form_uni) and not re.match('.*-$',form_uni):
                form_beta = beta_code.greek_to_beta_code(form_uni)
                if normalizations:
                    form_beta = re.sub('[’᾽]','\'',form_beta)
                    form_beta = re.sub(r'\\','/',form_beta)
                    form_beta = re.sub('([/=])(.*)/',r'\1\2',form_beta)
                    form_beta = re.sub('s1\'','s\'',form_beta)
                if not form_beta in wordlist_beta:
                    wordlist_beta.append(form_beta)
        return wordlist_beta
    
    def output_word_list(self,wordlist,out_file):
        f = open(out_file,'w')
        for word in wordlist:
            f.write(word+'\n')
        f.close()
    
def main():
    wle = WordListExtractor()
    wordlist = wle.read_xml(r'C:\Users\u0111778\Documents\Corpora\TreebankData_0721\XML\NicaeanCreed_Harrington.xml')
    wordlist_beta = wle.convert_beta_code(wordlist,True)
    wle.output_word_list(wordlist_beta,r'C:\Users\u0111778\Desktop\tmp.txt')
        
if __name__ == '__main__':
    main()