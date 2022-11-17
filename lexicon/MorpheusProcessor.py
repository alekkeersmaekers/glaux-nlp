from subprocess import Popen, PIPE
import re

class MorpheusProcessor:
    
    def regularize_lemma(self,lemma,form):
        digit = re.sub('[^0-9]','')
        lemma = lemma.sub('[0-9]','')
        lemma = lemma.sub('-pl$','')
        
    
    def send_word_list(self,wordlist):
        p = Popen(["cruncher","-d"], stdout=PIPE,stdin=PIPE)
        command = ''
        for word in wordlist:
            command += word + '\n'
        morpheus_output = p.communicate(command.encode('UTF-8'))[0]
        return morpheus_output.decode('UTF-8')

def main():
    
    wordlist = ['le/gw','gra/fw']
    processor = MorpheusProcessor()
    print(processor.send_word_list(wordlist))

if __name__ == '__main__':
    main()