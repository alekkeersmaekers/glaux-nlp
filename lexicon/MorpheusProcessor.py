from subprocess import Popen, PIPE
import re, os, beta_code, unicodedata
from tokenization.Tokenization import normalize_token

class MorpheusProcessor:
    
    word_accents = None
    
    def __init__(self):
        self.stemtypes = os.getenv('MORPHLIB') + '/Greek/rule_files/stemtypes.table'
    
    def load_accents(self,accent_file):
        word_accents = dict()
        file = open(accent_file,encoding='utf-8')
        lines = file.readlines()
        for line in lines:
            split = line.strip().split('\t')
            if len(split) == 3:
                word_accents[split[0]+'€'+split[1]] = split[2]
        self.word_accents = word_accents
    
    def beta_to_uni(self,form):
        # First line is a fix for unicodedata where the order of the diaeresis is wrong, can be fixed when this is fixed
        form_norm = re.sub(r'\+([/=])',r'\1+',form)
        form_uni = beta_code.beta_code_to_greek(form_norm)
        form_uni = unicodedata.normalize("NFD",form_uni)
        return form_uni
    
    def regularize_lemma(self,lemma,form):
        digit = re.sub('[^0-9]','',lemma)
        lemma = re.sub('[0-9]','',lemma)
        lemma = re.sub('-pl$','',lemma)
        lemma = re.sub(r'([aehiouw][\(\)]?/)i(?!\+)',r'\1i+',lemma)
        lemma = re.sub(r'([hw])i(/?)(?!\+)',r'\1i\2+',lemma)
        if lemma == 'kata/-ka/qhmai':
            lemma = 'ka/qhmai'
        elif lemma == 'kata/-kaqe/zomai':
            lemma = 'kaqe/zomai'
        elif lemma == 'a)mfi/-a)mfie/nnumi':
            lemma = 'a)mfie/nnumi'
        elif lemma == 'kata/-kaqeu/dw':
            lemma = 'kaqeu/dw'
        elif lemma == 'e)pi/-e)pikoure/w':
            lemma = 'e)pikoure/w'
        if re.match('.*-.*',lemma):
            splittedWord = lemma.split('-')
            prefices = splittedWord[0]
            base = splittedWord[1]
            splittedPrefix = prefices.split(',')
            if base == 'dikei=n':
                base = 'dike/w'
            elif base == 'e)/ssomai':
                base = 'ei)mi/'
            elif base == 'gei/nomai':
                base = 'gi/gnomai'
            elif re.match('.*e\\)?rw=$',base):
                base = re.sub('e\\)?rw=','le/gw',base)
            elif base == 'i)sxne/omai' and not splittedPrefix[len(splittedPrefix)-1] == 'u(po/':
                base = 'e)/xw'
            elif base == 'ei)=don':
                base = 'o(ra/w'
            elif base == 'ei)=pon':
                base = 'le/gw'
            elif base == 'i(sta/w':
                base = 'i(/sthmi'
            elif base == 'ei)/rw' and digit == '2':
                base = 'le/gw'
            elif base == 'ei)re/w':
                base = 'le/gw'
            elif base == 'r(e/omai':
                base = 'le/gw'
            elif base == 'dei=' and len(splittedPrefix)>0:
                base = 'de/w'
            elif base == 'store/nnumi':
                base = 'sto/rnumi'
            for i in range(len(splittedPrefix)):
                splittedPrefix[i] = re.sub('[/=\\\\]', '',splittedPrefix[i])
            for i in range(len(splittedPrefix)-1):
                if re.match('.*[aehiouw]$',splittedPrefix[i]) and not splittedPrefix[i] == 'peri' and not splittedPrefix[i] == 'pro' and not splittedPrefix[i] == 'a)mfi' and re.match('^[aehiouw].*',splittedPrefix[i+1]):
                    splittedPrefix[i] = re.sub('[aehiouw]$','',splittedPrefix[i])
                elif re.match('.*n$',splittedPrefix[i]) and re.match('^k.*',splittedPrefix[i+1]):
                    splittedPrefix[i] = re.sub('n$','g',splittedPrefix[i])
                elif re.match('.*n$',splittedPrefix[i]) and re.match('^p.*',splittedPrefix[i+1]):
                    splittedPrefix[i] = re.sub('n$','m',splittedPrefix[i])
                elif re.match('.*k$',splittedPrefix[i]) and re.match('^[aehiouw].*',splittedPrefix[i+1]):
                    splittedPrefix[i]  =re.sub('k$','c',splittedPrefix[i])
                if re.match('.*[tp]$',splittedPrefix[i]) and re.match('^u\\(.*',splittedPrefix[i+1]):
                    splittedPrefix[i] = re.sub('t$','q',splittedPrefix[i])
                    splittedPrefix[i] = re.sub('p$','f',splittedPrefix[i])
            if len(splittedPrefix) >1:
                for i in range(1,len(splittedPrefix)):
                    splittedPrefix[i] = re.sub('[\\(\\)]','',splittedPrefix[i])
            lastPrefix = splittedPrefix[len(splittedPrefix)-1]
            if re.match('^[aehiouw]+\\(.*',base):
                base = re.sub('\\(','',base)
                if re.match('^(ka|me)ta$',lastPrefix):
                    lastPrefix = re.sub('ta$','q',lastPrefix)
                elif re.match('^a(\\))?po$',lastPrefix) or re.match('^u(\\()?po$',lastPrefix):
                    lastPrefix = re.sub('po$','f',lastPrefix)
                elif re.match('^(dia|para)$',lastPrefix):
                    lastPrefix = re.sub('a$','',lastPrefix)
                elif re.match('^e(\\))?pi$',lastPrefix):
                    lastPrefix = re.sub('pi$','f',lastPrefix)
                elif re.match('^e(\\))?k$',lastPrefix):
                    lastPrefix = re.sub('k$','c',lastPrefix)
                elif re.match('^a(\\))?na$',lastPrefix) or re.match('^para$',lastPrefix):
                    lastPrefix = re.sub('a$','',lastPrefix)
                elif re.match('^a(\\))?nti$',lastPrefix):
                    lastPrefix = re.sub('ti$','q',lastPrefix)
            elif re.match('^[aehiouw]+\\).*',base):
                base = re.sub('\\)','',base)
                if re.match('^(ka|me)ta$',lastPrefix):
                    lastPrefix = re.sub('ta$','t',lastPrefix)
                elif re.match('^a(\\))?po$',lastPrefix) or re.match('^u(\\()?po$',lastPrefix):
                    lastPrefix = re.sub('po$','p',lastPrefix)
                elif re.match('^(dia|para)$',lastPrefix):
                    lastPrefix = re.sub('a$','',lastPrefix)
                elif re.match('^e(\\))?pi$',lastPrefix):
                    lastPrefix = re.sub('pi$','p',lastPrefix)
                elif re.match('^e(\\))?k$',lastPrefix):
                    lastPrefix = re.sub('k$','c',lastPrefix)
                elif re.match('^a(\\))?na$',lastPrefix):
                    lastPrefix = re.sub('a$','',lastPrefix)
                elif re.match('^a(\\))?nti$',lastPrefix):
                    lastPrefix = re.sub('ti$','t',lastPrefix)
            else:
                base = re.sub('\\(','',base)
                if re.match('^(sun|e(\\))?n)$',lastPrefix):
                    if re.match('^[bpfym].*',base):
                        lastPrefix = re.sub('n$','m',lastPrefix)
                    elif re.match('^[gkxc].*',base):
                        lastPrefix = re.sub('n$','g',lastPrefix)
                    elif re.match('^l.*',base):
                        lastPrefix = re.sub('n$','l',lastPrefix)
                    elif re.match('^r.*',base) and re.match('^sun$',lastPrefix):
                        lastPrefix = re.sub('n$','r',lastPrefix)
                    elif re.match('^s.*',base) and re.match('^sun$',lastPrefix):
                        lastPrefix = re.sub('n$','',lastPrefix)
                if(re.match('.*[aehiouw]$',lastPrefix) and re.match('^r.*',base)):
                    base = re.sub('^r','rr',base)
            splittedPrefix[len(splittedPrefix)-1] = lastPrefix
            if len(splittedPrefix)>0:
                if base == 'kei=mai' or base == 'h=mai'or base == 'ei=mi' or base == 'oi=da':
                    base = re.sub('=','',base)
                    splittedPrefix[len(splittedPrefix)-1] = re.sub('([aehiouw][\\)\\(]?)([^aehiouw]*)$','\\1/\\2',splittedPrefix[len(splittedPrefix)-1])
                elif base == 'eimi/' or base == 'fhmi/' or base == 'hmi/':
                    base = re.sub('/','',base)
                    splittedPrefix[len(splittedPrefix)-1] = re.sub('([aehiouw][\\)\\(]?)([^aehiouw]*)$','\\1/\\2',splittedPrefix[len(splittedPrefix)-1])
            form = ''
            for s in splittedPrefix:
                form += s
            form += base
            return form
        elif lemma == 'nu=n':
            if form == 'nun':
                return 'nun'
        elif lemma == 'ma/lista' or lemma == 'ma=llon':
            return 'ma/la'
        elif lemma == 'pe/r':
            return 'per'
        elif lemma == 'ei)=pon':
            return 'le/gw'
        elif lemma == 'e)pei/':
            if form == 'e)peidh/' or form == 'e)peidh/per':
                return 'e)peidh/'
        elif lemma == 'plei=stos':
            return 'polu/s'
        elif lemma == 'plei/wn':
            return 'polu/s'
        elif lemma == 'ei)=don':
            return 'o(ra/w'
        elif re.match('.*(kat|met|a\\)?nt)ei=don$',lemma):
            return re.sub('tei=don', 'qora/w',lemma)
        elif re.match('.*(a\\)?p|e\\)?p)ei=don$',lemma):
            return re.sub('pei=don', 'fora/w',lemma)
        elif re.match('.*ei=don$',lemma):
            return re.sub('ei=don', 'ora/w',lemma)
        elif lemma == 'ei)=mi':
            return 'e)/rxomai'
        elif re.match('.*qnh/skw$',lemma):
            return re.sub('qnh/skw', 'qnh/|skw',lemma)
        elif lemma == 'a)mei/nwn' or lemma == 'a)/ristos' or lemma == 'belti/wn' or lemma == 'be/ltistos':
            return 'a)gaqo/s'
        elif lemma == 'sautou=':
            return 'seautou='
        elif lemma == 'pou/':
            return 'pou'
        elif lemma == 'e)rw=':
            return 'le/gw'
        elif re.match('.*sw/zw$',lemma):
            return re.sub('sw/zw', 'sw/|zw',lemma)
        elif re.match('.*eimi$',lemma) and digit == '2':
            lemma_no_accent = re.sub('/','',lemma)
            return re.sub('eimi', 'e/rxomai',lemma_no_accent)
        elif re.match('.*(e\\)?/c|ei\\)?/s|peri/|ka/t|a\\)/?n)eimi$',lemma):
            lemma_no_accent = re.sub('/','',lemma)
            return re.sub('eimi', 'e/rxomai',lemma_no_accent)
        elif lemma == 'e)la/sswn' or lemma == 'e)la/xistos':
            return 'e)laxu/s'
        elif lemma == 'ka)n':
            return 'ka)/n'
        elif re.match('.*limpa/nw$',lemma):
            return re.sub('limpa/nw', 'lei/pw',lemma)
        elif re.match('.*esth/cw$',lemma):
            return re.sub('esth/cw', 'i/sthmi',lemma)
        elif lemma == 'i)xqu/s':
            return 'i)xqu=s'
        elif lemma == '*)aqh=nai':
            if re.match('^\\*\\)aqh/nhsin?$',form):
                return '*)aqh/nhsi'
            elif re.match('^\\*\\)aqh/naze/?$',form):
                return '*)aqh/naze'
        elif lemma == 'xei/rwn' or lemma == 'xei/ristos':
            return 'kako/s'
        elif re.match('.*w/xato$',lemma):
            return re.sub('w/xato', 'e/xw',lemma)
        elif lemma == 'ai)/rw':
            return 'a)ei/rw'
        elif re.match('.*(peri|pro|a\\)?mfi|pros|pro)ei=pon$',lemma):
            return re.sub('ei=pon', 'le/gw',lemma)
        elif re.match('.*a\\)?pei=pon$',lemma):
            return re.sub('ei=pon', 'ole/gw',lemma)
        elif re.match('.*(kat|met|di|par|a\\)?n)ei=pon$',lemma):
            return re.sub('ei=pon', 'ale/gw',lemma)
        elif re.match('.*(e\\)?p|a\\)?nt)ei=pon$',lemma):
            return re.sub('ei=pon', 'ile/gw',lemma)
        elif re.match('.*e\\)?cei=pon$',lemma):
            return re.sub('cei=pon', 'kle/gw',lemma)
        elif re.match('.*(sun|e\\)n)ei=pon$',lemma):
            return re.sub('nei=pon', 'lle/gw',lemma) 
        elif lemma == 'pantaxh=':
            return 'pantaxh=|'
        elif lemma == 'krei/sswn' or lemma == 'kra/tistos':
            return 'kratu/s'
        elif lemma == 'fi/ltatos':
            return 'fi/los'
        elif lemma == 'e)/xqistos':
            return 'e)xqro/s'
        elif lemma == 'gai=a':
            return 'gh='
        elif lemma == 'toiga/r' and form == 'toigarou=n':
            return 'toigarou=n'
        elif re.match('.*ske/ptomai$',lemma):
            return re.sub('ske/ptomai', 'skope/w',lemma)
        elif lemma == 'ou(twsi/':
            return 'ou(/tws'
        elif re.match('.*mimnh/skw$',lemma):
            return re.sub('mimnh/skw', 'mimnh/|skw',lemma)
        elif re.match('.*pi/tnw$',lemma):
            return re.sub('pi/tnw', 'pi/ptw',lemma)
        elif lemma == 'e)/qw':
            return 'ei)/wqa'
        elif lemma == 'kh=r':
            return 'ke/ar'
        elif lemma == 'w(/sper' and re.match('^w\\(sperei[/\\\\]$',form):
            return 'w(sperei/'
        elif form == 'e)a/nte' and lemma == 'e)a/n':
            return 'e)a/nte'
        elif lemma == 'e)xqe/s':
            return 'xqe/s'
        elif re.match('.*allacei/w$',lemma):
            return re.sub('allacei/w', 'alla/ssw',lemma)
        elif re.match('.*mi/gnumi$',lemma):
            return re.sub('mi/gnumi', 'mei/gnumi',lemma)
        elif lemma == 'filoneiki/a':
            return 'filoniki/a'
        elif lemma == 'proswte/rw':
            return 'pro/sw'
        elif lemma == 'oi)ktei/rw':
            return 'oi)kti/rw'
        elif re.match('.*gei/nomai$',lemma):
            return re.sub('gei/nomai', 'gi/gnomai',lemma)
        elif lemma == 'au)qh/meros' and re.match('^au\\)qhmero[/\\\\]n$',form):
            return 'au)qhmero/n'
        elif lemma == 'o)sfu/s':
            return 'o)sfu=s'
        elif lemma == '*peiqw/' and not re.match('^\\*.*',form):
            return 'peiqw/'
        elif re.match('.*o/xwka$',lemma):
            return re.sub('o/xwka', 'e/xw',lemma)
        elif lemma == '*mou=sai':
            return '*mou=sa'
        elif re.match('.*(peri|pro|a\\)?mfi|pros|pro)ere/w$',lemma):
            return re.sub('ere/w', 'le/gw',lemma)
        elif re.match('.*a\\)?pere/w$',lemma):
            return re.sub('ere/w', 'ole/gw',lemma)
        elif re.match('.*(kat|met|di|par|a\\)?n)ere/w$',lemma):
            return re.sub('ere/w', 'ale/gw',lemma)
        elif re.match('.*(e\\)?p|a\\)?nt)ere/w$',lemma):
            return re.sub('ere/w', 'ile/gw',lemma)
        elif re.match('.*e\\)?cere/w$',lemma):
            return re.sub('cere/w', 'kle/gw',lemma)
        elif re.match('.*(sun|e\\)n)ere/w$',lemma):
            return re.sub('nere/w', 'lle/gw',lemma)
        elif lemma == 'grai/dion':
            return 'grai+/dion'
        elif lemma == 'proi/sthmi':
            return 'proi+/sthmi'
        elif lemma == 'prw/tistos':
            return 'prw=tos'
        elif lemma == '*ma/gos' and not re.match('^\\*.*',form):
            return 'ma/gos'
        elif re.match('.*pi/plhmi$',lemma):
            return re.sub('pi/plhmi', 'pi/mplhmi',lemma)
        elif re.match('.*store/nnumi$',lemma):
            return re.sub('store/nnumi', 'sto/rnumi',lemma)
        elif form == 'a(plw=s' and lemma == 'a(plo/s':
            return 'a(plw=s'
        elif lemma == 'e)lea/w':
            return 'e)lee/w'
        elif lemma == 'e)pei/per':
            return 'e)pei/'
        elif lemma == 'eu)qh/s':
            return 'eu)qu/s'
        elif lemma == 'qa/sswn':
            return 'taxu/s'
        elif lemma == 'qea/w':
            return 'qea/omai'
        elif lemma == '*karxhdw/n' and re.match('.*do.?ni.*',form):
            return '*karxhdo/nios'
        elif lemma == 'lagw=s' or lemma == 'lagwo/s':
            return 'lagw/s'
        elif lemma == 'o(/loc':
            return 'o(/los'
        elif lemma == 'pa/rergon' and form == 'pare/rgws':
            return 'pa/rergos'
        elif lemma == 'parista/w':
            return 'pari/sthmi'
        elif form == 'poi=' and lemma == 'poi':
            return 'poi='
        elif form == 'pou=' and lemma == 'pou':
            return 'pou='
        elif lemma == 'xalkou=s':
            return 'xa/lkeos'
        elif lemma == '*)/adrhstos':
            return '*)/adrastos'
        elif lemma == 'o(/sper':
            return 'o(/s'
        elif lemma == 'a)lhqw=s':
            return 'a)lhqh/s'
        elif re.match('.*mimnh/skw',lemma):
            return re.sub('mimnh/skw','mimnh/|skw',lemma)
        elif lemma == 'a)nqista/w':
            return 'a)nqi/sthmi'
        elif re.match('.*limpa/nw',lemma):
            return re.sub('limpa/nw','lei/pw',lemma)
        elif lemma == 'dei=':
            return 'de/w'
        elif lemma == 'e)c':
            return 'e)k'
        elif re.match('.*ske/ptomai',lemma):
            return re.sub('ske/ptomai','skope/w',lemma)
        elif lemma == 'h(/kistos':
            return 'h)=ka'
        elif form == 'h)/per':
            return 'h)/'
        elif re.match('.*qnh/skw',lemma):
            return re.sub('qnh/skw','qnh/|skw',lemma)
        elif lemma == 'i)/laos':
            return 'i(/laos'
        elif lemma == 'katayeu/dw':
            return 'katayeu/domai'
        elif lemma == 'makra/n':
            return 'makro/s'
        elif form == 'nun' and lemma == 'nu=n':
            return 'nun'
        elif lemma == 'o(/ph':
            return 'o(/ph|'
        elif lemma == 'pai/dion':
            return 'paidi/on'
        elif lemma == 'paidih/':
            return 'paidia/'
        elif lemma == 'pampolu/s':
            return 'pa/mpolus'
        elif lemma == 'peri/plous':
            return 'peri/ploos'
        elif re.match('.*pe/rnhmi',lemma):
            return re.sub('pe/rnhmi','pipra/skw',lemma)
        elif lemma == 'pw=s' and form == 'pw/s':
            return 'pws'
        elif lemma == 'pws' and form == 'pw=s':
            return 'pw=s'
        elif lemma == 'r(a|qumi/a':
            return 'r(aqumi/a'
        elif lemma == 'tote/':
            return 'to/te'
        elif lemma == 'u(podu/omai':
            return 'u(podu/w'
        elif lemma == 'a)nagkai/h':
            return 'a)na/gkh'
        elif lemma == 'a)/reios':
            return '*)/areios'
        elif lemma == 'a)texnw=s':
            return 'a)/texnos'
        elif lemma == 'a)tima/w':
            return 'a)tima/zw'
        elif lemma == 'bia/w':
            return 'bia/zw'
        elif lemma == '*gai=a' and not re.match('\\*.*',form):
            return 'gh='
        elif lemma == 'glukero/s':
            return 'gluku/s'
        elif lemma == 'de/ndreon':
            return 'de/ndron'
        elif lemma == '*dionu/sos':
            return '*dio/nusos'
        elif lemma == 'doru/foros':
            return 'dorufo/ros'
        elif lemma == 'e)/dw':
            return 'e)sqi/w'
        elif lemma == 'ei)ko/s' or lemma == 'ei)ko/tws':
            return 'e)/oika'
        elif lemma == 'e)nergh/s':
            return 'e)nergo/s'
        elif lemma == 'peira/zw':
            return 'peira/w'
        elif lemma == 'pia/zw':
            return 'pie/zw'
        elif lemma == 'r(ipte/w':
            return 'r(i/ptw'
        elif lemma == 'r(u/omai':
            return 'e)ru/w'
        elif lemma == 'zw/i+on':
            return 'zw=|on'
        elif lemma == 'zw/w':
            return 'za/w'
        elif lemma == 'a)nia/zw':
            return 'a)nia/w'
        elif lemma == 'h)w=qen':
            return 'e(/wqen'
        elif lemma == 'i(ro/s':
            return 'i(ero/s'
        elif lemma == 'katakei/w':
            return 'kata/keimai'
        elif lemma == 'kate/dw':
            return 'katesqi/w'
        elif lemma == 'lhi+sth/s':
            return 'lh|sth/s'
        elif lemma == '*lilubai=on':
            return '*lilu/baion'
        elif lemma == 'nea/niskos':
            return 'neani/skos'
        elif lemma == 'neurh/':
            return 'neura/'
        elif lemma == 'pa/trh':
            return 'pa/tra'
        elif lemma == '*sura/kousa':
            return '*sura/kousai'
        elif lemma == 'te/tratos':
            return 'te/tartos'
        elif lemma == '*turshno/s':
            return '*turrhno/s'
        elif lemma == 'fulako/s':
            return 'fu/lac'
        elif lemma == 'w)|dh/':
            return 'a)oidh/'
        elif lemma == 'ple/os':
            return 'ple/ws'
        elif lemma == 'r(ino/n':
            return 'r(ino/s'
        elif lemma == 'a)dino/s':
            return 'a(dino/s'
        elif lemma == '*ko/lxos' and re.match('\\*kolxi/d.*',form):
            return '*kolxi/s'
        elif lemma == '*ku/zikos' and re.match('\\*kuzikhn.*',form):
            return '*kuzikhno/s'
        elif lemma == 'staqmo/n':
            return 'staqmo/s'
        elif lemma == '*qeo/s':
            return 'qeo/s'
        elif lemma == 'tau)to/s':
            return 'au)to/s'
        elif lemma == 'Curius':
            return 'ku/rios'
        elif form == 'menta)/n' or form == 'me/nta)n':
            return 'a)/n'
        elif lemma == '*(/aidhs':
            return '*(/|adhs'
        elif lemma == 'pro/seimi' and not re.match(r'pro/?si/?sqi',form) and (re.match(r'pro/?s(i|h[/=]\|).*',form) or form == 'pro/siqi'):
            # pro/seimi2 not defined. Fix at source?
            return 'prose/rxomai'
        elif lemma == '*phnelo/ph':
            return '*phnelo/peia'
        elif lemma == '*mwush=s':
            return '*mwu+sh=s'
        return lemma
    
    def find_word_class(self,word,stemtypes):
        wordclass = word
        f = open(stemtypes,'r',encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            splittedLine = line.split(' ')
            if len(splittedLine) == 3:
                if splittedLine[0] == word:
                    if re.match('^noun.*',splittedLine[2]):
                        wordclass = 'noun'
                    elif re.match('^adj.*',splittedLine[2]) and not re.match('^irreg_adj[13]$',splittedLine[0]) and splittedLine[0] != 'art_adj':
                        wordclass = 'adj'
                    elif re.match('^pp.*',splittedLine[2]):
                        wordclass = 'verb'
        f.close()
        return wordclass
    
    def send_word_list(self,wordlist,normalize_beta=True,unicode_normalization=None,case_sensitive=True):
        if case_sensitive:
            p = Popen(["cruncher","-d"], stdout=PIPE,stdin=PIPE)
        else:
            p = Popen(["cruncher","-d","-S"], stdout=PIPE,stdin=PIPE)
        command = ''
        for word in wordlist:
            if normalize_beta:
                if unicode_normalization is not None:
                    word = normalize_token(word,unicode_normalization)
                word = beta_code.greek_to_beta_code(word)
                word = re.sub('[’᾽]','\'',word)
            command += word + '\n'
        morpheus_output = p.communicate(command.encode('UTF-8'))[0]
        return morpheus_output.decode('UTF-8')
    
    def convert_morpheus_output(self,morpheus_analysis,poetic,print_problems=False):
        lexicon = {}
        lines = morpheus_analysis.split('\n')
        form = ''
        lemma = ''
        pos = ''
        stem = ''
        infl = ''
        prefix = ''
        augment = ''
        ending = ''
        lemma_morpheus = ''
        discardAnalysis = False
        descriptors = set()
        non_accented = ['min','e(/','o(','tis','sfei=s','e)gw/','su/','nin','e)gw/ge']
        for line in lines:
            if re.match('.*epic.*',line):
                descriptors.add('epic')
            if re.match('.*attic.*',line):
                descriptors.add('attic')
            if re.match('.*doric.*',line):
                descriptors.add('doric') 
            if re.match('.*ionic.*',line):
                descriptors.add('ionic')
            if re.match('.*aeolic.*',line):
                descriptors.add('aeolic')
            if re.match('.*poetic.*',line):
                descriptors.add('poetic')
            if re.match('.*late.*',line):
                descriptors.add('late')
            if re.match('.*laconia.*',line):
                descriptors.add('laconia')
            if re.match('.*boeotia.*',line):
                descriptors.add('boeotia')
            if re.match('^:raw.*',line):
                form = re.sub(':raw[ ]+','',line)
                form = re.sub('[’᾽]','\'',form)
                form = re.sub(r'\\','/',form)
                form = re.sub('([/=])(.*)/',r'\1\2',form)
                form = re.sub('s1\'','s\'',form)
            elif re.match('^:lem.*',line):
                lemma = re.sub(':lem[ ]+','',line)
                lemma_morpheus=re.sub('-pl','',lemma)
                lemma = self.regularize_lemma(lemma,form)
            elif re.match('^:prvb.*',line):
                prefix = re.sub(':prvb ','',line,1)
                prefix = re.sub('\t.*','',prefix,1)
            elif re.match('^:aug1.*',line):
                augment = re.sub(':aug1 ','',line,1)
                augment = re.sub('\t.*','',augment,1)
            elif re.match('^:stem.*',line):
                stem = re.sub(':stem[ ]+', '',line.split('\t')[0])
                if re.match('.*[ier](_)?$',stem):
                    ier = True
                else:
                    ier = False
                if re.match('^:stem.*(doric|aeolic|homeric|poetic).*',line) and not re.match('.*(ionic|attic).*',line) and not poetic:
                    discardAnalysis = True
                if re.match('.*irreg_superl.*',line):
                    superlative = True
                else:
                    superlative = False
                if re.match('.*irreg_comp.*',line):
                    comparative = True
                else:
                    comparative = False
            elif re.match('^:end.*',line):
                dialects = ','.join(descriptors)
                descriptors.clear()
                ending = re.sub(':end ','',line,1)
                ending = re.sub('\t.*', '',ending,1)
                if re.match('.*(doric|aeolic|homeric|poetic).*',line) and not re.match('.*(ionic|attic).*',line) and not poetic:
                    discardAnalysis = True
                elif re.match('.*(ionic|attic).*',line):
                    discardAnalysis = False
                if re.match('.*futperf.*(nom|voc|acc|gen|dat).*',line) or re.match('.*adverbial (comp|superl).*',line) or re.match('.*unaugmented.*',line):
                    discardAnalysis = True
                if discardAnalysis==True and ier==True and (re.match('.*fem .* sg.*',line) or re.match('.*masc .* sg.*hs_ou$',line)):
                    discardAnalysis = False
                if not discardAnalysis:
                    splittedLine = line.split('\t')
                    infl = splittedLine[len(splittedLine)-1]
                    morphology = re.sub('^[ ]+', '',splittedLine[1])
                    wordclass = splittedLine[len(splittedLine)-1]
                    if "," in wordclass:
                        wordclass = wordclass.split(',')[0]
                    wordclass = self.find_word_class(wordclass, self.stemtypes)
                    if (morphology == 'adverbial' or morphology == 'comp' or lemma == 'pote/' or lemma == 'ma/' or lemma == 'o(/mws' or lemma == 'dio/per') and not (lemma == 'i(/na' or lemma == 'o(/ti' or lemma == 'i)dou/' or lemma == 'h(ni/ka' or lemma == 'eu)=ge' or lemma == 'nai/' or lemma == 'ou(=' or lemma == 'kaqo/ti'):
                        pos = 'adverb'
                    elif wordclass == 'indef' or lemma == 'dei=na' or lemma == 'ou)dei/s' or lemma == 'mhdei/s' or lemma == 'poio/s' or lemma == 'poso/s' or lemma == 'e(/kastos' or lemma == 'e(ka/teros' or lemma == 'pa=s' or lemma == 'a(/pas' or lemma == 'e(/teros' or lemma == 'ou)de/teros' or lemma == 'mhde/teros' or lemma == 'a)/llos' or (lemma == 'mh/tis' and wordclass == 'pron_adj3') or lemma == 'ou)/tis' or lemma == 'a)/mfw':
                        pos = 'indefinite'
                    elif wordclass == 'pron1' or wordclass == 'pron3' or lemma == 'a)llh/lwn' or lemma == 'e(autou=':
                        pos = 'personal'
                    elif wordclass == 'art_adj' or wordclass == 'demonstr' or lemma == 'ou(=tos' or lemma == 'thliko/sde' or lemma == 'thlikou=tos' or lemma == 'toi=os' or lemma == 'to/sos' or lemma == 'thli/kos' or lemma == 'tau)to/s':
                        pos ='demonstrative'
                    elif wordclass == 'relative' or lemma == 'o(/sos' or lemma == 'oi(=os' or lemma == 'o(poi=os' or lemma == 'h(li/kos' or lemma == 'o(po/sos' or lemma == 'o(po/teros' or lemma == 'o(phli/kos':
                        pos = 'relative'
                    elif lemma == 'ti/s' or lemma == 'phli/kos' or lemma == 'poi=os' or lemma == 'po/sos' or lemma == 'po/teros': 
                        pos = 'interrogative'
                    elif wordclass == 'numeral' or lemma == 'ei(=s' or lemma == 'te/ssares' or re.match('.*ko/sioi$',lemma) or re.match('.*xi/lioi$',lemma) or re.match('.*mu/rioi$',lemma):
                        pos = 'numeral'
                    elif wordclass == 'exclam' or wordclass == 'expletive' or lemma == 'i)de/' or lemma == 'nai/' or lemma == 'i)dou/' or lemma == 'eu)=ge':
                        pos = 'interjection'
                    elif lemma == 'kai/' or lemma == 'de/' or lemma == 'te' or lemma == 'a)lla/' or lemma == 'ou)de/' or lemma == 'mhde/' or lemma == 'mh/te' or lemma == 'ou)/te':
                        pos = 'coordinator'
                    elif wordclass == 'conj' or wordclass == 'connect' or lemma == 'i(/na' or lemma == 'o(/ti' or lemma == 'h(ni/ka' or lemma == 'ou(=' or lemma == 'kaqo/ti':
                        pos = 'conjunction'
                    elif wordclass == 'particle' or lemma == 'a)=ra' or lemma == 'dh=ta' or lemma == 'h)=':
                        pos = 'particle'
                    elif wordclass == 'adverb' or wordclass == 'interrog' or (wordclass == 'adj' and morphology == 'adverbial'):
                        pos = 'adverb'
                    elif wordclass == 'noun' or wordclass == 'alphabetic' or wordclass == 'irreg_decl3' or wordclass == 'indecl' or wordclass == 'indecl_noun':
                        pos = 'noun'
                    elif wordclass == 'adj' or wordclass == 'irreg_adj1' or wordclass == 'irreg_adj3':
                        pos = 'adjective'
                    elif wordclass == 'verb':
                        if re.match('.*part.*',morphology):
                            pos = 'participle'
                        elif re.match('.*inf.*',morphology):
                            pos = 'infinitive'
                        else:
                            pos = 'verb'
                    elif wordclass == 'article':
                        pos = 'article'
                    elif wordclass == 'prep':
                        pos = 'preposition'
                    else:
                        pos = 'unknown'
                        print('POS not found: '+lemma+', '+wordclass)
                    
                    if pos == 'noun' or pos == 'adjective' or pos == 'article' or pos == 'personal' or pos == 'demonstrative' or pos == 'relative' or pos == 'indefinite' or pos == 'interrogative' or pos == 'numeral':
                        genders = []
                        cases = []
                        number = ''
                        degree = ''
                        if (pos == 'personal' and (lemma == 'e)gw/' or lemma == 'su/' or lemma == 'h(mei=s' or lemma == 'u(mei=s' or lemma == 'e(/' or lemma == 'sfei=s')) or (pos == 'indefinite' and ((lemma == 'tis' and not re.match('.*neut.*',morphology)) or lemma == 'dei=na' or lemma == 'mh/tis' or lemma == 'ou/tis')) or (pos == 'interrogative' and (lemma == 'ti/s' and not re.match('.*neut.*',morphology))):
                            genders.append('comm')
                        elif pos == 'numeral' and not lemma == 'ei(=s' and not re.match('.*ko/sioi$',lemma) and not re.match('.*xi/lioi$',lemma) and not re.match('.*mu/rioi$',lemma):
                            genders.append('none')
                        else:
                            if re.match('.*masc.*',morphology):
                                genders.append('masc')
                            if re.match('.*fem.*',morphology):
                                genders.append('fem')
                            if re.match('.*neut.*',morphology):
                                genders.append('neut')
                        
                        if re.match('.*((wn_on)|(hn_enos)|(ws_wn)|(is_idos)).*',line) and len(genders)==0:
                            genders.append('fem')
                            genders.append('masc')
                            genders.append('neut')
                        
                        if re.match('.*nom.*',morphology):
                            cases.append('nom')
                        if re.match('.*voc.*',morphology):
                            cases.append('voc')
                        if re.match('.*acc.*',morphology):
                            cases.append('acc')
                        if re.match('.*gen.*',morphology):
                            cases.append('gen')
                        if re.match('.*dat.*',morphology):
                            cases.append('dat')
                        
                        if pos == 'numeral' and len(cases)==0:
                            cases.append('none')
                            
                        if re.match('.*sg.*',morphology):
                            number='sg'
                        if re.match('.*pl.*',morphology):
                            number='pl'
                        if re.match('.*dual.*',morphology):
                            number='dual'
                        
                        if pos == 'numeral' and number == '':
                            number = 'none'
                        
                        if re.match('.*superl.*',morphology):
                            degree='sup'
                        if re.match('.*comp.*',morphology):
                            degree='comp'
                        
                        if pos=='adjective':
                            if degree=='':
                                degree='pos'
                        
                        if pos == 'adjective' and superlative==True:
                            degree = 'sup'
                        if pos == 'adjective' and comparative==True:
                            degree = 'comp'
                        
                        if (lemma == 'mhdei/s' or lemma == 'ou)dei/s') and len(genders)==0:
                            genders.append('masc')
                            genders.append('neut')

                        if len(genders)==0:
                            if print_problems:
                                print('No gender: '+form)
                            genders.append('none')
                        if len(cases)==0:
                            if print_problems:
                                print('No case: '+form+'\t'+morphology)
                            cases.append('none')
                        if number == '':
                            if print_problems:
                                print('No number: '+form+'\t'+morphology)
                            number = 'sg'
                        
                        form_uni = self.beta_to_uni(form)
                        form_uni = re.sub('\'','’',form_uni)
                        lemma_uni = self.beta_to_uni(lemma)
                        
                        for gender in genders:
                            for ncase in cases:
                                tag = []
                                tag.append(('XPOS',pos))
                                tag.append(('person','_'))
                                tag.append(('number',number))
                                tag.append(('tense','_'))
                                tag.append(('mood','_'))
                                tag.append(('voice','_'))
                                tag.append(('gender',gender))
                                tag.append(('case',ncase))
                                if pos == 'adjective':
                                    tag.append(('degree',degree))
                                else:
                                    tag.append(('degree','_'))
                                tag.append(('lemma',lemma_uni))
                                tag.append(('infl',infl))
                                tag.append(('prefix',prefix))
                                tag.append(('stem',stem))
                                tag.append(('augment',augment))
                                tag.append(('ending',ending))
                                tag.append(('dialects',dialects))
                                if self.word_accents is not None:
                                    key = lemma_morpheus+'€'+re.sub('\\|','i',stem)
                                    if not key in self.word_accents:
                                        tag.append(('accent','NA'))
                                        if not re.match('.*[/=].*',stem) and not lemma_morpheus in non_accented:
                                            print('accent class not found: '+lemma_morpheus+' '+stem)
                                    else:
                                        accent = self.word_accents[key]
                                        tag.append(('accent',accent))
                                tag = tuple(tag)
                                if form_uni in lexicon:
                                    tags = lexicon[form_uni]
                                    if tag not in tags:
                                        tags.append(tag)
                                else:
                                    tags = []
                                    tags.append(tag)
                                    lexicon[form_uni] = tags
                    elif pos == 'participle':
                        genders = []
                        cases = []
                        number = ''
                        tense = ''
                        voice = ''
                        if re.match('.*masc.*',morphology):
                            genders.append('masc')
                        if re.match('.*fem.*',morphology):
                            genders.append('fem')
                        if re.match('.*neut.*',morphology):
                            genders.append('neut')

                        if re.match('.*nom.*',morphology):
                            cases.append('nom')
                        if re.match('.*voc.*',morphology):
                            cases.append('voc')
                        if re.match('.*acc.*',morphology):
                            cases.append('acc')
                        if re.match('.*gen.*',morphology):
                            cases.append('gen')
                        if re.match('.*dat.*',morphology):
                            cases.append('dat')

                        if re.match('.*sg.*',morphology):
                            number='sg'
                        if re.match('.*pl$',morphology) or re.match('.* pl\\s.*',morphology):
                            number='pl'
                        if re.match('.*dual.*',morphology):
                            number='dual'

                        if re.match('.*aor.*',morphology):
                            tense='aor'
                        if re.match('.*pres.*',morphology):
                            tense='pres'
                        if re.match('.*fut.*',morphology):
                            tense='fut'
                        if re.match('.*perf.*',morphology):
                            tense='pf'

                        if re.match('.*act.*',morphology):
                            voice='act'
                        if re.match('.* (mid|mp) .*',morphology):
                            voice='mid'
                        if re.match('.*pass.*',morphology):
                            voice='pass'
                        
                        if len(genders)==0:
                            if print_problems:
                                print('No gender: '+form)
                            genders.append('none')
                        if len(cases)==0:
                            if print_problems:
                                print('No case: '+form)
                            cases.append('none')
                        if number == '':
                            if print_problems:
                                print('No number: '+form)
                            number = 'none'
                        if tense == '':
                            if print_problems:
                                print('No tense: '+form)
                            tense = 'pres'
                        if voice == '':
                            if print_problems:
                                print('No voice: '+form)
                            voice = 'act'
                        
                        form_uni = self.beta_to_uni(form)
                        form_uni = re.sub('\'','’',form_uni)
                        lemma_uni = self.beta_to_uni(lemma)
                        
                        for gender in genders:
                            for ncase in cases:
                                tag = []
                                tag.append(('XPOS',pos))
                                tag.append(('person','_'))
                                tag.append(('number',number))
                                tag.append(('tense',tense))
                                tag.append(('mood','_'))
                                tag.append(('voice',voice))
                                tag.append(('gender',gender))
                                tag.append(('case',ncase))
                                tag.append(('degree','_'))
                                tag.append(('lemma',lemma_uni))
                                tag.append(('infl',infl))
                                tag.append(('prefix',prefix))
                                tag.append(('stem',stem))
                                tag.append(('augment',augment))
                                tag.append(('ending',ending))
                                tag.append(('dialects',dialects))
                                tag.append(('accent','NA'))
                                tag = tuple(tag)
                                if form_uni in lexicon:
                                    tags = lexicon[form_uni]
                                    if tag not in tags:
                                        tags.append(tag)
                                else:
                                    tags = []
                                    tags.append(tag)
                                    lexicon[form_uni] = tags
                    elif pos == 'infinitive':
                        tense = ''
                        voice = ''

                        if re.match('.*aor.*',morphology):
                            tense='aor'
                        if re.match('.*pres.*',morphology):
                            tense='pres'
                        if re.match('.*fut.*',morphology):
                            tense='fut'
                        if re.match('.*perf.*',morphology):
                            tense='pf'

                        if re.match('.*act.*',morphology):
                            voice='act'
                        if re.match('.* (mid|mp).*',morphology):
                            voice='mid'
                        if re.match('.*pass.*',morphology):
                            voice='pass'    

                        if tense == '':
                            if print_problems:
                                print('No tense: '+form)
                            tense = 'pres'
                        if voice == '':
                            if print_problems:
                                print('No voice: '+form)
                            voice = 'act'
                        
                        form_uni = self.beta_to_uni(form)
                        form_uni = re.sub('\'','’',form_uni)
                        lemma_uni = self.beta_to_uni(lemma)
                        
                        tag = []
                        tag.append(('XPOS',pos))
                        tag.append(('person','_'))
                        tag.append(('number','_'))
                        tag.append(('tense',tense))
                        tag.append(('mood','_'))
                        tag.append(('voice',voice))
                        tag.append(('gender','_'))
                        tag.append(('case','_'))
                        tag.append(('degree','_'))
                        tag.append(('lemma',lemma_uni))
                        tag.append(('infl',infl))
                        tag.append(('prefix',prefix))
                        tag.append(('stem',stem))
                        tag.append(('augment',augment))
                        tag.append(('ending',ending))
                        tag.append(('dialects',dialects))
                        tag.append(('accent','NA'))
                        tag = tuple(tag)
                        if form_uni in lexicon:
                            tags = lexicon[form_uni]
                            if tag not in tags:
                                tags.append(tag)
                        else:
                            tags = []
                            tags.append(tag)
                            lexicon[form_uni] = tags

                    elif pos == 'verb':
                        pos = 'verb'

                        person = ''
                        number = ''
                        tense = ''
                        mood = ''
                        voice = ''

                        if re.match('.*1.*',morphology):
                            person='1'
                        if re.match('.*2.*',morphology):
                            person='2'
                        if re.match('.*3.*',morphology):
                            person='3'

                        if re.match('.*sg.*',morphology):
                            number='sg'
                        if re.match('.*pl$',morphology) or re.match('.* pl\\s.*',morphology):
                            number='pl'
                        if re.match('.*dual.*',morphology):
                            number='dual'

                        if re.match('^aor.*',morphology):
                            tense='aor'
                        if re.match('^pres.*',morphology):
                            tense='pres'
                        if re.match('^fut .*',morphology):
                            tense='fut'
                        if re.match('^perf .*',morphology):
                            tense='pf'
                        if re.match('^imperf.*',morphology):
                            tense='impf'
                        if re.match('^plup.*',morphology):
                            tense='plupf'
                        if re.match('^futperf.*',morphology):
                            tense='futpf'

                        if re.match('.*ind.*',morphology):
                            mood='ind'
                        if re.match('.*subj.*',morphology):
                            mood='subj'
                        if re.match('.*opt.*',morphology):
                            mood='opt'
                        if re.match('.*imperat.*',morphology):
                            mood='imp'

                        if re.match('.*act.*',morphology):
                            voice='act'
                        if re.match('.* (mid|mp).*',morphology):
                            voice='mid'
                        if re.match('.*pass.*',morphology):
                            voice='pass'
                        
                        if splittedLine[len(splittedLine)-1] == 'aor_pass':
                            voice = 'pass'
                        
                        if mood == 'imp' and person == '':
                            person = '2'
                        if mood == 'imp' and number == '':
                            number = 'sg'
                        
                        if person == '':
                            if print_problems:
                                print('No person: '+form)
                            person = '3'
                        if number == '':
                            if print_problems:
                                print('No number: '+form)
                            number = 'sg'
                        if tense == '':
                            if print_problems:
                                print('No tense: '+form)
                            tense = 'pres'
                        if mood == '':
                            if print_problems:
                                print('No mood: '+form)
                            mood = 'ind'
                        if voice == '':
                            if print_problems:
                                print('No voice: '+form+'\n')
                            voice = 'act'
                        
                        form_uni = self.beta_to_uni(form)
                        form_uni = re.sub('\'','’',form_uni)
                        lemma_uni = self.beta_to_uni(lemma)
                        tag = []
                        tag.append(('XPOS',pos))
                        tag.append(('person',person))
                        tag.append(('number',number))
                        tag.append(('tense',tense))
                        tag.append(('mood',mood))
                        tag.append(('voice',voice))
                        tag.append(('gender','_'))
                        tag.append(('case','_'))
                        tag.append(('degree','_'))
                        tag.append(('lemma',lemma_uni))
                        tag.append(('infl',infl))
                        tag.append(('prefix',prefix))
                        tag.append(('stem',stem))
                        tag.append(('augment',augment))
                        tag.append(('ending',ending))
                        tag.append(('dialects',dialects))
                        tag.append(('accent','NA'))
                        tag = tuple(tag)
                        if form_uni in lexicon:
                            tags = lexicon[form_uni]
                            if tag not in tags:
                                tags.append(tag)
                        else:
                            tags = []
                            tags.append(tag)
                            lexicon[form_uni] = tags

                    else:
                        all_pos = []
                        all_pos.append(pos)
                        degree = '_'
                        if pos == 'adverb':
                            if superlative==True:
                                degree = 'sup'
                            elif comparative==True:
                                degree = 'comp'
                            else:
                                degree = 'pos'
                        if form == 'w(/sper' or form == 'w(/ste':
                            all_pos.append('conjunction')
                        elif form == 'a)/nw' or re.match('.*a/nw$',form) or form == 'a(/ma':
                            all_pos.append('preposition')
                        elif form == 'mh/':
                            all_pos.append('adverb')
                            degree = 'pos'
                        elif form == 'w(s':
                            all_pos.append('preposition')
                            all_pos.append('adverb')
                            degree = 'pos'
                        elif lemma == 'xa/ris':
                            lemma = 'xa/rin'
                        elif lemma == 'plhsi/os':
                            lemma = 'plhsi/on'
                        elif lemma == 'e)nw/pios':
                            lemma = 'e)nw/pion'
                        elif lemma == 'h)/' or lemma == 'h)de/':
                            all_pos.append('coordinator')
                        elif lemma== 'ma/la' and re.match('.*is[tq].*',form):
                            degree = 'sup';
                        elif lemma == 'ma/la' and re.match('.*a=ll.*',form):
                            degree = 'comp'                       
                        
                        form_uni = self.beta_to_uni(form)
                        form_uni = re.sub('\'','’',form_uni)
                        lemma_uni = self.beta_to_uni(lemma)
                        
                        for poss_pos in all_pos:
                            tag = []
                            tag.append(('XPOS',poss_pos))
                            tag.append(('person','_'))
                            tag.append(('number','_'))
                            tag.append(('tense','_'))
                            tag.append(('mood','_'))
                            tag.append(('voice','_'))
                            tag.append(('gender','_'))
                            tag.append(('case','_'))
                            if poss_pos == 'adverb':
                                tag.append(('degree',degree))
                            else:
                                tag.append(('degree','_'))
                            tag.append(('lemma',lemma_uni))
                            tag.append(('infl',infl))
                            tag.append(('prefix',prefix))
                            tag.append(('stem',stem))
                            tag.append(('augment',augment))
                            tag.append(('ending',ending))
                            tag.append(('dialects',dialects))
                            tag.append(('accent','NA'))
                            tag = tuple(tag)
                            if form_uni in lexicon:
                                tags = lexicon[form_uni]
                                if tag not in tags:
                                    tags.append(tag)
                            else:
                                tags = []
                                tags.append(tag)
                                lexicon[form_uni] = tags

                else:
                    discardAnalysis = False
        
        return lexicon

def main():
    
    wordlist = ['le/gw','gra/fw']
    processor = MorpheusProcessor()
    print(processor.stemtypes)
    morpheus_output = processor.send_word_list(wordlist)
    lexicon = processor.convert_morpheus_output(morpheus_output, False)
    print(lexicon)
    

if __name__ == '__main__':
    main()