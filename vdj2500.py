import typing
import os
import gzip
import json
import requests
import pandas as pd

import MeCab
import platform
#isColab = platform.system() == 'Linux'
#if isColab:
#    !pip install 'konoha[mecab]'
import jaconv
from konoha import SentenceTokenizer

class VDJ2500():
    '''日本語を勉強する人のための語彙データベース: Basic 2500'''

    def __init__(self, 
                 excel_fname:str=None, 
                 reload:bool=False,
                 wakati=None,
                 splitter=None,
                ):

        mecab_dic_dirs = { # MeCab のインストール場所の相違. 浅川の個人設定の差分吸収のため
            # 'Sinope':' /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/',
            # 'Pasiphae': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
            # 'Leda': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
            'Sinope':' /opt/homebrew/lib/mecab/dic/ipadic',
            'Pasiphae': '/usr/local/lib/mecab/dic/ipadic',
            'Leda': '/usr/local/lib/mecab/dic/ipadic',
            'colab': '/usr/share/mecab/dic/ipadic'
        }

        isColab = platform.system() == 'Linux'
        hostname = 'colab' if isColab else os.uname().nodename.split('.')[0] 
        mecab_dic_dir = mecab_dic_dirs[hostname]
        if wakati == None:
            self.wakati = MeCab.Tagger(f'-Owakati').parse
        else:
            self.wakati = wakati
            
        if splitter == None:
            self.splitter = SentenceTokenizer()
        else:
            self.splitter = splitter
        
        self.max_length=10
        
        self.data_fname = './2022_0206vdj2500.gz'
        url='http://www17408ui.sakura.ne.jp/tatsum/database/VDLJ_Ver1_0_General-Learners_Basic-2500.xlsx'
        data_fname = '2022_0206vdj2500.gz'
        if excel_fname == None:
            excel_fname = url.split('/')[-1].replace('.xlsx','')

        if reload:
            if not os.path.exists(excel_fname):
                r = requests.get(url)
                with open(excel_fname, 'wb') as f:
                    total_length = int(r.headers.get('content-length'))
                    print('Downloading {0} - {1} bytes'.format(excel_fname, (total_length)))
                    f.write(r.content)
                
            df = pd.read_excel(excel_fname, sheet_name='基本語2500　Basic 2500 Words')
            df = df[['ふつうの（新聞の）書きかた\nStandard (Newspaper) Orthography','ふつうの読みかた（カタカナ）\nStandard Reading (Katakana)','品詞\nPart of Speech']]
            df = df.rename(columns = {'ふつうの（新聞の）書きかた\nStandard (Newspaper) Orthography':'word',
                                      'ふつうの読みかた（カタカナ）\nStandard Reading (Katakana)': 'yomi',
                                      '品詞\nPart of Speech':'POS'}, inplace = False)
            self.vocab = ['<EOS>','<SOS>','<UNK>','<PAD>','<MASK>']
            self.vocab = self.vocab + list(df['word'])
        else: # reload = False (default)
            with gzip.open(self.data_fname, 'rb') as fgz:
                self.vocab = json.loads(fgz.read().decode('utf-8'))
            
    def __len__(self):
        return len(self.vocab)

    
    def save_data(self, out_fname=None)->None:
        if out_fname == None:
            out_fname = self.data_fname
            with gzip.open(out_fname, 'wt', encoding='UTF-8') as zipfile:
                json.dump(self.vocab, zipfile)    


    def token2id(self, word:str)->int:
        return self.vocab.index(word) if word in self.vocab else self.vocab.index('<UNK>')

        
    def id2token(self, id:int)->int:
        return self.vocab[id] if id < len(self.vocab) else -1

    
    def convert_tokens2ids(self, s:list, add_eos:bool=True)->list:
        ret = [self.token2id(word) for word in s]
        if add_eos:
            ret.append(self.vocab.index('<EOS>'))
        return ret
    

    def convert_ids2tokens(self, ids:list)->list:
        return [self.id2token(x) for x in ids]
    

    def tokenize(self, inputs:list, pad=True, max_length=-1)->dict:
        _max_length = max_length if max_length != -1 else self.max_length
        inputs = jaconv.normalize(inputs).strip()
        ret = {}
        tokens = self.wakati(inputs).strip().split(' ')
        ret['tokens'] = tokens
        ret['input_ids'] = self.convert_tokens2ids(ret['tokens'])
        ret['input_tokens'] = self.convert_ids2tokens(ret['input_ids'])
        if pad:
            for i in range(_max_length - len(ret['tokens'])):
                ret['input_ids'].insert(0,self.vocab.index('<PAD>'))

        return ret


if __name__ == "__main__":
    vdj = VDJ2500()
    print(vdj.vocab)
