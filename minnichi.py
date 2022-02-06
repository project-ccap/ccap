import sys
import typing
import os
import numpy as np
#import unicodedata
import glob
import json
import gzip
import matplotlib.pyplot as plt

import MeCab
import jaconv

import platform
#if isColab:
#    !pip install japanize_matplotlib > /dev/null 2>&1 
import japanize_matplotlib

class Minnichi(object):
    ''' 「みんなの日本語」データから語彙辞書を作成して，上で作成した均衡コーパスのフィルタとして使用する'''
    def __init__(self, 
                 wakati=None, 
                 yomi=None,
                 splitter=None,
                 max_length=-1,
                 data_fname='2022_0205minnichi_data.json.gz',
                 reload:bool=False,
                 isColab=False,
                 )->None:
        
        mecab_dic_dirs = { # MeCab のインストール場所の相違. 浅川の個人設定の差分吸収のため
            # 'Sinope':' /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/',
            # 'Pasiphae': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
            # 'Leda': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
            'Sinope':' /opt/homebrew/lib/mecab/dic/ipadic',
            'Pasiphae': '/usr/local/lib/mecab/dic/ipadic',
            'Leda': '/usr/local/lib/mecab/dic/ipadic',
            'colab': '/usr/share/mecab/dic/ipadic'
        }

        self.data_fname = data_fname

        isColab = platform.system() == 'Linux'
        hostname = 'colab' if isColab else os.uname().nodename.split('.')[0] 
        mecab_dic_dir = mecab_dic_dirs[hostname]
        if wakati == None:
            self.wakati = MeCab.Tagger(f'-Owakati').parse
            #self.wakati = MeCab.Tagger(f'-Owakati -d {mecab_dic_dir}').parse
        else:
            self.wakati = wakati
            
        if yomi == None:
            self.yomi = MeCab.Tagger(f'-Oyomi').parse
            #self.yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dic_dir}').parse
        else:
            self.yomi = yomi
            
        if splitter == None:
            from konoha import SentenceTokenizer
            self.splitter = SentenceTokenizer()
        else:
            self.splitter = splitter

        if reload:
            # 岩下先生から頂いた「みんなの日本語」データの読み込み
            minnichi_dir = '/Users/asakawa/study/2021jlpt'
            minnichi_files = sorted(glob.glob(os.path.join(minnichi_dir, 'MINNICHI_*.txt')))

            minnichi_text = [] # みんなの日本語テキストを読み込み
            for fname in minnichi_files:
                _fname = os.path.split(fname)[-1].split('.')[0]

                with open(fname,'r') as f:
                    texts = f.readlines()
        
                for txt in texts:
                    txt = jaconv.normalize(txt).strip()
                    if len(txt) > 0:
                        for l in self.splitter.tokenize(txt):
                            minnichi_text.append(l)

            vocab, lines, freq = ['<EOS>','<SOS>','<UNK>','<PAD>','<MASK>'], {}, {}
            _max_length = 0
            for i, l in enumerate(minnichi_text):
                lines[i] = {}
                tokens = self.wakati(l).strip().split(' ')
                phon = self.yomi(self.wakati(l).strip()).strip().split(' ')
                #lines[i]['orig'] = l # いらんかなー
                lines[i]['tokens'] = tokens
                lines[i]['n_token'] = len(lines[i]['tokens'])
                if _max_length < lines[i]['n_token']:
                    _max_length = lines[i]['n_token']
                lines[i]['yomi'] = phon
                lines[i]['n_yomi'] = len(lines[i]['yomi'])
                for token in tokens:
                    if not token in vocab:
                        vocab.append(token)
                        freq[token] = 1
                    else:
                        freq[token] += 1
                        
            self.freq = freq
            if max_length == -1:
                self.max_length = _max_length
            else:
                self.max_length = max_length
                
            self.vocab = vocab
            self.lines = lines
            for line in lines:
                lines[line]['input_ids'] = self.convert_tokens2ids(lines[line]['tokens'])
        else: # reload==False
            self.lines = {}
            with gzip.open(self.data_fname, 'rb') as fgz:
                tmp = json.loads(fgz.read().decode('utf-8'))
                for k in tmp.keys():
                    self.lines[int(k)] = tmp[k]
                    
            _max_length = 0
            vocab, freq = ['<EOS>','<SOS>','<UNK>','<PAD>','<MASK>'], {}
            for i in range(len(self.lines)):
                if _max_length < self.lines[i]['n_token']:
                    _max_length = self.lines[i]['n_token']

                tokens = self.lines[i]['tokens']
                for token in tokens:
                    if not token in vocab:
                        vocab.append(token)
                        freq[token] = 1
                    else:
                        freq[token] += 1
            self.freq = freq
            if max_length == -1:
                self.max_length = _max_length
            else:
                self.max_length = max_length
            self.vocab = vocab
        return

    
    def save_data(self, out_fname=None)->None:
        if out_fname == None:
            out_fname = self.data_fname
        with gzip.open(out_fname, 'wt', encoding='UTF-8') as zipfile:
            json.dump(self.lines, zipfile)    


    def __len__(self)->int:
        return len(self.lines)


    def __call__(self, x:int)->dict:
        return self.lines[x]

        
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


    def draw_freq(self, 
                  save_fname=None, 
                  figsize=(18,8),
                  fontsize=8,
                  rotation=35,
                  #n_best = 100,
                 )->None:
        plt.figure(figsize=figsize)
        plt.plot(np.array(sorted(self.freq.values())[::-1])) #[:n_best])
        plt.title('頻度', fontsize=fontsize)
        #plt.xticks(np.arange(len(self.freq)), [self.vocab[self.vocab.index(k)] for k in np.array(sorted(self.freq.values())[::-1])[:n_best]], fontsize=fontsize, rotation=rotation)
        plt.xticks(np.arange(len(self.freq)), [self.vocab[self.vocab.index(k)] for k, _ in self.freq.items()], fontsize=fontsize, rotation=rotation)
        plt.xlabel('頻度順に並べ替え', fontsize=fontsize)
        plt.ylabel('頻度', fontsize=fontsize)
        if save_fname != None:
            plt.savefig(save_fname)
            
