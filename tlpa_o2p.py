import os
import MeCab
import pandas as pd
import jaconv
import numpy as np
from tqdm.notebook import tqdm

from ccap import ccap_w2v

isColab = 'google.colab' in str(get_ipython())
w2v = ccap_w2v(is2017=False, isColab=isColab).w2v

from ccap.mecab_settings import yomi

class TLPA():
    '''TLPA の語彙を元に考えてみよう!'''    
    def __init__(self, 
                 w2v:w2v=w2v, 
                 yomi:MeCab.Tagger=yomi,
                 reload=True,
                 traindata_size = 10000,
                ):

        super().__init__()
        self.w2v = w2v
        self.yomi = yomi
        self.traindata_size = traindata_size

        if reload:
            self.tlpa_vocab = self.get_tlpa_vocab()
            #print(f'word2vec のインデックスのうち最大値 max_idx:{max_idx} {w2v.index_to_key[max_idx]}')
            self.ntt_freq = self.get_ntt_freq()
            self.training_vocab = self.get_training_vocab()
            #self.training_vocab = vocab
            self.vocab = self.tlpa_vocab + self.training_vocab

            ortho_vocab, ortho_freq = ['<EOW>','<SOW>','<UNK>', '<PAD>'], {}
            phone_vocab, phone_freq = ['<EOW>','<SOW>','<UNK>', '<PAD>'], {}
            phone_vocab= ['<EOW>', '<SOW>', '<UNK>', '<PAD>', \
                          'n', 'o', 'h', 'a', 'i', 't', 'g', 'r', 'u', 'd', 'e', 'sh', 'q', 'm', 'k', 's', 'y', 'p', 'N', 'b', 'ts', 'o:',\
                          'ky', 'f', 'w', 'ch', 'ry', 'gy', 'u:', 'z', 'j', 'py', 'hy', 'i:', 'e:', 'a:', 'by', 'ny', 'my', 'dy', \
                          'a::', 'u::', 'o::']

            training_data, excluded_data = {}, []
            max_ortho_length, max_phone_length = 0, 0
            for orth in tqdm(self.training_vocab):
                i = len(training_data)

                if orth in self.ntt_orth2hira:
                    _yomi = jaconv.kata2hira(self.ntt_orth2hira[orth])
                else:
                    _yomi = jaconv.kata2hira(yomi(orth).strip())
                _phon = self.hira2julius(_yomi).split(' ')
                _orth = [c for c in orth]

                if False in [True if p in phone_vocab else False for p in _phon]:
                    excluded_data.append(orth)
                    continue
                phone_ids = [phone_vocab.index(p) for p in _phon]

                for o in _orth:
                    if not o in ortho_vocab:
                        ortho_vocab.append(o)
                ortho_ids = [ortho_vocab.index(o) for o in _orth]

                training_data[i] = {'orig': orth,
                                    'ortho':_orth,
                                    'phone':_phon,
                                    'ortho_ids': ortho_ids,
                                    'phone_ids': phone_ids,
                                    'semem':w2v[orth],
                                   }
                #orth2idx[orth] = training_data[i]
                len_orth, len_phon = len(_orth), len(_phon)
                max_ortho_length = len_orth if len_orth > max_ortho_length else max_ortho_length
                max_phone_length = len_phon if len_phon > max_phone_length else max_phone_length
                if len(training_data) >= self.traindata_size:
                    break

            self.training_data = training_data

            tlpa_data = {}
            for orth in self.tlpa_vocab:
                i = len(tlpa_data)
                if orth in self.ntt_orth2hira:
                    _yomi = jaconv.kata2hira(self.ntt_orth2hira[orth])
                else:
                    _yomi = jaconv.kata2hira(yomi(orth).strip())
                _phon = self.hira2julius(_yomi).split(' ')
                _orth = [c for c in orth]

                for p in _phon:
                    if not p in phone_vocab:
                        phone_vocab.append(p)
                phone_ids = [phone_vocab.index(p) for p in _phon]

                for o in _orth:
                    if not o in ortho_vocab:
                        ortho_vocab.append(o)
                ortho_ids = [ortho_vocab.index(o) for o in _orth]

                tlpa_data[i] = {'orig': orth,
                                'ortho':_orth,
                                'phone':_phon,
                                'ortho_ids': ortho_ids,
                                'phone_ids': phone_ids,
                                'semem':w2v[orth],
                               }
                len_orth, len_phon = len(_orth), len(_phon)
                max_ortho_length = len_orth if len_orth > max_ortho_length else max_ortho_length
                max_phone_length = len_phon if len_phon > max_phone_length else max_phone_length
                #orth2idx[orth] = tlpa_data[i]

            #self.orth2idx = orth2idx
            self.ortho_vocab, self.phone_vocab = ortho_vocab, phone_vocab
            self.max_ortho_length = max_ortho_length
            self.max_phone_length = max_phone_length

            self.tlpa_data = tlpa_data


    def hira2julius(self, text:str)->str:
        text = text.replace('ゔぁ', ' b a')
        text = text.replace('ゔぃ', ' b i')
        text = text.replace('ゔぇ', ' b e')
        text = text.replace('ゔぉ', ' b o')
        text = text.replace('ゔゅ', ' by u')

        #text = text.replace('ぅ゛', ' b u')
        text = jaconv.hiragana2julius(text)
        return text

    def __len__(self)->int:
        return len(self.data)

    def __call__(self, x:int)->dict:
        return self.data[x]

    def __getitem__(self, x:int)->dict:
        return self.data[x]

    def get_tlpa_vocab(self):

        print('# 上間先生からいただいた TLPA 文字データをエクセルファイルから読み込む')
        ccap_base = 'ccap'
        #ccap_base = '/Users/asakawa/study/2020ccap'
        uema_excel = '呼称(TLPA)語彙属性.xlsx'
        k = pd.read_excel(os.path.join(ccap_base, uema_excel), header=[1], sheet_name='刺激属性')
        tlpa_words = list(k['標的語'])

        #標的語には文字 '／' が混じっているので，切り分けて別の単語と考える
        _tlpa_words = []
        for word in tlpa_words:
            for x in word.split('／'):
                _tlpa_words.append(x)

        # word2vec に存在するか否かのチェックと，存在するのであれば word2vec におけるその単語のインデックスを得る
        # word2vec では，単語は頻度順に並んでいるので最大値インデックスを取る単語が，最低頻度語ということになる
        # そのため，TLPA の刺激語のうち，最低頻度以上の全単語を使って学習を行うことを考えるためである。
        max_idx, __tlpa_words = 0, []
        for w in _tlpa_words:
            if w in w2v:
                idx = w2v.key_to_index[w]
                max_idx = idx if idx > max_idx else max_idx
                __tlpa_words.append(w)
            # else:
            #     print(f'{w} word2vec に存在せず')

        #self.tlpa_vocab = __tlpa_words.copy()
        return __tlpa_words


    def get_training_vocab(self):

        #JISX2008-1990 コードから記号とみなしうるコードを集めて ja_symbols とする
        #記号だけから構成されている word2vec の項目は排除するため
        self.ja_symbols = '、。，．・：；？！゛゜´\' #+ \'｀¨＾‾＿ヽヾゝゞ〃仝々〆〇ー—‐／＼〜‖｜' + \
        '…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋−±×÷＝≠＜＞≦≧∞∴♂♀°′″℃¥＄¢£％＃＆＊＠§' + \
        '☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨¬⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪†‡¶◯#'+ \
        'ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        self.ja_symbols_normalized = jaconv.normalize(self.ja_symbols)

        print(f'# 訓練に用いる単語の選定 {self.traindata_size} 語')
        vocab = []; i=0
        #while (len(vocab) < self.traindata_size) and (i<len(self.ntt_freq)):
        while i<len(self.ntt_freq):
            word = self.ntt_freq[i]
            if word == '\u3000': # NTT 日本語の語彙特性で，これだけ変なので特別扱い
                i += 1
                continue
            word = jaconv.normalize(word).replace('・','').replace('ヴ','ブ')
            if (not word in self.ja_symbols) and (not word.isascii()) and (word in self.w2v):
                if not word in self.tlpa_vocab:
                    vocab.append(word)
            i+=1
        return vocab


    def get_ntt_freq(self):
        print('#NTT日本語語彙特性より頻度情報を取得')

        #データファイルの保存してあるディレクトリの指定
        ntt_dir = 'ccap'
        psy71_fname = 'psylex71utf8.txt'  # ファイル名
        #print('# 頻度情報の取得')
        #print('# psylex71.txt を utf8 に変換したファイルを用いる')
        with open(os.path.join(ntt_dir,psy71_fname), 'r') as f:
            ntt71raw = f.readlines()

        tmp = [line.split(' ')[:6] for line in ntt71raw]
        tmp2 = [[int(line[0]),line[2],line[4],int(line[5]), line[3]] for line in tmp]
        #単語ID(0), 単語，品詞，頻度 だけ取り出す
        ntt_freq = {x[0]-1:{'単語':x[1],'品詞':x[2],'頻度':x[3], 'よみ':x[4]} for x in tmp2}
        self.ntt_orth2hira = {jaconv.normalize(ntt_freq[x]['単語']):
                              jaconv.normalize(ntt_freq[x]['よみ']) for x in ntt_freq}
        #print(f'#登録総単語数: {len(ntt_freq)}')

        Freq = np.zeros((len(ntt_freq)),dtype=np.uint)  #ソートに使用する numpy 配列
        for i, x in enumerate(ntt_freq):
            Freq[i] = ntt_freq[i]['頻度']

        Freq_sorted = np.argsort(Freq)[::-1]  #頻度降順に並べ替え
        # Freq_val_sorted = np.sort(Freq)[::-1] #並べ方対応単語
        # print('#動作確認 頻度降順に並んでいるか否かの確認？')
        # for no in Freq_sorted[:15]:
        #     xx = ntt_freq[no]
        #     word = xx['単語']
        #     freq = xx['頻度']
        #     pos = xx['品詞']
        #     print(f'no:{no:<7d} {word:<6s} {freq:<8d} {pos:<5s}')

        # self.ntt_freq には頻度順に単語が並んでいる。
        #self.ntt_freq = [ntt_freq[x]['単語'] for x in Freq_sorted]
        return [ntt_freq[x]['単語']for x in Freq_sorted]


if __name__ == '__main__':
    #tlpa = TLPA(traindata_size=20000)
    tlpa = TLPA(traindata_size=traindata_size) 

    print(colored(f'訓練データサイズ:{len(tlpa.training_data)}','blue',attrs=['bold']))
    print(colored(f'検証データサイズ:{len(tlpa.tlpa_data)} TLPA 単語数','blue',attrs=['bold']))
    print(colored(f'音素数:{len(tlpa.phone_vocab)}','blue',attrs=['bold']))
    print(colored(f'書記素数:{len(tlpa.ortho_vocab)}','blue',attrs=['bold']))
    print(colored(f'最長書記素(文字)数:{tlpa.max_ortho_length}, 最長音素数:{tlpa.max_phone_length}', 'blue',attrs=['bold']))
