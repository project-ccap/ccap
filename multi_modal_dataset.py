"""
- filename: multi_modal_dataset.py
- author: 浅川伸一
- date: 2022_0311
"""
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

import IPython
# isColab = 'google.colab' in str(IPython.get_ipython())
# if isColab:
#     !pip install jaconv
import jaconv

#import MeCab
#isColab = 'google.colab' in str(get_ipython())
#from ccap import ccap_w2v
#w2v = ccap_w2v(is2017=False, isColab=isColab).w2v
#from ccap.mecab_settings import yomi

import re

class Vocab_ja():
    '''
    訓練データとしては，NTT 日本語語彙特性 (天野，近藤, 1999, 三省堂) の頻度データ，実際のファイル名としては `pslex71.txt` から頻度データを読み込んで，高頻度語を訓練データとする。
    ただし，検証データに含まれる単語は訓練データとして用いない。

    検証データとして，以下のいずれかを考える
    1. TLPA (藤田 他, 2000, 「失語症語彙検査」の開発，音声言語医学 42, 179-202) 
    2. SALA 上智大学失語症語彙検査 

    このオブジェクトクラスでは，
    `phone_vocab`, `ortho_vocab`, `ntt_freq`, に加えて，単語の読みについて ntt_orth2hira によって読みを得ることにした。

    * `train_data`, `test_data` という辞書が本体である。
    各辞書の項目には，さらに
    `Vocab_ja.test_data[0].keys() = dict_keys(['orig', 'ortho', 'phone', 'ortho_ids', 'phone_ids', 'semem'])`

    各モダリティ共通トークンとして以下を設定した
    * <PAD>: 埋め草トークン
    * <EQW>: 単語終端トークン
    * <SOW>: 単語始端トークン
    * <UNK>: 未定義トークン

    このクラスで定義されるデータは 2 つの辞書である。すなわち 1. train_data, 2. tlpa_data である。
    各辞書は，次のような辞書項目を持つ。
    ```
    {0: {'orig': 'バス',
    'yomi': 'ばす',
    'ortho': ['バ', 'ス'],
    'ortho_ids': [695, 514],
    'ortho_r': ['ス', 'バ'],
    'ortho_ids_r': ['ス', 'バ'],
    'phone': ['b', 'a', 's', 'u'],
    'phone_ids': [23, 7, 19, 12],
    'phone_r': ['u', 's', 'a', 'b'],
    'phone_ids_r': [12, 19, 7, 23],
    'mora': ['ば', 'す'],
    'mora_r': ['す', 'ば'],
    'mora_ids': [87, 47],
    'mora_p': ['b', 'a', 's', 'u'],
    'mora_p_r': ['s', 'u', 'b', 'a'],
    'mora_p_ids': [6, 5, 31, 35],
    'mora_p_ids_r': [31, 35, 6, 5]},
    ```
    '''
    
    def __init__(self, 
                 traindata_size = 10000,
                 w2v=None,
                 yomi=None,
                 test_name='TLPA',  # or 'SALA'
                ):

        #isColab = 'google.colab' in str(IPython.get_ipython())

        if w2v != None:
            self.w2v = w2v
        else:
            from ccap import ccap_w2v
            #from ccap_w2v import ccap_w2v
            self.w2v = ccap_w2v(is2017=False).w2v

        if yomi != None:
            self.yomi = yomi
        else:
            #from mecab_settings import yomi
            from ccap.mecab_settings import yomi
            self.yomi = yomi

        # if wakati != None:
        #     self.wakati = wakati
        # else:
        #     from ccap.mecab_settings import wakati
        #     self.wakati = wakati

        # 訓練語彙数の上限 `training_size` を設定 
        self.traindata_size = traindata_size

        # `self.moraWakachi()` で用いる正規表現のあつまり 各条件を正規表現で表す
        self.c1 = '[うくすつぬふむゆるぐずづぶぷゔ][ぁぃぇぉ]' #ウ段＋「ァ/ィ/ェ/ォ」
        self.c2 = '[いきしちにひみりぎじぢびぴ][ゃゅぇょ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        self.c3 = '[てで][ぃゅ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        self.c4 = '[ぁ-ゔー]' #カタカナ１文字（長音含む）
        self.c5 = '[ふ][ゅ]'
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）
        #cond = '('+c1+'|'+c2+'|'+c3+'|'+c4+')'
        self.cond = '('+self.c5+'|'+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        # 以上 `self.moraWakachi()` で用いる正規表現の定義


        self.ortho_vocab, self.ortho_freq = ['<PAD>', '<EOW>','<SOW>','<UNK>'], {}
        self.phone_vocab, self.phone_freq = ['<PAD>', '<EOW>','<SOW>','<UNK>'], {}
        self.phone_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>', \
        'n', 'o', 'h', 'a', 'i', 't', 'g', 'r', 'u', 'd', 'e', 'sh', 'q', 'm', 'k', 's', 'y', 'p', 'N', 'b', 'ts', 'o:',\
        'ky', 'f', 'w', 'ch', 'ry', 'gy', 'u:', 'z', 'j', 'py', 'hy', 'i:', 'e:', 'a:', 'by', 'ny', 'my', 'dy', \
        'a::', 'u::', 'o::']

        # 全モーラリストを `mora_vocab` として登録
        self.mora_vocab=['<PAD>', '<EOW>', '<SOW>', '<UNK>', \
        'ぁ', 'あ', 'ぃ', 'い', 'ぅ', 'う', 'うぃ', 'うぇ', 'うぉ', 'ぇ', 'え', 'お', \
        'か', 'が', 'き', 'きゃ', 'きゅ', 'きょ', 'ぎ', 'ぎゃ', 'ぎゅ', 'ぎょ', 'く', 'くぁ', 'くぉ', 'ぐ', 'ぐぁ', 'け', 'げ', 'こ', 'ご', \
        'さ', 'ざ', 'し', 'しぇ', 'しゃ', 'しゅ', 'しょ', 'じ', 'じぇ', 'じゃ', 'じゅ', 'じょ', 'す', 'ず', 'せ', 'ぜ', 'そ', 'ぞ', \
        'た', 'だ', 'ち', 'ちぇ', 'ちゃ', 'ちゅ', 'ちょ', 'ぢ', 'ぢゃ', 'ぢょ', 'っ', 'つ', 'つぁ', 'つぃ', 'つぇ', 'つぉ', 'づ', 'て', 'てぃ', 'で', 'でぃ', 'でゅ', 'と', 'ど', \
        'な', 'に', 'にぇ', 'にゃ', 'にゅ', 'にょ', 'ぬ', 'ね', 'の', \
        'は', 'ば', 'ぱ', 'ひ', 'ひゃ', 'ひゅ', 'ひょ', 'び', 'びゃ', 'びゅ', 'びょ', 'ぴ', 'ぴゃ', 'ぴゅ', 'ぴょ', 'ふ', 'ふぁ', 'ふぃ', 'ふぇ', 'ふぉ', 'ふゅ', 'ぶ', 'ぷ', 'へ', 'べ', 'ぺ', 'ほ', 'ぼ', 'ぽ', \
        'ま', 'み', 'みゃ', 'みゅ', 'みょ', 'む', 'め', 'も', \
        'や', 'ゆ', 'よ', 'ら', 'り', 'りゃ', 'りゅ', 'りょ', 'る', 'れ', 'ろ', 'ゎ', 'わ', 'ゐ', 'ゑ', 'を', 'ん', 'ー']

        # モーラから音への変換表を表す辞書を `mora2jul` として登録
        self.mora2jul={'ぁ': ['a'], 'あ': ['a'], 'ぃ': ['i'], 'い': ['i'], 'ぅ': ['u'], 'う': ['u'], 'うぃ': ['w', 'i'], 'うぇ': ['w', 'e'], 'うぉ': ['w', 'o'], \
        'ぇ': ['e'], 'え': ['e'], 'お': ['o'], 'か': ['k', 'a'], 'が': ['g', 'a'], 'き': ['k', 'i'], 'きゃ': ['ky', 'a'], 'きゅ': ['ky', 'u'], 'きょ': ['ky', 'o'], \
        'ぎ': ['g', 'i'], 'ぎゃ': ['gy', 'a'], 'ぎゅ': ['gy', 'u'], 'ぎょ': ['gy', 'o'], 'く': ['k', 'u'], 'くぁ': ['k', 'u', 'a'], 'くぉ': ['k', 'u', 'o'], 'ぐ': ['g', 'u'], \
        'ぐぁ': ['g', 'u', 'a'], 'け': ['k', 'e'], 'げ': ['g', 'e'], 'こ': ['k', 'o'], 'ご': ['g', 'o'], 'さ': ['s', 'a'], 'ざ': ['z', 'a'], 'し': ['sh', 'i'], \
        'しぇ': ['sh', 'e'], 'しゃ': ['sh', 'a'], 'しゅ': ['sh', 'u'], 'しょ': ['sh', 'o'], 'じ': ['j', 'i'], 'じぇ': ['j', 'e'], 'じゃ': ['j', 'a'], 'じゅ': ['j', 'u'], \
        'じょ': ['j', 'o'], 'す': ['s', 'u'], 'ず': ['z', 'u'], 'せ': ['s', 'e'], 'ぜ': ['z', 'e'], 'そ': ['s', 'o'], 'ぞ': ['z', 'o'], 'た': ['t', 'a'], 'だ': ['d', 'a'], \
        'ち': ['ch', 'i'], 'ちぇ': ['ch', 'e'], 'ちゃ': ['ch', 'a'], 'ちゅ': ['ch', 'u'], 'ちょ': ['ch', 'o'], 'ぢ': ['j', 'i'], 'ぢゃ': ['j', 'a'], 'ぢょ': ['j', 'o'], \
        'っ': ['q'], 'つ': ['ts', 'u'], 'つぁ': ['ts', 'a'], 'つぃ': ['ts', 'i'], 'つぇ': ['ts', 'e'], 'つぉ': ['ts', 'o'], 'づ': ['z', 'u'], 'て': ['t', 'e'], 'てぃ': ['t', 'i'], \
        'で': ['d', 'e'], 'でぃ': ['d', 'i'], 'でゅ': ['dy', 'u'], 'と': ['t', 'o'], 'ど': ['d', 'o'], 'な': ['n', 'a'], 'に': ['n', 'i'], 'にぇ': ['n', 'i', 'e'], \
        'にゃ': ['ny', 'a'], 'にゅ': ['ny', 'u'], 'にょ': ['ny', 'o'], 'ぬ': ['n', 'u'], 'ね': ['n', 'e'], 'の': ['n', 'o'], 'は': ['h', 'a'], 'ば': ['b', 'a'], \
        'ぱ': ['p', 'a'], 'ひ': ['h', 'i'], 'ひゃ': ['hy', 'a'], 'ひゅ': ['hy', 'u'], 'ひょ': ['hy', 'o'], 'び': ['b', 'i'], 'びゃ': ['by', 'a'], 'びゅ': ['by', 'u'], \
        'びょ': ['by', 'o'], 'ぴ': ['p', 'i'], 'ぴゃ': ['py', 'a'], 'ぴゅ': ['py', 'u'], 'ぴょ': ['py', 'o'], 'ふ': ['f', 'u'], 'ふぁ': ['f', 'a'], 'ふぃ': ['f', 'i'], \
        'ふぇ': ['f', 'e'], 'ふぉ': ['f', 'o'], 'ふゅ': ['hy', 'u'], 'ぶ': ['b', 'u'], 'ぷ': ['p', 'u'], 'へ': ['h', 'e'], 'べ': ['b', 'e'], 'ぺ': ['p', 'e'], 'ほ': ['h', 'o'], \
        'ぼ': ['b', 'o'], 'ぽ': ['p', 'o'], 'ま': ['m', 'a'], 'み': ['m', 'i'], 'みゃ': ['my', 'a'], 'みゅ': ['my', 'u'], 'みょ': ['my', 'o'], 'む': ['m', 'u'], 'め': ['m', 'e'], \
        'も': ['m', 'o'], 'や': ['y', 'a'], 'ゆ': ['y', 'u'], 'よ': ['y', 'o'], 'ら': ['r', 'a'], 'り': ['r', 'i'], 'りゃ': ['ry', 'a'], 'りゅ': ['ry', 'u'], 'りょ': ['ry', 'o'], \
        'る': ['r', 'u'], 'れ': ['r', 'e'], 'ろ': ['r', 'o'], 'ゎ': ['w', 'a'], 'わ': ['w', 'a'], 'ゐ': ['i'], 'ゑ': ['e'], 'を': ['o'], 'ん': ['N'], 'ー':[':']}

        # モーラに用いる音を表すリストを `mora_p_vocab` として登録
        self.mora_p_vocab = ['<PAD>', '<EOW>', '<SOW>', '<UNK>',  \
        'N', 'a', 'b', 'by', 'ch', 'd', 'dy', 'e', 'f', 'g', 'gy', 'h', 'hy', 'i', 'j', 'k', 'ky', \
        'm', 'my', 'n', 'ny', 'o', 'p', 'py', 'q', 'r', 'ry', 's', 'sh', 't', 'ts', 'u', 'w', 'y', 'z']

        # 母音を表す音から ひらがな への変換表を表す辞書を `vow2hira` として登録
        self.vow2hira = {'a':'あ', 'i':'い', 'u':'う', 'e':'え', 'o':'お', 'N':'ん'}

        self.mora_freq = {'<PAD>':0, '<EOW>':0, '<SOW>':0, '<UNK>':0}
        self.mora_p = {}

        # NTT 日本語語彙特性データから，`self.train_data` を作成
        self.ntt_freq, self.ntt_orth2hira = self.make_ntt_freq_data()

        if test_name == 'TLPA':                      # TLPA データを読み込み
            self.test_name = test_name
            self.test_vocab = self.get_tlpa_vocab()  # tlpa データを読み込み
        elif test_name == 'SALA':                    # sala データの読み込み
            self.test_name = test_name
            self.test_vocab = self.get_sala_vocab() # sala データを読み込む
        else:
            self.test_name = None
            print(f'Invalid test_name:{test_name}')
            sys.exit()

        self.ntt_freq_vocab = self.set_train_vocab_minus_test_vocab()
        self.train_data, self.excluded_data = {}, []
        max_ortho_length, max_phone_length, max_mora_length, max_mora_p_length = 0, 0, 0, 0

        for orth in tqdm(self.ntt_freq_vocab):
            i = len(self.train_data)

            # 書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
            # 書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る
            _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = self.get7lists_from_orth(orth)

            # 音韻語彙リスト `self.phone_vocab` に音韻が存在していれば True そうでなければ False というリストを作成し，
            # そのリスト無いに False があれば，排除リスト `self.excluded_data` に登録する
            if False in [True if p in self.phone_vocab else False for p in _phon]:
                self.excluded_data.append(orth)
                continue

            phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = self.get6ids(_phon, _orth, _yomi)
            _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = self.yomi2mora_transform(_yomi)
            self.train_data[i] = {'orig': orth, 'yomi': _yomi,
                                 'ortho':_orth, 'ortho_ids': orth_ids, 'ortho_r': _orth_r, 'ortho_ids_r': orth_ids_r,
                                 'phone':_phon, 'phone_ids': phon_ids, 'phone_r': _phon_r, 'phone_ids_r': phon_ids_r,
                                 'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p,
                                 'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                                 #'semem':self.w2v[orth],
                               }
            len_orth, len_phon, len_mora, len_mora_p = len(_orth), len(_phon), len(_mora), len(_mora_p)
            max_ortho_length = len_orth if len_orth > max_ortho_length else max_ortho_length
            max_phone_length = len_phon if len_phon > max_phone_length else max_phone_length
            max_mora_length = len_mora if len_mora > max_mora_length else max_mora_length
            max_mora_p_length = len_mora_p if len_mora_p > max_mora_p_length else max_mora_p_length
            if len(self.train_data) >= self.traindata_size: # 上限値に達したら終了する
                self.train_vocab = [self.train_data[x]['orig'] for x in self.train_data]
                break 

        # 検証データ `self.test_data` の作成
        self.test_data = {}
        #self.tlpa_data = {}
        for orth in self.test_vocab:
            i = len(self.test_data)

            _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r = self.get7lists_from_orth(orth)
            phon_ids, phon_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r = self.get6ids(_phon, _orth, _yomi)
            _yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls = self.yomi2mora_transform(_yomi)
            self.test_data[i] = {'orig': orth, 'yomi': _yomi,
                                 'ortho':_orth, 'ortho_ids': orth_ids, 'ortho_r': _orth_r, 'ortho_ids_r': orth_ids_r,
                                 'phone':_phon, 'phone_ids': phon_ids, 'phone_r': _phon_r, 'phone_ids_r': phon_ids_r,
                                 'mora': _mora1, 'mora_r': _mora1_r, 'mora_ids': _mora_ids, 'mora_p': _mora_p,
                                 'mora_p_r': _mora_p_r, 'mora_p_ids': _mora_p_ids, 'mora_p_ids_r': _mora_p_ids_r,
                                 #'semem':self.w2v[orth],
                               }
            len_orth, len_phon, len_mora, len_mora_p = len(_orth), len(_phon), len(_mora), len(_mora_p)
            max_ortho_length = len_orth if len_orth > max_ortho_length else max_ortho_length
            max_phone_length = len_phon if len_phon > max_phone_length else max_phone_length
            max_mora_length = len_mora if len_mora > max_mora_length else max_mora_length
            max_mora_p_length = len_mora_p if len_mora_p > max_mora_p_length else max_mora_p_length

        self.max_ortho_length = max_ortho_length
        self.max_phone_length = max_phone_length
        self.max_mora_length = max_mora_length
        self.max_mora_p_length = max_mora_p_length

    def yomi2mora_transform(self, yomi):
        """ひらがな表記された引数 `yomi` から，日本語の 拍(モーラ)  関係のデータを作成する
        引数:
        yomi:str ひらがな表記された単語 UTF-8 で符号化されていることを仮定している

        戻り値:
        yomi:str 入力された引数
        _mora1:list[str] `_mora` に含まれる長音 `ー` を直前の母音で置き換えた，モーラ単位の分かち書きされた文字列のリスト
        _mora1_r:list[str] `_mora1` を反転させた文字列リスト
        _mora:list[str] `self.moraWakatchi()` によってモーラ単位で分かち書きされた文字列のリスト
        _mora_ids:list[int] `_mora` を対応するモーラ ID で置き換えた整数値からなるリスト
        _mora_p:list[str] `_mora` を silius によって音に変換した文字列リスト
        _mora_p_r:list[str] `_mora_p` の反転リスト
        _mora_p_ids:list[int] `mora_p` の各要素を対応する 音 ID に変換した数値からなるリスト
        _mora_p_ids_r:list[int] `mora_p_ids` の各音を反転させた数値からなるリスト
        _juls:list[str]: `yomi` を julius 変換した音素からなるリスト
        """
        _mora = self.moraWakachi(yomi) # 一旦モーラ単位の分かち書きを実行して `_mora` に格納
    
        # 単語をモーラ反転した場合に長音「ー」の音が問題となるので，長音「ー」を母音で置き換えるためのプレースホルダとして. `_mora` を用いる
        _mora1 = _mora.copy()     

        # その他のプレースホルダの初期化，モーラ，モーラ毎 ID, モーラ音素，モーラの音素の ID， モーラ音素の反転，モーラ音素の反転 ID リスト
        _mora_ids, _mora_p, _mora_p_ids, _mora_p_r, _mora_p_ids_r = [], [], [], [], []
        _m0 = 'ー' # 長音記号
    
        for i, _m in enumerate(_mora): # 各モーラ単位の処理と登録
        
            __m = _m0 if _m == 'ー' else _m               # 長音だったら，前音の母音を __m とし，それ以外は自分自身を __m に代入
            _mora1[i] = __m                               # 長音を変換した結果を格納
            _mora_ids.append(self.mora_vocab.index(__m))  # モーラを ID 番号に変換
            _mora_p += self.mora2jul[__m]                 # モーラを音素に変換して `_mora_p` に格納
        
            # 変換した音素を音素 ID に変換して，`_mora_p_ids` に格納
            _mora_p_ids += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[__m]] 
        
            if not _m in self.mora_freq: # モーラの頻度表を集計
                self.mora_freq[__m] = 1
            else:
                self.mora_freq[__m] +=1
            
            if self.hira2julius(__m)[-1] in self.vow2hira:           # 直前のモーラの最終音素が母音であれば
                _m0 = self.vow2hira[self.hira2julius(__m)[-1]]  # 直前の母音を代入しておく。この処理が 2022_0311 でのポイントであった

        # モーラ分かち書きした単語 _mora1 の反転を作成し `_mora1_r` に格納
        _mora1_r = [m for m in _mora1[::-1]]
    
        for _m in _mora1_r:                   # 反転した各モーラについて
            # モーラ単位で julius 変換して音素とし `_mora_p_r` に格納
            _mora_p_r += self.mora2jul[_m]

            # `_mora_p_r` に格納した音素を音素 ID に変換し `_mora_p_ids` に格納
            _mora_p_ids_r += [self.mora_p_vocab.index(_p) for _p in self.mora2jul[_m]]

        _juls = self.hira2julius(yomi)
        
        return yomi, _mora1, _mora1_r, _mora, _mora_ids, _mora_p, _mora_p_r, _mora_p_ids, _mora_p_ids_r, _juls


    def get6ids(self, _phon, _orth, yomi):

        # 音韻 ID リスト `phone_ids` に音素を登録する
        phone_ids = [self.phone_vocab.index(p) for p in _phon]

        # 直上の音韻 ID リストの逆転を作成
        phone_ids_r = [p_id for p_id in phone_ids[::-1]]

        # 書記素 ID リスト `ortho_ids` に書記素を登録
        for o in _orth:
            if not o in self.ortho_vocab:
                self.ortho_vocab.append(o)
        orth_ids = [self.ortho_vocab.index(o) for o in _orth]

        # 直上の書記素 ID リストの逆転を作成
        orth_ids_r = [o_id for o_id in orth_ids[::-1]]
        #orth_ids_r = [o_id for o_id in _orth[::-1]]

        mora_ids = []
        for _p in self.hira2julius(yomi):
            mora_ids.append(self.phone_vocab.index(_p))

        mora_ids_r = []
        #print(f'yomi:{yomi}, self.moraWakati(yomi):{self.moraWakachi(jaconv.hira2kata(yomi))}')
        # for _p in self.moraWakachi(jaconv.hira2kata(yomi))[::-1]:
        #     for __p in  self.phone_vocab.index(self.hira2julius(jaconv.kata2hira(_p))):
        #         mora_ids_r.append(__p)

        return phone_ids, phone_ids_r, orth_ids, orth_ids_r, mora_ids, mora_ids_r


    def moraWakachi(self, hira_text):
        """ ひらがなをモーラ単位で分かち書きする
        https://qiita.com/shimajiroxyz/items/a133d990df2bc3affc12"""

        return self.re_mora.findall(hira_text)


    def _kana_moraWakachi(kan_text):
        # self.c1 = '[ウクスツヌフムユルグズヅブプヴ][ァィェォ]' #ウ段＋「ァ/ィ/ェ/ォ」
        # self.c2 = '[イキシチニヒミリギジヂビピ][ャュェョ]' #イ段（「イ」を除く）＋「ャ/ュ/ェ/ョ」
        # self.c3 = '[テデ][ィュ]' #「テ/デ」＋「ャ/ィ/ュ/ョ」
        # self.c4 = '[ァ-ヴー]' #カタカナ１文字（長音含む）

        self.cond = '('+self.c1+'|'+self.c2+'|'+self.c3+'|'+self.c4+')'
        self.re_mora = re.compile(self.cond)
        return re_mora.findall(kana_text)



    def get7lists_from_orth(self, orth):
        """書記素 `orth` から 読みリスト，音韻表現リスト，音韻表現反転リスト，
        書記表現リスト，書記表現反転リスト，モーラ表現リスト，モーラ表現反転リスト の 7 つのリストを得る"""

        # 単語の表層形を，読みに変換して `_yomi` に格納
        # ntt_orth2hira という命名はおかしかったから修正 2022_0309
        if orth in self.ntt_orth2hira:
            _yomi = self.ntt_orth2hira[orth]
        else:
            _yomi = jaconv.kata2hira(self.yomi(orth).strip())

        # `_yomi` を julius 表記に変換して `_phon` に代入
        _phon = self.hira2julius(_yomi)# .split(' ')

        # 直上の `_phon` の逆転を作成して `_phone_r` に代入
        _phon_r = [_p_id for _p_id in _phon[::-1]]

        # 書記素をリストに変換
        _orth = [c for c in orth]

        # 直上の `_orth` の逆転を作成して `_orth_r` に代入
        _orth_r = [c for c in _orth[::-1]]

        _mora = self.moraWakachi(jaconv.hira2kata(_yomi))
        _mora_r = [_m for _m in _mora[::-1]]
        _mora = self.moraWakachi(_yomi)
        for _m in _mora:
            if not _m in self.mora_vocab:
                self.mora_vocab.append(_m)
            for _j in self.hira2julius(_m):
                if not _j in self.mora_p:
                    self.mora_p[_j] = 1
                else:
                    self.mora_p[_j] += 1

        return _yomi, _phon, _phon_r, _orth, _orth_r, _mora, _mora_r



    def hira2julius(self, text:str)->str:
        """`jaconv.hiragana2julius()` では未対応の表記を扱う"""
        text = text.replace('ゔぁ', ' b a')
        text = text.replace('ゔぃ', ' b i')
        text = text.replace('ゔぇ', ' b e')
        text = text.replace('ゔぉ', ' b o')
        text = text.replace('ゔゅ', ' by u')

        #text = text.replace('ぅ゛', ' b u')
        text = jaconv.hiragana2julius(text).split()
        return text

    def __len__(self)->int:
        return len(self.data)

    def __call__(self, x:int)->dict:
        return self.data[x]

    def __getitem__(self, x:int)->dict:
        return self.data[x]

    def get_tlpa_vocab(self):
        """TLPA データの読み込み"""

        print('# 上間先生からいただいた TLPA 文字データ `呼称(TLPA)語彙属性.xlsx` を読み込む')

        ccap_base = 'ccap'
        #ccap_base = '/Users/asakawa/study/2020ccap'
        uema_excel = '呼称(TLPA)語彙属性.xlsx'
        k = pd.read_excel(os.path.join(ccap_base, uema_excel), header=[1], sheet_name='刺激属性')
        tlpa_words = list(k['標的語'])

        # 標的語には文字 '／' が混じっているので，切り分けて別の単語と考える
        _tlpa_words = []
        for word in tlpa_words:
            for x in word.split('／'):
                _tlpa_words.append(x)

        # word2vec に存在するか否かのチェックと，存在するのであれば word2vec におけるその単語のインデックスを得る
        # word2vec では，単語は頻度順に並んでいるので最大値インデックスを取る単語が，最低頻度語ということになる
        # そのため，TLPA の刺激語のうち，最低頻度以上の全単語を使って学習を行うことを考えるためである。
        max_idx, __tlpa_words = 0, []
        for w in _tlpa_words:
            if w in self.w2v:
                idx = self.w2v.key_to_index[w]
                max_idx = idx if idx > max_idx else max_idx
                __tlpa_words.append(w)
            # else:
            #     print(f'{w} word2vec に存在せず')

        #self.tlpa_vocab = __tlpa_words.copy()
        return __tlpa_words

    def get_sala_vocab(self):
        """SALA データの読み込み"""

        with open('ccap/data/sala_data.txt', 'r', encoding='utf-8') as f:
            p = f.readlines()

        sala, sala_vocab = {}, []
        for l in p:
            _l = l.strip().split(' ')
            tag1 = jaconv.normalize(_l[-1].split('（')[0])
            tag2 = _l[-1].split('（')[-1].replace('）','')
            yomi = self.ntt_orth2hira[tag1] if tag1 in self.ntt_orth2hira else None
            if yomi == None:
                if tag1 == '轆轤首':
                    yomi = 'ろくろくび'
                elif tag1 == '擂粉木':
                    yomi = 'すりこぎ'
                elif tag1 == '擂鉢':
                    yomi = 'すりばち'
                elif tag1 == 'こま':
                    yomi = 'こま'
                elif tag1 == 'おにぎり':
                    yomi = 'おにぎり'
                sala_vocab.append(yomi)
            else:
                sala_vocab.append(tag1)
        
            sala[len(sala)] = {'tag': tag1, 
                            'code': _l[0], 
                            'feat': _l[1],
                            'yomi': yomi }
        return sala_vocab


    def set_train_vocab_minus_test_vocab(self):
        """JISX2008-1990 コードから記号とみなしうるコードを集めて ja_symbols とする
        記号だけから構成されている word2vec の項目は排除するため
        """
        self.ja_symbols = '、。，．・：；？！゛゜´\' #+ \'｀¨＾‾＿ヽヾゝゞ〃仝々〆〇ー—‐／＼〜‖｜…‥‘’“”（）〔〕［］｛｝〈〉《》「」『』【】＋−±×÷＝≠＜＞≦≧∞∴♂♀°′″℃¥＄¢£％＃＆＊＠§☆★○●◎◇◆□■△▲▽▼※〒→←↑↓〓∈∋⊆⊇⊂⊃∪∩∧∨¬⇒⇔∀∃∠⊥⌒∂∇≡≒≪≫√∽∝∵∫∬Å‰♯♭♪†‡¶◯#ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ'
        self.ja_symbols_normalized = jaconv.normalize(self.ja_symbols)

        print(f'# 訓練に用いる単語の選定 {self.traindata_size} 語')
        vocab = []; i=0
        #while (len(vocab) < self.traindata_size) and (i<len(self.ntt_freq)):
        while i<len(self.ntt_freq):
            word = self.ntt_freq[i]
            if word == '\u3000': # NTT 日本語の語彙特性で，これだけ変なので特別扱い
                i += 1
                continue

            # 良い回避策が見つからないので，以下の行の変換だけ特別扱いしている
            word = jaconv.normalize(word).replace('・','').replace('ヴ','ブ')

            if (not word in self.ja_symbols) and (not word.isascii()) and (word in self.w2v):
                if not word in self.test_vocab:
                #if not word in self.tlpa_vocab:
                    vocab.append(word)
                    if len(vocab) >= self.traindata_size:
                        return vocab
            i += 1
        return vocab


    def make_ntt_freq_data(self):
        print('# NTT日本語語彙特性 (天野，近藤; 1999, 三省堂)より頻度情報を取得')

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

        ntt_freq = {x[0]-1:{'単語':jaconv.normalize(x[1]),
                            '品詞':x[2],
                            '頻度':x[3], 
                            'よみ':jaconv.kata2hira(jaconv.normalize(x[4]))
                            } for x in tmp2}
        #ntt_freq = {x[0]-1:{'単語':x[1],'品詞':x[2],'頻度':x[3], 'よみ':x[4]} for x in tmp2}
        ntt_orth2hira = {ntt_freq[x]['単語']:ntt_freq[x]['よみ'] for x in ntt_freq}
        #print(f'#登録総単語数: {len(ntt_freq)}')

        Freq = np.zeros((len(ntt_freq)), dtype=np.uint)  #ソートに使用する numpy 配列
        for i, x in enumerate(ntt_freq):
            Freq[i] = ntt_freq[i]['頻度']

        Freq_sorted = np.argsort(Freq)[::-1]  #頻度降順に並べ替え

        # self.ntt_freq には頻度順に単語が並んでいる
        return [ntt_freq[x]['単語']for x in Freq_sorted], ntt_orth2hira


if __name__ == '__main__':
    vocab = Vocab_ja(test_name='TLPA', traindata_size=20000)

    print(colored(f'訓練データサイズ:{len(vocab.train_data)}','blue',attrs=['bold']))
    print(colored(f'検証データサイズ:{len(vocab.test_data)} テストデータ 語彙数','blue',attrs=['bold']))
    print(colored(f'音素数:{len(vocab.phone_vocab)}','blue',attrs=['bold']))
    print(colored(f'書記素数:{len(vocab.ortho_vocab)}','blue',attrs=['bold']))
    print(colored(f'最長書記素(文字)数:{vocab.max_ortho_length}, 最長音素数:{voab.max_phone_length}', 'blue',attrs=['bold']))
