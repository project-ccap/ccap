# -*- coding: utf-8 -*-
import os
import platform
import numpy as np
import requests

# if platform.system() != 'Dawrin':
#     !pip install googledrivedownloader
    
#from google_drive_downloader import GoogleDriveDownloader as gdd
import googledrivedownloader as gdd
                
# word2vec のため gensim を使う
from gensim.models import KeyedVectors
from gensim.models import Word2Vec

class ccap_w2v():


    def __init__(self, is2017=True, isColab=True):
        # local Mac で実行しているか, それとも colab 上で実行しているかを判定
        
        
        w2v_2017 = {
            'cbow200': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz',
            'sgns200': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_sgns.bin.gz',
            'cbow300': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid300_win20_neg20_sgns.bin.gz',
            'sgns300': 'http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz'
        }

        w2v_2021 = {
            'cbow128': { 'id': '1B9HGhLZOja4Xku5c_d-kMhCXn1LBZgDb',
                        'outfile': '2021_05jawiki_hid128_win10_neg10_cbow.bin.gz'},
            'sgns128': { 'id': '1OWmFOVRC6amCxsomcRwdA6ILAA5s4y4M',
                        'outfile': '2021_05jawiki_hid128_win10_neg10_sgns.bin.gz'},
            'cbow200': { 'id': '1JTkU5SUBU2GkURCYeHkAWYs_Zlbqob0s',
                        'outfile': '2021_05jawiki_hid200_win20_neg20_sgns.bin.gz'}
        }
    

        (isMac, isColab) = (True, False) if platform.system() == 'Darwin' else (False, True)
        if is2017:
            self.is2017, self.is2021 = True, False
        else:
            self.is2017, self.is2021 = False, True
            

        if isColab:
            # 形態素分析ライブラリーMeCab と 辞書(mecab-ipadic-NEologd)のインストール 
            # reference: https://qiita.com/jun40vn/items/78e33e29dce3d50c2df1
            #!apt-get -q -y install sudo file mecab libmecab-dev mecab-ipadic-utf8 git curl python-mecab
            #!git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git
            #!echo yes | mecab-ipadic-neologd/bin/install-mecab-ipadic-neologd -n
            #!pip install mecab-python3
    
            # シンボリックリンクによるエラー回避
            #!ln -s /etc/mecabrc /usr/local/etc/mecabrc    

            if self.is2017:
                response = requests.get(w2v_2017['cbow200'])
                fname = w2v_2017['cbow200'].split('/')[-1]
                with open(fname, 'wb') as f:
                    f.write(response.content)
                # word2vec の訓練済モデルを入手
                #!wget http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz
                #!wget http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_sgns.bin.gz
                #!wget http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid300_win20_neg20_sgns.bin.gz'
                #!wget http://www.cis.twcu.ac.jp/~asakawa/2017jpa/2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.g
            else:    
                #訓練済 word2vec ファイルの取得
                (f_id, outfile) = w2v_2021['sgns128']['id'], w2v_2021['sgns128']['outfile']
                #print(f_id, outfile)
                gdd.download_file_from_google_drive(file_id=f_id,
                                                    dest_path=outfile,
                                                    unzip=False,
                                                    showsize=True)
#!wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1B9HGhLZOja4Xku5c_d-kMhCXn1LBZgDb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1B9HGhLZOja4Xku5c_d-kMhCXn1LBZgDb" -O 2021_05jawiki_hid128_win10_neg10_cbow.bin.gz && rm -rf /tmp/cookies.txt
#!wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OWmFOVRC6amCxsomcRwdA6ILAA5s4y4M' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OWmFOVRC6amCxsomcRwdA6ILAA5s4y4M" -O 2021_05jawiki_hid128_win10_neg10_sgns.bin.gz && rm -rf /tmp/cookies.txt
#!wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1JTkU5SUBU2GkURCYeHkAWYs_Zlbqob0s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1JTkU5SUBU2GkURCYeHkAWYs_Zlbqob0s" -O 2021_05jawiki_hid200_win20_neg20_cbow.bin.gz && rm -rf /tmp/cookies.txt
#!wget --no-check-certificate --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VPL2Mr9JgWHik9HjRmcADoxXIdrQ3ds7' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VPL2Mr9JgWHik9HjRmcADoxXIdrQ3ds7" -O 2021_05jawiki_hid200_win20_neg20_sgns.bin.gz && rm -rf /tmp/cookies.txt

        #import MeCab
    
        HOME = os.environ['HOME']

        # word2vec データの読み込み, ファイルの所在に応じて変更してください
        if self.is2017:
            w2v_base = os.path.join(HOME, 'study/2016wikipedia/') if isMac else '.'
            w2v_file = '2017Jul_jawiki-wakati_neologd_hid200_win20_neg20_cbow.bin.gz'
            w2v_file = os.path.join(w2v_base, w2v_file)
        else:
            w2v_base = os.path.join(HOME, 'study/2019attardi_wikiextractor.git/wiki_texts/AA') if isMac else '.'
            w2v_file = '2021_05jawiki_hid128_win10_neg10_sgns.bin'

        #if self.isColab:
        #    neologd_path = "-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd"
        #else:
        #    neologd_path = "-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd"

        w2v_base = '.' if isColab else w2v_base
        w2v_file = os.path.join(w2v_base, w2v_file)
        w2v = KeyedVectors.load_word2vec_format(w2v_file, 
                                                encoding='utf-8', 
                                                unicode_errors='replace',
                                                binary=True)
        self.w2v = w2v
        #self.tagger = MeCab.Tagger('-Oyomi ' + neologd_path)    
