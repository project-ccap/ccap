# MeCab のインストール場所の相違を吸収する
'''
import ccap.mecab_settings

あるいは

from ccap.mecab_settings import wakati, parser

などとする
'''

mecab_dic_dirs = {
    # 'Sinope':' /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/',
    # 'Pasiphae': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
    # 'Leda': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
    'Sinope':' /opt/homebrew/lib/mecab/dic/ipadic',
    'Pasiphae': '/usr/local/lib/mecab/dic/ipadic',
    'Leda': '/usr/local/lib/mecab/dic/ipadic',
    'colab': '/usr/share/mecab/dic/ipadic'
}

import os
import MeCab

isColab = 'google.colab' in str(get_ipython())
hostname = 'colab' if isColab else os.uname().nodename.split('.')[0]
if isColab:
    wakati = MeCab.Tagger('-Owakati').parse
    yomi = MeCab.Tagger('-Oyomi').parse
    #yomi = MeCab.Tagger('-Oyomi -d /content/mecab-ipadic-neologd/build/mecab-ipadic-2.7.0-20070801-neologd-20200910').parse

else:
    mecab_dic_dir = mecab_dic_dirs[hostname]
    wakati = MeCab.Tagger(f'-Owakati -d {mecab_dic_dir}').parse
    yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dic_dir}').parse
