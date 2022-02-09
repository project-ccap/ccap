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
}

import os
import MeCab
hostname = os.uname().nodename.split('.')[0]
mecab_dic_dir = mecab_dic_dirs[hostname]
wakati = MeCab.Tagger(f'-Owakati -d {mecab_dic_dir}').parse
yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dic_dir}').parse

