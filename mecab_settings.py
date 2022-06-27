'''
MeCab のインストール場所の相違を吸収する。
ローカルかつユニークな設定が多いので，それら設定の相違を吸収する目的で作成。
とりわけ，Google colab 上での実施と，M1 mac, Intel Mac の相違に主たる関心がある。 

# How to use me
```python
import ccap.mecab_settings
```

あるいは
```python
from ccap.mecab_settings import wakati, parser
```
などとする
'''
import os
import MeCab

import IPython
# Google colab 上で実施しているかどうかを判定
isColab = 'google.colab' in str(IPython.get_ipython())

mecab_dic_dirs = {
    # 'Sinope':' /opt/homebrew/lib/mecab/dic/mecab-ipadic-neologd/',
    # 'Pasiphae': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
    # 'Leda': '/usr/local/lib/mecab/dic/mecab-ipadic-neologd',
    'Sinope':' /opt/homebrew/lib/mecab/dic/ipadic',
    'Pasiphae': '/usr/local/lib/mecab/dic/ipadic',
    'Leda': '/usr/local/lib/mecab/dic/ipadic',
    'colab': '/usr/share/mecab/dic/ipadic'
}


hostname = 'colab' if isColab else os.uname().nodename.split('.')[0]
if isColab:
    wakati = MeCab.Tagger('-Owakati').parse
    yomi = MeCab.Tagger('-Oyomi').parse
    #yomi = MeCab.Tagger('-Oyomi -d /content/mecab-ipadic-neologd/build/mecab-ipadic-2.7.0-20070801-neologd-20200910').parse

else:
    mecab_dic_dir = mecab_dic_dirs[hostname]
    wakati = MeCab.Tagger(f'-Owakati -d {mecab_dic_dir}').parse
    yomi = MeCab.Tagger(f'-Oyomi -d {mecab_dic_dir}').parse
