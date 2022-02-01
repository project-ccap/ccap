# encoding: utf8
# https://github.com/neologd/mecab-ipadic-neologd/wiki/Regexp.ja
## 解析前に行うことが望ましい文字列の正規化処理
## 辞書データを冗長にして異表記を吸収するのにも限界がある。
## 辞書データを生成する際には以下で述べる正規化処理を全て適用しているため、 解析対象のテキストに対して以下の正規化処理を適用すると、辞書中の語とマッチしやすくなる。
## mecab-ipadic-neologd のエントリを生成する際の正規化処理
## 以下にmecab-ipadic-neologd のエントリを生成する際に、処理の各所に分散している正規化処理をまとめる。
## 生成時には色々置換と削除をしているが、最後に反映されているのは以下である。
## 全角英数字は半角に置換
## ０-９=> 0-9
## Ａ-Ｚ=> A-Z
## ａ-ｚ=> a-z
## 半角カタカナは全角に置換
## 半角の濁音と半濁音の記号が1文字扱いになってるので気をつけること。
## ハイフンマイナスっぽい文字を置換
## 以下はハイフンマイナスに置換する。
## MODIFIER LETTER MINUS SIGN(U+02D7)
## ARMENIAN HYPHEN(U+058A)
## ハイフン(U+2010)
## ノンブレーキングハイフン(U+2011)
## フィギュアダッシュ(U+2012)
## エヌダッシュ(U+2013)
## Hyphen bullet(U+2043)
## 上付きマイナス(U+207B)
## 下付きマイナス(U+208B)
## 負符号(U+2212)
## 長音記号っぽい文字を置換
## 以下は全角長音記号に置換する。

## エムダッシュ(U+2014)
## ホリゾンタルバー(U+2015)
## 横細罫線(U+2500)
## 横太罫線(U+2501)
## SMALL HYPHEN-MINUS(U+FE63)
## 全角ハイフンマイナス(U+FF0D)
## 半角長音記号(U+FF70)
## 1回以上連続する長音記号は1回に置換
## 連続したら削除する。

## スーーーーーーパーーーーーー => スーパー
## チルダっぽい文字は削除
## 以下は削除する。
##
## 半角チルダ
## チルダ演算子
## INVERTED LAZY S
## 波ダッシュ
## WAVY DASH
## 全角チルダ
## 以下の全角記号は半角記号に置換
## /！”＃＄％＆’（）＊＋，−．／：；＜＞？＠［￥］＾＿｀｛｜｝
## 以下の半角記号は全角記号に置換
## ｡､･=｢｣
## 中黒とイコールは人名によく使われるので、半角にするのが難しい。
## カギ括弧はおもに全角が使われる
## 全角スペースは半角スペースに置換
## '　' => ' '
## 1つ以上の半角スペースは、1つの半角スペースに置換
## ' ' => ' '
## 解析対象テキストの先頭と末尾の半角スペースは削除
## ' テキストの前' => 'テキストの前'
## 'テキストの後 ' => 'テキストの後'
## 「ひらがな・全角カタカナ・半角カタカナ・漢字・全角記号」間に含まれる半角スペースは削除
## 検索 エンジン 自作 入門 を 買い ました ！！！ => 検索エンジン自作入門を買いました！！！
## Coding the Matrix => Coding the Matrix
## 「ひらがな・全角カタカナ・半角カタカナ・漢字・全角記号」と「半角英数字」の間に含まれる半角スペースは削除
## アルゴリズム C => アルゴリズムC
## Algorithm C => Algorithm C


from __future__ import unicode_literals
import re
import unicodedata

def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)

    s = re.sub('。。。+','。。。', s) ### 。。。を 。にする
    s = re.sub('、、、+','、、、', s) # 、、、 を 、、、
    s = re.sub('・・・+','・・・', s) # ・・・ を ・・・
    s = re.sub('へへへ+','へへへ', s) # ・・・ を ・・・
    return s

if __name__ == "__main__":
    assert "0123456789" == normalize_neologd("０１２３４５６７８９")
    assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ" == normalize_neologd("ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ")
    assert "abcdefghijklmnopqrstuvwxyz" == normalize_neologd("ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ")
    assert "!\"#$%&'()*+,-./:;<>?@[¥]^_`{|}" == normalize_neologd("！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝")
    assert "＝。、・「」" == normalize_neologd("＝。、・「」")
    assert "ハンカク" == normalize_neologd("ﾊﾝｶｸ")
    assert "o-o" == normalize_neologd("o₋o")
    assert "majikaー" == normalize_neologd("majika━")
    assert "わい" == normalize_neologd("わ〰い")
    assert "スーパー" == normalize_neologd("スーパーーーー")
    assert "!#" == normalize_neologd("!#")
    assert "ゼンカクスペース" == normalize_neologd("ゼンカク　スペース")
    assert "おお" == normalize_neologd("お             お")
    assert "おお" == normalize_neologd("      おお")
    assert "おお" == normalize_neologd("おお      ")
    assert "検索エンジン自作入門を買いました!!!" == \
        normalize_neologd("検索 エンジン 自作 入門 を 買い ました!!!")
    assert "アルゴリズムC" == normalize_neologd("アルゴリズム C")
    assert "PRML副読本" == normalize_neologd("　　　ＰＲＭＬ　　副　読　本　　　")
    assert "Coding the Matrix" == normalize_neologd("Coding the Matrix")
    assert "南アルプスの天然水Sparking Lemonレモン一絞り" == \
        normalize_neologd("南アルプスの　天然水　Ｓｐａｒｋｉｎｇ　Ｌｅｍｏｎ　レモン一絞り")
    assert "南アルプスの天然水-Sparking*Lemon+レモン一絞り" == \
        normalize_neologd("南アルプスの　天然水-　Ｓｐａｒｋｉｎｇ*　Ｌｅｍｏｎ+　レモン一絞り")
