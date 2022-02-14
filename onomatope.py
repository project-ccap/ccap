import jaconv
from tqdm import tqdm
from termcolor import colored
import os
import sys
import numpy as np
import typing
import matplotlib.pyplot as plt

isColab = 'google.colab' in str(get_ipython()) 
#import platform
#isColab = platform.system() == 'Linux'

#if isColab:
#    !pip install japanize_matplotlib
import japanize_matplotlib

import unicodedata
import pandas as pd

class Onomatopea():
    """オノマトペデータのハンドリング
    
    Args:
    reload: boolean
        Ture であれば，エクセルファイルからデータを再読み込みする。
        読み込むためのエクセルデータは，内部に埋め込んである
        ```python
        ccap_base = '/Users/asakawa/study/2021ccap/notebooks'
        onomatopea_excel = '2021-0325日本語オノマトペ辞典4500より.xls'
        onmtp2761 = pd.read_excel(os.path.join(ccap_base, onomatopea_excel), sheet_name='2761語')
        ```
    Description:
        内部で 辞書 data が定義してある。この辞書には以下のエントリがある
        dict_keys(['katakana', 'orth', 'orth_ids', 'phon', 'phon_ids'])
        
    Functions:

    tokenize(self, word:str)->list: 任意の単語を ['input_ids'] と ['teach_ids'] の ID リストとして返す

    graph_ids2wrd(self, graph_ids:list)->list: 書記素 ID からなるリストを渡して，対応する単語を返す

    phon_ids2wrd(self, phon_ids:list)->list: 音韻 ID からなるリストを渡して，対応する単語を返す
    
    draw_phoneme_freq(): 音素の頻度グラフを描画する
            
    draw_grapheme_freq(): 書記素の頻度グラフを描画する

    __len__(self)->int: 登録されているオノマトペ総数を返す
    
    __call__(self, args:list=None)->dict: 登録されているオノマトペの情報を返す。
        引数として，数字 または 単語 をとる
        
    Variables:
    data: オノマトペ情報を入れた dict
    grapheme: 書記素データ
    grapheme_freq: 書記素の頻度
    orthography: grapheme に同じ
    orthography_freq: grapheme_freq に同じ
    phoneme: 音素データ
    phoneme_freq: 音素データの頻度情報
    phonology: phoneme に同じ
    phonology_freq: phoneme_freq に同じ
    vocab: 全オノマトペ単語からなるリスト
    """
    
    def __init__(self, vocab:list=None, reload:bool=False)->None:
        super().__init__()

        if vocab == None:
            vocab = [\
                     'あくせく', 'あたふた', 'あっけらかん', 'あっはっは', 'あっぱらぱー', 'あっぷあっぷ', 'あはあは', 'あはは', 'あやふや', 'あわあわ', 'あわわ', 'あんあん', 'あんぐり', 'あーん', 'いがいが', 'いけしゃーしゃー', 'いけずーずー', 'いけつんつん', 'いじいじ', 'いそいそ', 'いひひ', 'いらいら', 'うえんうえん', 'うきうき', 'うぎゃー', 'うざうざ', 'うしうし', 'うじうじ', 'うじゃうじゃ', 'うじょうじょ', 'うずうず', 'うだうだ', 'うっしっし', 'うっすら', 'うっとり', 'うつらうつら', 'うとうと', 'うとっ', 'うねうね', 'うはうは', 'うひひ', 'うひょひょ', 'うふっ', 'うふふ', 'うやむや', 'うようよ', 'うらうら', 'うらら', 'うるうる', 'うるっ', 'うろうろ', 'うろちょろ', 'うわーん', 'うんざり', 'えっちらおっちら', 'えへへ', 'えへらえへら', 'えへん', 'えんえん', 'えーん', 'おいおい', 'おぎゃーおぎゃー', 'おずおず', 'おたおた', 'おっとり', 'おどおど', 'おひゃりこひゃり', 'おほほ', 'おほん', 'おめおめ', 'おろおろ', 'おんおん', 'おーおー', 'かかか', 'かきかき', 'かくっ', 'かくん', 'かくんかくん', 'かさかさ', 'かさこそ', 'かさっ', 'かさり', 'かしゃかしゃ', 'かしゃっ', 'かしゃん', 'かすかす', 'かたかた', 'かたこと', 'かたっ', 'かたり', 'かたん', 'かたんかたん', 'かたんことん', 'かちかち', 'かちっ', 'かちゃかちゃ', 'かちゃり', 'かちゃん', 'かちり', 'かちん', 'かちんかちん', 'かちんこちん', 'かっ', 'かっか', 'かっかっ', 'かっきり', 'かっくり', 'かっくん', 'かっしゃり', 'かっしゃんかっしゃん', 'かったん', 'かったんこっとん', 'かっちゃん', 'かっちんかっちん', 'かっぽかっぽ', 'かっぽん', 'かつかつ', 'かつっ', 'かつん', 'かばかば', 'かぽかぽ', 'かぽっ', 'からから', 'からころ', 'からっ', 'からり', 'かりかり', 'かりっ', 'かりり', 'かりん', 'かん', 'かんから', 'かんかん', 'かんらかんら', 'かーっ', 'かーん', 'ががっ', 'がくがく', 'がくっ', 'がくり', 'がくりがくり', 'がくん', 'がくんがくん', 'がさがさ', 'がさごそ', 'がさっ', 'がさり', 'がしがし', 'がしっ', 'がしゃがしゃ', 'がしゃっ', 'がしゃん', 'がじがじ', 'がたがた', 'がたごと', 'がたっ', 'がたぴし', 'がたり', 'がたん', 'がたんがたん', 'がたんごとん', 'がちがち', 'がちっ', 'がちゃがちゃ', 'がちゃっ', 'がちゃり', 'がちゃん', 'がちり', 'がちん', 'がちんがちん', 'がっ', 'がっかり', 'がっがっ', 'がっき', 'がっくり', 'がっしゃり', 'がっしゃん', 'がっしり', 'がったん', 'がったんがったん', 'がっちゃん', 'がっちり', 'がっちんがっちん', 'がっぷり', 'がっぽがっぽ', 'がっぽり', 'がつがつ', 'がつっ', 'がつん', 'がはがは', 'がはは', 'がばがば', 'がばちょ', 'がばっ', 'がびがび', 'がびょーん', 'がびーん', 'がぶがぶ', 'がぶっ', 'がぶり', 'がぶりがぶり', 'がぷがぷ', 'がぼがぼ', 'がぼっ', 'がみがみ', 'がやがや', 'がらがら', 'がらがらぺっ', 'がらり', 'がらんがらん', 'がりがり', 'がりっ', 'がりり', 'がん', 'がんがらがん', 'がんがん', 'がーっ', 'がーん', 'きくきく', 'きざきざ', 'きしっ', 'きちきち', 'きちっ', 'きちん', 'きちんきちん', 'きっ', 'きっかし', 'きっかり', 'きっぱり', 'きゃっ', 'きゃっきゃっ', 'きゃぴきゃぴ', 'きゃんきゃん', 'きゃー', 'きゃーきゃー', 'きゅっ', 'きゅっきゅっ', 'きゅん', 'きゅー', 'きゅーきゅー', 'きゅーん', 'きょときょと', 'きょとっ', 'きょとん', 'きょろきょろ', 'きょろっ', 'きょろり', 'きらきら', 'きらっ', 'きらら', 'きらり', 'きりきり', 'きろきろ', 'きろっ', 'きろり', 'きんきら', 'きんきらきん', 'きんきん', 'きんきんきらきら', 'きんぴか', 'きんぴかぴか', 'きー', 'きーきー', 'きーっ', 'きーん', 'ぎく', 'ぎくぎく', 'ぎくしゃく', 'ぎくっ', 'ぎくり', 'ぎくん', 'ぎしぎし', 'ぎしっ', 'ぎたぎた', 'ぎちぎち', 'ぎっくん', 'ぎっこんぎっこん', 'ぎっしり', 'ぎとぎと', 'ぎとっ', 'ぎゃっ', 'ぎゃはは', 'ぎゃふん', 'ぎゃんぎゃん', 'ぎゃー', 'ぎゃーぎゃー', 'ぎゅっ', 'ぎゅっぎゅっ', 'ぎゅー', 'ぎょえー', 'ぎょぎょ', 'ぎょっ', 'ぎょろぎょろ', 'ぎょろっ', 'ぎょろり', 'ぎらkつ', 'ぎらぎら', 'ぎらり', 'ぎりぎり', 'ぎりっ', 'ぎりり', 'ぎろっ', 'ぎんぎらぎん', 'ぎんぎん', 'ぎんぎんぎらぎら', 'ぎー', 'ぎーぎー', 'ぎーこぎーこ', 'ぎーっ', 'ぎーとんぎーとん', 'くい', 'くいくい', 'くくっ', 'くくー', 'くさくさ', 'くしゃくしゃ', 'くしゃっ', 'くしゅん', 'くすくす', 'くすっ', 'くすり', 'くすりくすり', 'くすん', 'くたくた', 'くたっ', 'くだくだ', 'くちゃくちゃ', 'くっ', 'くっく', 'くっくっ', 'くつくつ', 'くどくど', 'くにくに', 'くにゃ', 'くにゃくにゃ', 'くにゃり', 'くにゅくにゅ', 'くねくね', 'くよくよ', 'くらくら', 'くらっ', 'くらり', 'くらりくらり', 'くー', 'くーくー', 'くーっ', 'ぐい', 'ぐいぐい', 'ぐきり', 'ぐぎり', 'ぐぐっ', 'ぐさぐさ', 'ぐさっ', 'ぐさり', 'ぐしゃぐしゃ', 'ぐしゃっ', 'ぐしょぐしょ', 'ぐじぐじ', 'ぐじゃぐじゃ', 'ぐじゅぐじゅ', 'ぐじょぐじょ', 'ぐすぐす', 'ぐすっ', 'ぐすり', 'ぐすりぐすり', 'ぐすん', 'ぐずぐず', 'ぐずらぐずら', 'ぐずり', 'ぐたぐた', 'ぐたっ', 'ぐだぐだ', 'ぐちぐち', 'ぐちゃぐちゃ', 'ぐちゅぐちゅ', 'ぐちょぐちょ', 'ぐっ', 'ぐっぐっ', 'ぐっしょり', 'ぐっすり', 'ぐっすん', 'ぐったり', 'ぐつぐつ', 'ぐでんぐでん', 'ぐにゃ', 'ぐにゃぐにゃ', 'ぐにゃらぐにゃら', 'ぐにゃり', 'ぐびぐび', 'ぐびっ', 'ぐびり', 'ぐびりぐびり', 'ぐらぐら', 'ぐらっ', 'ぐらり', 'ぐらりぐらり', 'ぐりぐり', 'ぐんぐん', 'ぐんにゃり', 'ぐー', 'ぐーぐー', 'ぐーすか', 'ぐーたら', 'ぐーっ', 'けけ', 'けたけた', 'けちけち', 'けちょん', 'けっけっ', 'けほけほ', 'けらけら', 'けろけろ', 'けろっ', 'けろり', 'げたげた', 'げたっ', 'げっ', 'げっそり', 'げほげほ', 'げらげら', 'げろげろ', 'げろっ', 'げんなり', 'げー', 'げーげー', 'こうこう', 'こきっ', 'こくこく', 'こくっ', 'こくり', 'こくりこくり', 'こくん', 'こしこし', 'こせこせ', 'こそこそ', 'こそっ', 'こそり', 'こちこち', 'こちっ', 'こちゃこちゃ', 'こちり', 'こちん', 'こちんこちん', 'こっくり', 'こっくりこっくり', 'こっくん', 'こっつん', 'こっつんこ', 'こってり', 'こっとんこっとん', 'こつこつ', 'こつっ', 'こつり', 'こつん', 'こてこて', 'こてっ', 'ことこと', 'ことっ', 'ことり', 'ことん', 'ことんことん', 'こほこほ', 'こほんこほん', 'こぽこぽ', 'こりこり', 'こりっ', 'ころころ', 'ころっ', 'ころり', 'ころりん', 'ころりんしゃん', 'ころん', 'ころんころん', 'こん', 'こんがり', 'こんこん', 'こんもり', 'こーん', 'ごきごき', 'ごきっ', 'ごくごく', 'ごくっ', 'ごくり', 'ごくりごくり', 'ごくん', 'ごくんごくん', 'ごしごし', 'ごしゃごしゃ', 'ごじゃごじゃ', 'ごそ', 'ごそごそ', 'ごそっ', 'ごそり', 'ごたごた', 'ごたすた', 'ごちっ', 'ごちゃ', 'ごちゃくちゃ', 'ごちゃごちゃ', 'ごちょごちょ', 'ごちん', 'ごっくごっく', 'ごっくり', 'ごっくん', 'ごっそり', 'ごった', 'ごっちゃ', 'ごってり', 'ごっとんごっとん', 'ごっぽり', 'ごつごつ', 'ごつっ', 'ごつり', 'ごつん', 'ごてごて', 'ごてっ', 'ごとごと', 'ごとっ', 'ごとり', 'ごとりごとり', 'ごとん', 'ごとんごとん', 'ごにょごにょ', 'ごぶごぶ', 'ごほごほ', 'ごほん', 'ごほんごほん', 'ごぼごぼ', 'ごぽごぽ', 'ごみごみ', 'ごりごり', 'ごりっ', 'ごろごろ', 'ごろちゃら', 'ごろっ', 'ごろり', 'ごろりごろり', 'ごろん', 'ごろんごろん', 'ごわごわ', 'ごん', 'ごんごん', 'ごーっ', 'さくさく', 'さくっ', 'さくり', 'さくりさくり', 'さっ', 'さっくり', 'さっさ', 'さっさっ', 'さっぱり', 'さばさば', 'さぶさぶ', 'さめざめ', 'さやさや', 'さらさら', 'さらっ', 'さらり', 'さらりさらり', 'さわさわ', 'さわっ', 'さんさん', 'さーさー', 'さーっ', 'ざくざく', 'ざくっ', 'ざくり', 'ざぐり', 'ざざっ', 'ざざんざ', 'ざっ', 'ざっく', 'ざっくざっく', 'ざっくばらん', 'ざっくり', 'ざばっ', 'ざぶざぶ', 'ざぶっ', 'ざぶん', 'ざらざら', 'ざらり', 'ざらりざらり', 'ざらりん', 'ざわざわ', 'ざんざ', 'ざんざら', 'ざんざん', 'ざんぶり', 'ざーざー', 'ざーっ', 'しおしお', 'しく', 'しくしく', 'しくりしくり', 'しげしげ', 'しこしこ', 'しこっ', 'しっかり', 'しっちゃかめっちゃか', 'しっちゃこっちゃ', 'しっとり', 'しっぽり', 'しとしと', 'しとっ', 'しとり', 'しどもど', 'しどろもどろ', 'しなしな', 'しなっ', 'しねしね', 'しばしば', 'しぼしぼ', 'しゃかしゃか', 'しゃきしゃき', 'しゃきっ', 'しゃっきり', 'しゃっしゃっ', 'しゃなりしゃなり', 'しゃらしゃら', 'しゃらりしゃらり', 'しゃりしゃり', 'しゃりんしゃりん', 'しゃわしゃわ', 'しゃん', 'しゃんしゃん', 'しゃーしゃー', 'しゅっ', 'しゅっしゅっ', 'しゅるしゅる', 'しゅるっ', 'しゅん', 'しゅんしゅん', 'しゅー', 'しょぼしょぼ', 'しょぼっ', 'しょぼん', 'しょりしょり', 'しょろしょろ', 'しょんぼり', 'しらっ', 'しれっ', 'しわくしゃ', 'しわくちゃ', 'しわしわ', 'しん', 'しんなり', 'しんねり', 'しんみり', 'じくじく', 'じぐざぐ', 'じじ', 'じたじた', 'じたばた', 'じたんばたん', 'じっ', 'じっくり', 'じっとり', 'じとじと', 'じとっ', 'じとり', 'じぶじぶ', 'じみじみ', 'じめじめ', 'じゃかじゃか', 'じゃかすか', 'じゃくり', 'じゃっ', 'じゃぶじゃぶ', 'じゃぶん', 'じゃぼん', 'じゃらくら', 'じゃらじゃら', 'じゃらん', 'じゃらんじゃらん', 'じゃりじゃり', 'じゃりっ', 'じゃん', 'じゃんじゃか', 'じゃんじゃん', 'じゃーじゃー', 'じゅくじゅく', 'じゅっ', 'じゅっじゅっ', 'じゅるじゅる', 'じゅるっ', 'じゅわじゅわ', 'じゅわっ', 'じゅー', 'じゅーじゅー', 'じょきじょき', 'じょきり', 'じょりじょり', 'じょりっ', 'じょろりじょろり', 'じりじり', 'じりりじりり', 'じろじろ', 'じろっ', 'じろり', 'じろりじろり', 'じわじわ', 'じわっ', 'じわり', 'じわりじわり', 'じん', 'じんじん', 'じんたった', 'じんわり', 'じーじー', 'じーっ', 'じーん', 'すい', 'すいすい', 'すかすか', 'すかっ', 'すくすく', 'すくっ', 'すこん', 'すごすご', 'すすっ', 'すたこら', 'すたこらさっさ', 'すたすた', 'すたっ', 'すっ', 'すっき', 'すっく', 'すっすっ', 'すってんてれつく', 'すってんてん', 'すっとんとん', 'すっぱり', 'すっぽすっぽ', 'すとん', 'すぱっ', 'すぱり', 'すぽぽん', 'すやすや', 'すらすら', 'すらっ', 'すらり', 'するする', 'するり', 'すんなり', 'すーすー', 'すーっ', 'ずい', 'ずいこん', 'ずかずか', 'ずかり', 'ずきずき', 'ずきっ', 'ずきり', 'ずきん', 'ずきんずきん', 'ずくずく', 'ずけずけ', 'ずしりずしり', 'ずずっ', 'ずたずた', 'ずたぼろ', 'ずっ', 'ずっずっ', 'ずっぷり', 'ずどん', 'ずばずば', 'ずばっ', 'ずばり', 'ずばりずばり', 'ずぶずぶ', 'ずぶっ', 'ずぶり', 'ずぼずぼ', 'ずぼっ', 'ずらずら', 'ずらっ', 'ずらり', 'ずるずる', 'ずるっ', 'ずん', 'ずんぐり', 'ずんぐりむっくり', 'ずんずん', 'ずーずー', 'ずーっ', 'せかせか', 'せんせん', 'ぜりぜり', 'ぜーぜー', 'そくそく', 'そげそげ', 'そそくさ', 'そよ', 'そよそよ', 'そより', 'そろそろ', 'そわそわ', 'ぞくぞく', 'ぞくっ', 'ぞくり', 'ぞぞっ', 'ぞっ', 'ぞよぞよ', 'ぞりぞり', 'ぞりっ', 'ぞろぞろ', 'ぞろっ', 'ぞろり', 'ぞろりぞろり', 'ぞーっ', 'たかたか', 'たじたじ', 'たたたた', 'たたーっ', 'たっ', 'たった', 'たったかたったか', 'たったっ', 'たっぷり', 'たどたど', 'たぶたぶ', 'たぷたぷ', 'たぷんたぷん', 'たぽたぽ', 'たらたら', 'たわわ', 'たん', 'たんたん', 'たんまり', 'たー', 'だくだく', 'だだだだ', 'だだーっ', 'だっ', 'だっだっ', 'だぶだぶ', 'だぼだぼ', 'だぼん', 'だらだら', 'だらりだらり', 'だんだん', 'だー', 'だーん', 'ちかちか', 'ちかっ', 'ちかり', 'ちくちく', 'ちくっ', 'ちくり', 'ちくりちくり', 'ちくん', 'ちびちび', 'ちびりちびり', 'ちゃかちゃか', 'ちゃかぽこ', 'ちゃきちゃき', 'ちゃっ', 'ちゃっかり', 'ちゃぷちゃぷ', 'ちゃぽっ', 'ちゃぽん', 'ちゃらちゃら', 'ちゃらんちゃらん', 'ちゃりん', 'ちゃりんちゃりん', 'ちゃん', 'ちゃんちき', 'ちゃんちゃか', 'ちゃんちゃん', 'ちゃんちゃんばらばら', 'ちゃんぽん', 'ちゅっ', 'ちゅるちゅる', 'ちゅーちゅー', 'ちょきちょき', 'ちょきんちょっきり', 'ちょこちょこ', 'ちょこっ', 'ちょこん', 'ちょっくら', 'ちょっぴり', 'ちょびっ', 'ちょぼちょぼ', 'ちょぼっ', 'ちょろちょろ', 'ちょろり', 'ちょん', 'ちらちら', 'ちらっ', 'ちらほら', 'ちらり', 'ちらりちらり', 'ちりちり', 'ちりとんてちん', 'ちりんちりん', 'ちろちろ', 'ちろっ', 'ちん', 'ちんからり', 'ちんちん', 'ちんとんしゃん', 'ちんどん', 'ちーちーぱーぱー', 'つい', 'ついつい', 'つかつか', 'つくづく', 'つけつけ', 'つっけんどん', 'つべこべ', 'つやつや', 'つらつら', 'つるつる', 'つん', 'つんけん', 'つんつん', 'つーつー', 'てかっ', 'てかてか', 'てきぱき', 'てくてく', 'てけてけ', 'てらてら', 'てれつく', 'てれつくてん', 'てれんこてれんこ', 'てろてろ', 'てんつるてん', 'てんてこ', 'てんてんてれつく', 'てんやわんや', 'でこぼこ', 'でっぷり', 'ででん', 'でぶでぶ', 'でんでん', 'とくとく', 'とげとげ', 'とことこ', 'とっちり', 'とっと', 'とっとことっとこ', 'とっとっ', 'とっぷり', 'とてちてたー', 'ととと', 'ととんとんとん', 'とぷんとぷん', 'とほほ', 'とぼとぼ', 'とろっ', 'とろとろ', 'とろり', 'とろん', 'とん', 'とんてんかん', 'とんとん', 'とーとー', 'とーん', 'どかっ', 'どかどか', 'どかり', 'どかん', 'どかんどしん', 'どがちゃが', 'どきっ', 'どきどき', 'どきり', 'どきんどきん', 'どぎまぎ', 'どくどく', 'どさくさ', 'どさっ', 'どさどさ', 'どさり', 'どさん', 'どしどし', 'どしゃどしゃ', 'どしり', 'どしんどしん', 'どしんばたん', 'どすっ', 'どすん', 'どたっ', 'どたどた', 'どたばた', 'どたり', 'どたりばたり', 'どたん', 'どたんばたん', 'どっ', 'どっきり', 'どっきんどっきん', 'どっさり', 'どっしどっし', 'どっと', 'どっぷり', 'どどっ', 'どどんがどん', 'どばっ', 'どばどば', 'どぶどぶ', 'どぶん', 'どぼどぼ', 'どぼん', 'どやどや', 'どろっ', 'どろどろ', 'どろん', 'どん', 'どんちき', 'どんちゃか', 'どんちゃん', 'どんつくどんどこ', 'どんどこ', 'どんどん', 'どんどんかかか', 'どんぱち', 'どんぶりこ', 'どんより', 'どー', 'どーっ', 'どーどー', 'どーん', 'なみなみ', 'なよなよ', 'なんなん', 'にかっ', 'にこっ', 'にこにこ', 'にこり', 'にたっ', 'にたにた', 'にたりにたり', 'にちゃにちゃ', 'にっ', 'にっこにこ', 'にっこり', 'にやっ', 'にやにや', 'にやり', 'にゅるにゅる', 'にゅるり', 'にょきにょき', 'にょっき', 'にょっきり', 'にょろっ', 'にょろにょろ', 'にょろり', 'にんまり', 'にーっ', 'ぬくぬく', 'ぬけぬけ', 'ぬたくた', 'ぬっく', 'ぬっくり', 'ぬっぺり', 'ぬめっ', 'ぬめぬめ', 'ぬめり', 'ぬらっ', 'ぬらぬら', 'ぬらりくらり', 'ぬるっ', 'ぬるぬる', 'ぬるり', 'ぬるりぬるり', 'ぬんめり', 'ぬーっ', 'ねちねち', 'ねちゃっ', 'ねちゃねちゃ', 'ねっとり', 'ねとっ', 'ねとねと', 'ねばねば', 'のこのこ', 'のさのさ', 'のしのし', 'のそのそ', 'のそり', 'のそりのそり', 'のたのた', 'のたらのたら', 'のたり', 'のたりのたり', 'のっしのっし', 'のっそり', 'のびのび', 'のほほん', 'のめのめ', 'のらくら', 'のらりくらり', 'のらりのらり', 'のろのろ', 'のろりのろり', 'のんびり', 'のんべんぐらり', 'のんべんだらり', 'のーのー', 'はきはき', 'はた', 'はたはた', 'はたり', 'はちゃめちゃ', 'はっ', 'はっきり', 'はっし', 'はった', 'はっはっ', 'ははは', 'はらっ', 'はらはら', 'はらり', 'ばかすか', 'ばきっ', 'ばきばき', 'ばきゅん', 'ばきん', 'ばくっ', 'ばくばく', 'ばくりばくり', 'ばさっ', 'ばさばさ', 'ばさらばさら', 'ばさり', 'ばしっ', 'ばしばし', 'ばしゃっ', 'ばしり', 'ばしん', 'ばたっ', 'ばたばた', 'ばたんきゅー', 'ばちっ', 'ばちばち', 'ばちゃっ', 'ばちん', 'ばっ', 'ばっさばっさ', 'ばっさり', 'ばっばっ', 'ばらっ', 'ばらばら', 'ばらり', 'ばりっ', 'ばりばり', 'ばんばん', 'ばーん', 'ぱかぱか', 'ぱきっ', 'ぱきぱき', 'ぱきん', 'ぱくっ', 'ぱくぱく', 'ぱくり', 'ぱくりぱくり', 'ぱくん', 'ぱこっ', 'ぱさっ', 'ぱさぱさ', 'ぱさり', 'ぱしっ', 'ぱしぱし', 'ぱしゃっ', 'ぱしん', 'ぱたぱた', 'ぱたり', 'ぱちくり', 'ぱちっ', 'ぱちぱち', 'ぱちゃっ', 'ぱちり', 'ぱっ', 'ぱっくん', 'ぱっさり', 'ぱっちり', 'ぱっぱ', 'ぱっぱか', 'ぱっぱっ', 'ぱっぱらぱー', 'ぱぱっ', 'ぱふぱふ', 'ぱらっ', 'ぱらぱら', 'ぱらり', 'ぱりっ', 'ぱりぱり', 'ぱりん', 'ぱん', 'ぱんぱかぱん', 'ぱんぱん', 'ぱー', 'ぱーぱー', 'ひくっ', 'ひくひく', 'ひそひそ', 'ひたひた', 'ひっ', 'ひっく', 'ひっひっ', 'ひひひ', 'ひやっ', 'ひやひや', 'ひやり', 'ひやりひやり', 'ひゅっ', 'ひゅるひゅる', 'ひゅるるん', 'ひゅー', 'ひゅーひゅー', 'ひゅーふっ', 'ひょい', 'ひょこ', 'ひょこひょこ', 'ひょろっ', 'ひょろひょろ', 'ひょろり', 'ひょろん', 'ひらっ', 'ひらひら', 'ひらり', 'ひらりひらり', 'ひりっ', 'ひりひり', 'ひりり', 'ひろひろ', 
                     'ひんやり', 'ひー', 'ひーこら', 'ひーひー', 'びかびか', 'びくっ', 'びくびく', 'びくり', 'びしっ', 'びしびし', 'びしゃっ', 'びしゃびしゃ', 'びしゃり', 'びしょびしょ', 'びしり', 'びたっ', 'びたり', 'びちゃっ', 'びちゃびちゃ', 'びちょびちょ', 'びっ', 'びっくら', 'びっくり', 'びっしょり', 'びっしり', 'びっちょり', 'びびっ', 'びゅっ', 'びゅわーん', 'びゅん', 'びゅんびゅん', 'びゅー', 'びゅーびゅー', 'びよーん', 'びりっ', 'びりびり', 'びりり', 'びりん', 'びんびん', 'びーん', 'ぴかっ', 'ぴかぴか', 'ぴかり', 'ぴかりぴかり', 'ぴきぴき', 'ぴくっ', 'ぴくぴく', 'ぴくり', 'ぴくりぴくり', 'ぴくん', 'ぴくんぴくん', 'ぴこぴこ', 'ぴしっ', 'ぴしぴし', 'ぴしゃ', 'ぴしゃっ', 'ぴしゃぴしゃ', 'ぴしゃり', 'ぴしり', 'ぴたっ', 'ぴたぴた', 'ぴたり', 'ぴたん', 'ぴたんぴたん', 'ぴちっ', 'ぴちぴち', 'ぴちゃっ', 'ぴちゃぴちゃ', 'ぴっ', 'ぴっかぴか', 'ぴっかりこ', 'ぴったり', 'ぴっちり', 'ぴぴっ', 'ぴやぴちゃ', 'ぴゅっ', 'ぴゅん', 'ぴゅんぴゅん', 'ぴゅー', 'ぴゅーぴゅー', 'ぴょこぴょこ', 'ぴょー', 'ぴらぴら', 'ぴりっ', 'ぴりぴり', 'ぴりり', 'ぴん', 'ぴんしゃん', 'ぴんぴん', 'ぴー', 'ぴーちくぱーちく', 'ぴーひゃら', 'ぴーぴー', 'ぴーん', 'ふかふか', 'ふかり', 'ふがふが', 'ふくふく', 'ふさふさ', 'ふっ', 'ふっくら', 'ふっくり', 'ふっさり', 'ふっつり', 'ふっふっ', 'ふにゃ', 'ふにゃふにゃ', 'ふにゃり', 'ふふ', 'ふふん', 'ふやふや', 'ふよふよ', 'ふらふら', 'ふらりふらり', 'ふるふる', 'ふわっ', 'ふわふわ', 'ふわり', 'ふわりふわり', 'ふんわか', 'ふんわり', 'ふー', 'ぶいぶい', 'ぶぉーっ', 'ぶかぶか', 'ぶかぶかどんどん', 'ぶくっ', 'ぶくぶく', 'ぶくんぶくん', 'ぶすっ', 'ぶすぶす', 'ぶすり', 'ぶすりぶすり', 'ぶちっ', 'ぶちぶち', 'ぶっつり', 'ぶっつん', 'ぶつくさ', 'ぶつっ', 'ぶつぶつ', 'ぶつり', 'ぶつん', 'ぶよぶよ', 'ぶらぶら', 'ぶらり', 'ぶらりぶらり', 'ぶらん', 'ぶらんぶらん', 'ぶりぶり', 'ぶるっ', 'ぶるぶる', 'ぶるり', 'ぶるる', 'ぶるん', 'ぶるんぶるん', 'ぶわぶわ', 'ぶんぶん', 'ぶーぶー', 'ぷい', 'ぷかっ', 'ぷかぷか', 'ぷかり', 'ぷかりぷかり', 'ぷくっ', 'ぷくぷく', 'ぷくり', 'ぷくん', 'ぷすぷす', 'ぷすり', 'ぷちっ', 'ぷっ', 'ぷっくり', 'ぷっつ', 'ぷっつり', 'ぷっつん', 'ぷっぷっ', 'ぷつっ', 'ぷつぷつ', 'ぷつり', 'ぷつん', 'ぷよぷよ', 'ぷらぷら', 'ぷらりぷらり', 'ぷりっ', 'ぷりぷり', 'ぷりん', 'ぷりんぷりん', 'ぷるっ', 'ぷるぷる', 'ぷるる', 'ぷるん', 'ぷん', 'ぷんぷん', 'ぷー', 'へこへこ', 'へたへた', 'へっへっ', 'へとへと', 'へどもど', 'へなへな', 'へへへ', 'へべけれ', 'へらへら', 'へろへろ', 'べこべこ', 'べしゃ', 'べしゃり', 'べそべそ', 'べたっ', 'べたべた', 'べたり', 'べたりべたり', 'べたん', 'べちゃくちゃ', 'べちゃべちゃ', 'べちょっ', 'べちょべちょ', 'べったりべっちゃり', 'べっとり', 'べとっ', 'べとべと', 'べべんべんべん', 'べらべら', 'べらり', 'べりっ', 'べりべり', 'べろっ', 'べろべろ', 'べろり', 'べろん', 'べろんべろん', 'べんべん', 'ぺこぺこ', 'ぺしゃり', 'ぺたっ', 'ぺたぺた', 'ぺたり', 'ぺたりぺたり', 'ぺたん', 'ぺちゃくちゃ', 'ぺちゃぺちゃ', 'ぺちょ', 'ぺったり', 'ぺったんぺったん', 'ぺっとり', 'ぺっぺっ', 'ぺらっ', 'ぺらぺら', 'ぺりっ', 'ぺりぺり', 'ぺろっ', 'ぺろぺろ', 'ぺろり', 'ぺろん', 'ぺろんぺろん', 'ぺんぺん', 'ほいほい', 'ほかほか', 'ほくほく', 'ほこっ', 'ほこほこ', 'ほっこり', 'ほっそり', 'ほっほっ', 'ほのぼの', 'ほほほ', 'ほやほや', 'ほろっ', 'ほろほろ', 'ほろり', 'ほろりほろり', 'ほわっ', 'ほんのり', 'ほんわか', 'ぼいん', 'ぼかすか', 'ぼかん', 'ぼきっ', 'ぼきぼき', 'ぼきり', 'ぼきん', 'ぼくっ', 'ぼくぼく', 'ぼけっ', 'ぼこっ', 'ぼこぼこ', 'ぼさっ', 'ぼさぼさ', 'ぼしゃっ', 'ぼそっ', 'ぼそぼそ', 'ぼそり', 'ぼそん', 'ぼたっ', 'ぼたぼた', 'ぼたり', 'ぼたん', 'ぼちぼち', 'ぼちゃぼちゃ', 'ぼっ', 'ぼっこり', 'ぼっちゃり', 'ぼってり', 'ぼつっ', 'ぼつぼつ', 'ぼつり', 'ぼつりぼつり', 'ぼつん', 'ぼてっ', 'ぼてぼて', 'ぼてれん', 'ぼとぼと', 'ぼとり', 'ぼとん', 'ぼやっ', 'ぼやぼや', 'ぼやり', 'ぼりっ', 'ぼりぼり', 'ぼろぼろ', 'ぼろん', 'ぼわっ', 'ぼわぼわ', 'ぼん', 'ぼんきゅっぼん', 'ぼんぼこ', 'ぼんぼん', 'ぼんやり', 'ぼーっ', 'ぼーぼー', 'ぼーん', 'ぽかっ', 'ぽかぽか', 'ぽかり', 'ぽかりぽかり', 'ぽかん', 'ぽきっ', 'ぽきぽき', 'ぽきり', 'ぽきん', 'ぽこぽこ', 'ぽこり', 'ぽこん', 'ぽたっ', 'ぽたぽた', 'ぽたり', 'ぽたん', 'ぽちぽち', 'ぽちゃっ', 'ぽちゃぽちゃ', 'ぽちゃん', 'ぽっ', 'ぽっかり', 'ぽっきり', 'ぽっこり', 'ぽっちゃり', 'ぽっちり', 'ぽってり', 'ぽっぽっ', 'ぽつっ', 'ぽつぽつ', 'ぽつり', 'ぽつん', 'ぽてっ', 'ぽてぽて', 'ぽとっ', 'ぽとぽと', 'ぽとり', 'ぽとりぽとり', 'ぽとん', 'ぽやぽや', 'ぽよぽよ', 'ぽりぽり', 'ぽろっ', 'ぽろぽろ', 'ぽろり', 'ぽろりぽろり', 'ぽろん', 'ぽわん', 'ぽん', 'ぽんぽこ', 'ぽんぽこぽん', 'ぽんぽん', 'ぽーん', 'まごまご', 'まざまざ', 'まじまじ', 'まろまろ', 'まんじり', 'みし', 'みしみし', 'みしり', 'みしりみしり', 'みっしり', 'みりみり', 'みりり', 'むかっ', 'むかむか', 'むくっ', 'むくむく', 'むぐむぐ', 'むしむし', 'むしゃくしゃ', 'むしゃむしゃ', 'むしゃり', 'むしゃりむしゃり', 'むすっ', 'むずむず', 'むちっ', 'むちむち', 'むっ', 'むっく', 'むっくり', 'むっちり', 'むっつり', 'むにゃむにゃ', 'むにゅむにゅ', 'むらっ', 'むらむら', 'むわっ', 'むん', 'むんむ', 'むんむん', 'むーっ', 'めきめき', 'めそめそ', 'めためた', 'めちゃくちゃ', 'めちゃめちゃ', 'めっきり', 'めらめら', 'めりっ', 'めりめり', 'めーめー', 'もくもく', 'もぐもぐ', 'もこっ', 'もこもこ', 'もごもご', 'もさもさ', 'もしゃもしゃ', 'もじもじ', 'もじゃくじゃ', 'もじゃもじゃ', 'もそもそ', 'もぞもぞ', 'もたもた', 'もだもだ', 'もっこり', 'もっさり', 'もったらもったら', 'もったり', 'もにゃもにゃ', 'ももくちゃ', 'もやっ', 'もやもや', 'もりもり', 'もろもろ', 'もわっ', 'もわもわ', 'もー', 'やいのやいの', 'やいやい', 'やきもき', 'やっさもっさ', 'やわやわ', 'やんさもんさ', 'やんわり', 'ゆさゆさ', 'ゆさりゆさり', 'ゆっくり', 'ゆっさゆっさ', 'ゆったり', 'ゆらっ', 'ゆらゆら', 'ゆらり', 'ゆらりゆらり', 'ゆるゆる', 'ゆるり', 'よたよた', 'よちよち', 'よぼよぼ', 'よよ', 'よれよれ', 'よろよろ', 'よろりよろり', 'らったった', 'らんらん', 'りゅーりゅー', 'りん', 'るんるん', 'れろれろ', 'わいわい', 'わくわく', 'わさわさ', 'わしわし', 'わたわた', 'わちゃわちゃ', 'わっ', 'わっさわっさ', 'わなわな', 'わはは', 'わやくや', 'わやわや', 'わらわら', 'わんさ', 'わんさか', 'わんさわんさ', 'わんわ', 'わんわん', 'わーっ', 'わーわー', 'わーん']
            
        if reload == True: # エクセルからの再読み込みをする
            print("'日本語オノマトペ辞典4500より.xls' は著作権の問題があり，公にできません。その点ご注意ください")
            #そのため Google Colab での解法，ローカルファイルよりアップロードする
            #from google.colab import files
            #uploaded = files.upload()  # ここで `日本語オノマトペ辞典4500より.xls` を指定してアップロードする
            ccap_base = '/Users/asakawa/study/2021ccap/notebooks'
            #onomatopea_excel = '日本語オノマトペ辞典4500より.xlsx'  # オリジナルファイル名，次行は勝手に rename したファイル名
            onomatopea_excel = '2021-0325日本語オノマトペ辞典4500より.xls'
            onmtp2761 = pd.read_excel(os.path.join(ccap_base, onomatopea_excel), sheet_name='2761語')

            # すべてカタカナ表記にしてデータとして利用する
            # onomatopea = list(sorted(set([jaconv.hira2kata(o) for o in onmtp2761['オノマトペ']])))
            onomatopea_vocab = list(sorted(set([jaconv.normalize(o) for o in onmtp2761['オノマトペ']])))
            #onomatopea_k = list(sorted(set([jaconv.hira2kata(w) for w in onomatopea])))
            #onomatopea = (onomatopea + onomatopea_k)
            #print(f'データファイル名: {os.path.join(ccap_base, onomatopea_excel)}\n',
            #      f'オノマトペ総数: {len(onomatopea_vocab)}')
            vocab = onomatopea_vocab
        
        for w in vocab:  # 'ぎらｋつ' ID=1537 は，'ぎらっ' が本当。なので入れ替える
            if 'k' in w:
                vocab[vocab.index(w)] = 'ぎらっ' 
        self.vocab = vocab

        self.orthography = ['<SOW>', '<EOW>', '<UNK>', '<PAD>'] + list(sorted(set([ch for w in self.vocab for ch in w])))
        self.grapheme = self.orthography
        self.phonology = ['<SOW>', '<EOW>', '<UNK>', '<PAD>'] + list(sorted(set([p for w in self.vocab for p in hiragana2julius(w).split(' ')])))
        self.phoneme = self.phonology

        data = {w:{'katakana': jaconv.hira2kata(w)} for w in self.vocab}
        orth_max_length, phon_max_length = 0, 0
        for w in self.vocab:
            data[w]['orth'] = [ch for ch in w]
            data[w]['orth_ids'] = [self.orthography.index(g) for g in data[w]['orth']]
            o_len = len(data[w]['orth_ids'])
            orth_max_length = o_len if o_len > orth_max_length else orth_max_length
            
            data[w]['phon'] = hiragana2julius(w).split(' ')
            data[w]['phon_ids'] = [self.phonology.index(p) for p in data[w]['phon']]
            p_len = len(data[w]['phon_ids'])
            phon_max_length = p_len if p_len > phon_max_length else phon_max_length
        self.data = data
        self.orth_max_length = orth_max_length
        self.phon_max_length = phon_max_length

        phoneme_freq = {}
        for word in self.vocab:
            for _phon in self.data[word]['phon']:
                if not _phon in phoneme_freq:
                    phoneme_freq[_phon] = 1
                else:
                    phoneme_freq[_phon] += 1
        self.phoneme_freq = phoneme_freq
        self.phonology_freq = phoneme_freq

        grapheme_freq = {}
        for word in self.vocab:
            for char in self.data[word]['orth']:
                if char in grapheme_freq:
                    grapheme_freq[char] += 1
                else:
                    grapheme_freq[char] = 1
        self.grapheme_freq = grapheme_freq
        self.orthography_freq = self.grapheme_freq


    def tokenize(self, inputs:str) -> dict:
        ret = {}
        if isinstance(inputs, str) and (inputs in self.vocab):
            ret['input_ids'] = self.data[inputs]['orth_ids'] + [self.orthography.index('<EOW>')]
            ret['teach_ids'] = self.data[inputs]['phon_ids'] + [self.phonology.index('<EOW>')]
            return ret
        else:
            chars = [ch for ch in inputs]
            ret['inputs_ids'] = [self.orthography.index(ch) if ch in self.orthography else self.orthography.index('<UNK>') for ch in chars] + [self.orthography.index('<EOW>')]

            _chars = hiragana2julius(jaconv.kata2hira(inputs)).split(' ')
            ret['teach_ids'] = [self.phonology.index(ch) if ch in self.phonology else self.phonology.index('<UNK>') for ch in _chars]  + [self.phonology.index('<EOW>')]
            return ret



    def graph_ids2wrd(self, graph_ids:list)->list:
        ret = " ".join(self.grapheme[idx] for idx in graph_ids)
        return ret
    
    def phon_ids2wrd(self, phon_ids:list)->list:
        ret = " ".join(self.phoneme[idx] for idx in phon_ids)
        return ret
        
    def __len__(self)->int:
        return len(self.vocab)
    
    def __call__(self, args:list=None)->dict:
        if args == None:
            return self.vocab
        
        if isinstance(args, list):
            ret = []
            for arg in args:
                if isinstance(arg, str):
                    ret.append(self.data[arg])
                elif isinstance(arg, int):
                    word = self.vocab[arg]
                    ret.append(self.data[word])
            return ret
        else:
            if isinstance(args, str):
                return self.data[args]
            elif isinstance(args, int):
                word = self.vocab[args]
                return self.data[word]
    

    def draw_phoneme_freq(self, 
                          save_fname=None, 
                          figsize=(18,8),
                          fontsize=16,
                         )->None:
        plt.figure(figsize=figsize)
        plt.plot(np.array(sorted(self.phoneme_freq.values())[::-1]))
        plt.title('小野「オノマトペ辞典4500」項目に現れる全オノマトペを jaconv.julius で音に変換した場合の音素頻度', fontsize=fontsize)
        plt.xticks(np.arange(len(self.phoneme_freq)), [self.phoneme[self.phoneme.index(k)] for k, _ in self.phoneme_freq.items()], fontsize=fontsize)
        plt.xlabel('音素 頻度順に並べ替え', fontsize=fontsize)
        plt.ylabel('頻度', fontsize=fontsize)
        if save_fname != None:
            plt.savefig(save_fname)
            
    def draw_grapheme_freq(self,
                           save_fname=None,
                           figsize=(18,8),
                           fontsize=12,
                          )->None:
        plt.figure(figsize=figsize)
        plt.plot(np.array(sorted(self.grapheme_freq.values())[::-1]))
        plt.title('小野「オノマトペ辞典4500」項目に現れる全オノマトペの文字(書記素)の出現頻度', fontsize=fontsize)
        plt.xticks(np.arange(len(self.grapheme_freq)), [self.grapheme[self.grapheme.index(k)] for k, _ in self.grapheme_freq.items()], fontsize=fontsize)
        plt.xlabel('頻度順に並べ替え', fontsize=fontsize)
        plt.ylabel('頻度', fontsize=fontsize)
        if save_fname != None:
            plt.savefig(save_fname)


import re

# from jaconv/conv_table.py
JULIUS_LONG_VOWEL = tuple(
    (
        (re.compile('( a){2,}'), ' a:'),
        (re.compile('( i){2,}'), ' i:'),
        (re.compile('( u){2,}'), ' u:'),
        (re.compile('( e){2,}'), ' e:'),
        (re.compile('( o){2,}'), ' o:')
    )
)

# from jaconv/jaconv.py
def hiragana2julius(text):
    """Convert Hiragana to Julius's phoneme format.

    Parameters
    ----------
    text : str
        Hiragana string.

    Return
    ------
    str
        Alphabet string.

    Examples
    --------
    >>> print(jaconv.hiragana2julius('てんきすごくいいいいいい'))
    t e N k i s u g o k u i:
    """

    # 3文字以上からなる変換規則
    text = text.replace('う゛ぁ', ' b a')
    text = text.replace('う゛ぃ', ' b i')
    text = text.replace('う゛ぇ', ' b e')
    text = text.replace('う゛ぉ', ' b o')
    text = text.replace('う゛ゅ', ' by u')

    # 2文字からなる変換規則
    text = text.replace('ぅ゛', ' b u')

    text = text.replace('あぁ', ' a a')
    text = text.replace('いぃ', ' i i')
    text = text.replace('いぇ', ' i e')
    text = text.replace('いゃ', ' y a')
    text = text.replace('うぅ', ' u:')
    text = text.replace('えぇ', ' e e')
    text = text.replace('おぉ', ' o:')
    text = text.replace('かぁ', ' k a:')
    text = text.replace('きぃ', ' k i:')
    text = text.replace('くぅ', ' k u:')
    text = text.replace('くゃ', ' ky a')
    text = text.replace('くゅ', ' ky u')
    text = text.replace('くょ', ' ky o')
    text = text.replace('けぇ', ' k e:')
    text = text.replace('こぉ', ' k o:')
    text = text.replace('がぁ', ' g a:')
    text = text.replace('ぎぃ', ' g i:')
    text = text.replace('ぐぅ', ' g u:')
    text = text.replace('ぐゃ', ' gy a')
    text = text.replace('ぐゅ', ' gy u')
    text = text.replace('ぐょ', ' gy o')
    text = text.replace('げぇ', ' g e:')
    text = text.replace('ごぉ', ' g o:')
    text = text.replace('さぁ', ' s a:')
    text = text.replace('しぃ', ' sh i:')
    text = text.replace('すぅ', ' s u:')
    text = text.replace('すゃ', ' sh a')
    text = text.replace('すゅ', ' sh u')
    text = text.replace('すょ', ' sh o')
    text = text.replace('せぇ', ' s e:')
    text = text.replace('そぉ', ' s o:')
    text = text.replace('ざぁ', ' z a:')
    text = text.replace('じぃ', ' j i:')
    text = text.replace('ずぅ', ' z u:')
    text = text.replace('ずゃ', ' zy a')
    text = text.replace('ずゅ', ' zy u')
    text = text.replace('ずょ', ' zy o')
    text = text.replace('ぜぇ', ' z e:')
    text = text.replace('ぞぉ', ' z o:')
    text = text.replace('たぁ', ' t a:')
    text = text.replace('ちぃ', ' ch i:')
    text = text.replace('つぁ', ' ts a')
    text = text.replace('つぃ', ' ts i')
    text = text.replace('つぅ', ' ts u:')
    text = text.replace('つゃ', ' ch a')
    text = text.replace('つゅ', ' ch u')
    text = text.replace('つょ', ' ch o')
    text = text.replace('つぇ', ' ts e')
    text = text.replace('つぉ', ' ts o')
    text = text.replace('てぇ', ' t e:')
    text = text.replace('とぉ', ' t o:')
    text = text.replace('だぁ', ' d a:')
    text = text.replace('ぢぃ', ' j i:')
    text = text.replace('づぅ', ' d u:')
    text = text.replace('づゃ', ' zy a')
    text = text.replace('づゅ', ' zy u')
    text = text.replace('づょ', ' zy o')
    text = text.replace('でぇ', ' d e:')
    text = text.replace('どぉ', ' d o:')
    text = text.replace('なぁ', ' n a:')
    text = text.replace('にぃ', ' n i:')
    text = text.replace('ぬぅ', ' n u:')
    text = text.replace('ぬゃ', ' ny a')
    text = text.replace('ぬゅ', ' ny u')
    text = text.replace('ぬょ', ' ny o')
    text = text.replace('ねぇ', ' n e:')
    text = text.replace('のぉ', ' n o:')
    text = text.replace('はぁ', ' h a:')
    text = text.replace('ひぃ', ' h i:')
    text = text.replace('ふぅ', ' f u:')
    text = text.replace('ふゃ', ' hy a')
    text = text.replace('ふゅ', ' hy u')
    text = text.replace('ふょ', ' hy o')
    text = text.replace('へぇ', ' h e:')
    text = text.replace('ほぉ', ' h o:')
    text = text.replace('ばぁ', ' b a:')
    text = text.replace('びぃ', ' b i:')
    text = text.replace('ぶぅ', ' b u:')
    text = text.replace('ふゃ', ' hy a')
    text = text.replace('ぶゅ', ' by u')
    text = text.replace('ふょ', ' hy o')
    text = text.replace('べぇ', ' b e:')
    text = text.replace('ぼぉ', ' b o:')
    text = text.replace('ぱぁ', ' p a:')
    text = text.replace('ぴぃ', ' p i:')
    text = text.replace('ぷぅ', ' p u:')
    text = text.replace('ぷゃ', ' py a')
    text = text.replace('ぷゅ', ' py u')
    text = text.replace('ぷょ', ' py o')
    text = text.replace('ぺぇ', ' p e:')
    text = text.replace('ぽぉ', ' p o:')
    text = text.replace('まぁ', ' m a:')
    text = text.replace('みぃ', ' m i:')
    text = text.replace('むぅ', ' m u:')
    text = text.replace('むゃ', ' my a')
    text = text.replace('むゅ', ' my u')
    text = text.replace('むょ', ' my o')
    text = text.replace('めぇ', ' m e:')
    text = text.replace('もぉ', ' m o:')
    text = text.replace('やぁ', ' y a:')
    text = text.replace('ゆぅ', ' y u:')
    text = text.replace('ゆゃ', ' y a:')
    text = text.replace('ゆゅ', ' y u:')
    text = text.replace('ゆょ', ' y o:')
    text = text.replace('よぉ', ' y o:')
    text = text.replace('らぁ', ' r a:')
    text = text.replace('りぃ', ' r i:')
    text = text.replace('るぅ', ' r u:')
    text = text.replace('るゃ', ' ry a')
    text = text.replace('るゅ', ' ry u')
    text = text.replace('るょ', ' ry o')
    text = text.replace('れぇ', ' r e:')
    text = text.replace('ろぉ', ' r o:')
    text = text.replace('わぁ', ' w a:')
    text = text.replace('をぉ', ' o:')

    text = text.replace('う゛', ' b u')
    text = text.replace('でぃ', ' d i')
    text = text.replace('でぇ', ' d e:')
    text = text.replace('でゃ', ' dy a')
    text = text.replace('でゅ', ' dy u')
    text = text.replace('でょ', ' dy o')
    text = text.replace('てぃ', ' t i')
    text = text.replace('てぇ', ' t e:')
    text = text.replace('てゃ', ' ty a')
    text = text.replace('てゅ', ' ty u')
    text = text.replace('てょ', ' ty o')
    text = text.replace('すぃ', ' s i')
    text = text.replace('ずぁ', ' z u a')
    text = text.replace('ずぃ', ' z i')
    text = text.replace('ずぅ', ' z u')
    text = text.replace('ずゃ', ' zy a')
    text = text.replace('ずゅ', ' zy u')
    text = text.replace('ずょ', ' zy o')
    text = text.replace('ずぇ', ' z e')
    text = text.replace('ずぉ', ' z o')
    text = text.replace('きゃ', ' ky a')
    text = text.replace('きゅ', ' ky u')
    text = text.replace('きょ', ' ky o')
    text = text.replace('しゃ', ' sh a')
    text = text.replace('しゅ', ' sh u')
    text = text.replace('しぇ', ' sh e')
    text = text.replace('しょ', ' sh o')
    text = text.replace('ちゃ', ' ch a')
    text = text.replace('ちゅ', ' ch u')
    text = text.replace('ちぇ', ' ch e')
    text = text.replace('ちょ', ' ch o')
    text = text.replace('とぅ', ' t u')
    text = text.replace('とゃ', ' ty a')
    text = text.replace('とゅ', ' ty u')
    text = text.replace('とょ', ' ty o')
    text = text.replace('どぁ', ' d o a')
    text = text.replace('どぅ', ' d u')
    text = text.replace('どゃ', ' dy a')
    text = text.replace('どゅ', ' dy u')
    text = text.replace('どょ', ' dy o')
    text = text.replace('どぉ', ' d o:')
    text = text.replace('にゃ', ' ny a')
    text = text.replace('にゅ', ' ny u')
    text = text.replace('にょ', ' ny o')
    text = text.replace('ひゃ', ' hy a')
    text = text.replace('ひゅ', ' hy u')
    text = text.replace('ひょ', ' hy o')
    text = text.replace('みゃ', ' my a')
    text = text.replace('みゅ', ' my u')
    text = text.replace('みょ', ' my o')
    text = text.replace('りゃ', ' ry a')
    text = text.replace('りゅ', ' ry u')
    text = text.replace('りょ', ' ry o')
    text = text.replace('ぎゃ', ' gy a')
    text = text.replace('ぎゅ', ' gy u')
    text = text.replace('ぎょ', ' gy o')
    text = text.replace('ぢぇ', ' j e')
    text = text.replace('ぢゃ', ' j a')
    text = text.replace('ぢゅ', ' j u')
    text = text.replace('ぢょ', ' j o')
    text = text.replace('じぇ', ' j e')
    text = text.replace('じゃ', ' j a')
    text = text.replace('じゅ', ' j u')
    text = text.replace('じょ', ' j o')
    text = text.replace('びゃ', ' by a')
    text = text.replace('びゅ', ' by u')
    text = text.replace('びょ', ' by o')
    text = text.replace('ぴゃ', ' py a')
    text = text.replace('ぴゅ', ' py u')
    text = text.replace('ぴょ', ' py o')
    text = text.replace('うぁ', ' u a')
    text = text.replace('うぃ', ' w i')
    text = text.replace('うぇ', ' w e')
    text = text.replace('うぉ', ' w o')
    text = text.replace('ふぁ', ' f a')
    text = text.replace('ふぃ', ' f i')
    text = text.replace('ふぅ', ' f u')
    text = text.replace('ふゃ', ' hy a')
    text = text.replace('ふゅ', ' hy u')
    text = text.replace('ふょ', ' hy o')
    text = text.replace('ふぇ', ' f e')
    text = text.replace('ふぉ', ' f o')

    # 1音からなる変換規則
    text = text.replace('あ', ' a')
    text = text.replace('い', ' i')
    text = text.replace('う', ' u')
    text = text.replace('え', ' e')
    text = text.replace('お', ' o')
    text = text.replace('か', ' k a')
    text = text.replace('き', ' k i')
    text = text.replace('く', ' k u')
    text = text.replace('け', ' k e')
    text = text.replace('こ', ' k o')
    text = text.replace('さ', ' s a')
    text = text.replace('し', ' sh i')
    text = text.replace('す', ' s u')
    text = text.replace('せ', ' s e')
    text = text.replace('そ', ' s o')
    text = text.replace('た', ' t a')
    text = text.replace('ち', ' ch i')
    text = text.replace('つ', ' ts u')
    text = text.replace('て', ' t e')
    text = text.replace('と', ' t o')
    text = text.replace('な', ' n a')
    text = text.replace('に', ' n i')
    text = text.replace('ぬ', ' n u')
    text = text.replace('ね', ' n e')
    text = text.replace('の', ' n o')
    text = text.replace('は', ' h a')
    text = text.replace('ひ', ' h i')
    text = text.replace('ふ', ' f u')
    text = text.replace('へ', ' h e')
    text = text.replace('ほ', ' h o')
    text = text.replace('ま', ' m a')
    text = text.replace('み', ' m i')
    text = text.replace('む', ' m u')
    text = text.replace('め', ' m e')
    text = text.replace('も', ' m o')
    text = text.replace('ら', ' r a')
    text = text.replace('り', ' r i')
    text = text.replace('る', ' r u')
    text = text.replace('れ', ' r e')
    text = text.replace('ろ', ' r o')
    text = text.replace('が', ' g a')
    text = text.replace('ぎ', ' g i')
    text = text.replace('ぐ', ' g u')
    text = text.replace('げ', ' g e')
    text = text.replace('ご', ' g o')
    text = text.replace('ざ', ' z a')
    text = text.replace('じ', ' j i')
    text = text.replace('ず', ' z u')
    text = text.replace('ぜ', ' z e')
    text = text.replace('ぞ', ' z o')
    text = text.replace('だ', ' d a')
    text = text.replace('ぢ', ' j i')
    text = text.replace('づ', ' z u')
    text = text.replace('で', ' d e')
    text = text.replace('ど', ' d o')
    text = text.replace('ば', ' b a')
    text = text.replace('び', ' b i')
    text = text.replace('ぶ', ' b u')
    text = text.replace('べ', ' b e')
    text = text.replace('ぼ', ' b o')
    text = text.replace('ぱ', ' p a')
    text = text.replace('ぴ', ' p i')
    text = text.replace('ぷ', ' p u')
    text = text.replace('ぺ', ' p e')
    text = text.replace('ぽ', ' p o')
    text = text.replace('や', ' y a')
    text = text.replace('ゆ', ' y u')
    text = text.replace('よ', ' y o')
    text = text.replace('わ', ' w a')
    text = text.replace('ゐ', ' i')
    text = text.replace('ゑ', ' e')
    text = text.replace('ん', ' N')
    text = text.replace('っ', ' q')
    # ここまでに処理されてない ぁぃぅぇぉ はそのまま大文字扱い
    text = text.replace('ぁ', ' a')
    text = text.replace('ぃ', ' i')
    text = text.replace('ぅ', ' u')
    text = text.replace('ぇ', ' e')
    text = text.replace('ぉ', ' o')
    text = text.replace('ゎ', ' w a')
    text = text.replace('ぉ', ' o')

    # 長音の処理
    for (pattern, replace_str) in JULIUS_LONG_VOWEL:
        text = pattern.sub(replace_str, text)
    text = text.replace('o u', 'o:')  # おう -> おーの音便
    text = text.replace('ー', ':')
    text = text.replace('〜', ':')
    text = text.replace('−', ':')
    text = text.replace('-', ':')


    #その他特別な処理
    text = text.replace('を', ' o')

    text = text.strip()

    text = text.replace(':+', ':')
    return text
            
