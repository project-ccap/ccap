import urllib.request
#import torchvision.datasets.imagenet
from nltk.corpus import wordnet as wn
import os
import numpy as np
import sys
import pandas as pd
import json
import glob
import PIL.Image as PILImage

import matplotlib.pyplot as plt

# グラフ中の日本語がトーフになるので，その対策
import matplotlib.font_manager

# Windows, Linux の場合には適当に書き換える必要がある
fontprop = matplotlib.font_manager.FontProperties(
    fname="/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc")


class snodgrassDataset():
    """
    Snodgrass and Vanderwart (1980) 画像管理
    ---------------------------------------

    Nishimoto et al. (2005) からデータを入手 済であることが前提

    - papers:
    - Joan Gay Snodgrass and Mary Vanderwart (1980) A Standardized Set of 260 Pictures:
    Norms for Name Agreement, Image Agreement, Familiarity, and Visual Complexity. Journal
    of Experimental Psychology: Human Learning and Memory Vol. 6, No. 2, 174-215.
    - The role of imagery-related properties in picture naming: A newly standardized
    set of 360 pictures for Japanese, Nishimoto et al., (2012) Behavior Research Method.
    44:934–945.
    - Japanese normative set of 359 pictures, Nishimoto et al., (2005) Behavior Research
    Method 37(3) 398-416.

    - url: `Nishimoto-BRM-2005.zip`, http://www.psychonomic.org/archive/ and look for the
    journal (Behavior Research Methods), the first author’s name (Nishimoto), and the
    publication year (2005).

    Returns:
    --------
        - __call__(etnry): entry(int) を与えると，画像ファイル名とラベル名を返す

    """

    def __init__(self, root=None, download=False):

        self.snodgrass_base = 'ccap/data'
        self.image_dir = 'SnodgrassAndVanderwart'

        # Nishimoto et. al (2005) のエクセルデータ読み込み
        self.excel_filename = os.path.join(
            self.snodgrass_base, 'Nishimoto-BRM-2005/appendix_a.xls')
        self.pd = pd.read_excel(self.excel_filename,
                                names=['No', 'Type', 'Set', 'Kana', 'Romaji', 'English',
                                       'SV', 'RTst', 'RTlib', 'H', 'NAst', 'NAlib',
                                       'AoA', 'FAM', 'mora'],
                                skiprows=[0])

        # オリジナル画像だけを抽出
        self.SV = self.pd[self.pd['Type'] != 'A']
        _a = self.SV['SV'].to_list()
        _a = [int(_x) for _x in _a]
        self.SVen = self.SV['English'].to_list()
        self.SVja = self.SV['Kana'].to_list()

        self.SVen = [x.lower() for x in self.SV['English']]

        self.en2ja = {e: j for e, j in zip(self.SVen, self.SVja)}
        self.ja2en = {j: e for e, j in zip(self.SVen, self.SVja)}
        self.en2no = {self.SV.iloc[i]['English']: int(
            self.SV.iloc[i]['SV']) for i in range(len(self.SV))}
        self.no2en = {n: e for e, n in self.en2no.items()}
        self.ja2no = {self.SV.iloc[i]['Kana']: int(
            self.SV.iloc[i]['SV']) for i in range(len(self.SV))}
        self.no2ja = {n: j for j, n in self.ja2no.items()}
        self.labels_ja = self.SVja
        self.labels = self.SVen

        self.data = {}
        for i in range(len(self.SV)):
            w_e = self.SV.iloc[i]['English']
            w_j = self.SV.iloc[i]['Kana']

            img_no = int(self.SV.iloc[i]['SV'])
            img_file = os.path.join(
                self.snodgrass_base, self.image_dir, '{0:03d}.png'.format(img_no))
            wn_id = w_e.replace(' ', '_').lower()
            self.data[i] = {
                'img_no': img_no,
                'label_ja': w_j,
                'label': w_e.lower(),
                'img': img_file,
                'WordNet': self.WordNet2ent(wn_id)
            }

        # self.WordNetDict = {}
        # for word in self.SVen:
        #    img_no = self.en2no[word]
        #    word_ = word.replace(' ', '_').lower()
        #    self.WordNetDict[img_no] = self.WordNet2ent(word_)

    # WordNet 情報の付加

    def WordNet2ent(self, word):
        synsets = wn.synsets(word, pos='n')
        if len(synsets) == 0:
            return None
        else:
            return {'id': word,
                    'synset': [s_.name() for s_ in synsets],
                    # 'label': [wn.lemma(s_) for s_ in synsets],
                    # 'label_ja': [wn.lemma(s_, lang='jpn') for lemma in synset.lemmas('jpn')],
                    'definition': [s_.definition() for s_ in synsets],
                    'lexname': [s_.lexname() for s_ in synsets]
                    }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, entry):
        return self.data[entry]['img'], self.data[entry]['label']

    def __call__(self, entry, lang='eng'):
        return self.no_or_label(entry)

    def no_or_label(self, ent, lang='jpn'):
        if isinstance(ent, int):
            if (ent < 0) or (ent > self.__len__()):
                return False
            else:
                ret = self.data[ent]
                label = ret['label'] if lang == 'jpn' else ret['label']
                return ret['img'], label

    def show_an_image(self, entry):
        """display an image(entry)"""
        img_file, label = self.no_or_label(entry)
        if not img_file:
            return None
        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{0}'.format(label), fontdict={
                     "fontproperties": fontprop}, fontsize=14)
        ax.set_axis_off()
        img = PILImage.open(img_file)
        if img.mode == 'L':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    def show_all_images(self, cols=6, lang='jpn'):
        """display all the images"""
        rows = int(self.__len__() / cols + 1)

        fig = plt.figure(figsize=(12, 2 * rows))

        i = 1
        for no in range(self.__len__()):
            img_file, label = self.no_or_label(no + 1, lang=lang)
            ax = fig.add_subplot(rows, cols, no+1)  # 縦，横，通し番号

            ax.set_title('{0}'.format(label), fontdict={
                "fontproperties": fontprop}, fontsize=14)
            ax.set_axis_off()
            img = PILImage.open(img_file)
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title('{0} {1}'.format(no, label), fontdict={
                         "fontproperties": fontprop}, fontsize=14)
            ax.axis(False)


imagenet_base = '/Users/asakawa/study/data/ImageNet/'
ImangeNet_base = imagenet_base


class imagenetDataset():
    """
    - ImageNet のデータセット管理
    - note: ImageNet のデータは登録しないと使えない. 従って `meta.bin` が入手済であることは
    仮定できない。

    getitem_from_no(no):
        inputs: no(int) は，WordNet の offset 順にソートされている。多くの実装に従っている。
        returns:

    前準備として ImageNet の訓練データセットのスーパータイニーサブセットをダウンロードする
    `https://drive.google.com/file/d/1WR1U3q5FwnD8L3qik4PyuzjZWtTho2UX/view?usp=sharing`
    """

    def __init__(self, bin=None):

        import torch

        #self.ImageNet_base = '/Users/asakawa/study/data/ImageNet'
        self.ImageNet_base = 'ccap/data/ImageNet'
        # `meta.bin` は登録していない限り使えない。だから公共の場所に置くことができない。どうする？
        #self.meta = torchvision.datasets.imagenet.load_meta_file(
        #    root=self.ImageNet_base, file='meta.bin')
        #self.meta = torch.load(os.path.join(self.ImageNet_base, 'imagenet2012_meta.bin'))
        self.meta = torch.load(os.path.join(self.ImageNet_base, 'meta.bin'))
        self.ImageNet_wnids = list(sorted(self.meta[0].keys()))
        self.wnid2no = {wnid: i for i,
                        wnid in enumerate(self.ImageNet_wnids)}
        self.no2wnid = {no: wnid for wnid, no in self.wnid2no.items()}
        self.ImageNet_WN = {}
        self.data = {}
        for i, wnid in enumerate(self.ImageNet_wnids):
            self.ImageNet_WN[i] = self.WordNetID2ent(wnid)
            self.data[i] = self.ImageNet_WN[i]
        self.labels = [self.WordNetID2ent(wnid)['label']
                       for wnid in self.ImageNet_wnids]
        self.label2no = {}
        for i, labels_ in enumerate(self.labels):
            for label in labels_:
                self.label2no[label] = i

    def __len__(self):
        return len(self.ImageNet_wnids)

    def WordNetID2ent(self, wn_id):
        synset = wn.synset_from_pos_and_offset(wn_id[0], int(wn_id[1:]))
        return {'id': wn_id,
                'synset': synset.name(),
                'label': [str(lemma.name()) for lemma in synset.lemmas()],
                'label_ja': [str(lemma.name()) for lemma in synset.lemmas('jpn')],
                'definition': synset.definition(),
                'lexname': synset.lexname()
                }

    def __call__(self, ent):
        if isinstance(ent, int):
            img = self.sample_image(ent)
            label = self.ImageNet_WN[ent]['id']
            return img, label
        else:
            ent = self.wnid2no[ent]
            img = self.sample_image(ent)
            label = self.ImageNet_WN[ent]['id']
            return img, label

    def getitem_from_no(self, no):
        return self.ImageNet_WN[no]

    def getitem_from_wnid(self, wn_id):
        return self.ImageNet_WN[self.wnid2no[wn_id]]

    def sample_image(self, ent):
        """ent に対応する画像をサンプリングして返す"""
        if isinstance(ent, int):
            if (0 <= ent) and (ent <= self.__len__()):
                filenames = glob.glob(os.path.join(self.ImageNet_base, 'train',
                                                   self.ImageNet_wnids[ent]+'/*.JPEG'))
                return np.random.choice(filenames)
            else:
                return None
        else:
            if ent in self.ImageNet_wnids:
                filenames = glob.glob(os.path.join(
                    self.ImageNet_base, 'train', ent+'/*.JPEG'))
                return np.random.choice(filenames)
            else:
                return None

    def sample_and_show(self, entry):
        img_file, label = self.__call__(entry)
        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)

        label_ = self.WordNetID2ent(label)['label_ja']
        if label_:
            label = label_
        else:
            label = self.WordNetID2ent(label)['label']
        ax.set_title('{0}'.format(label), fontdict={
                     "fontproperties": fontprop}, fontsize=14)
        ax.set_axis_off()
        img = PILImage.open(img_file)
        if img.mode == 'L':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)


#SALA_base = '/Users/asakawa/study/2020ccap'
SALA_image_dir = 'sala_imgs'
#SALA_anno_file = '2020SALA_data_tmp.txt'
SALA_anno_file = 'sala_data.txt'


class salaDataset():
    """
    - 上智大学 SALA プロジェクト，藤林眞理子著
    - PR20 と PR24 の画像全 186 画像。
    - ただし，PR24-22 呼称II (やかん) の画像が欠如しているため，全185枚の画像
    """

    def __init__(self):
        self.base = 'ccap/data'
        self.img_dir = 'sala_imgs'
        self.anno_file = 'sala_data.txt'
        with open(os.path.join(self.base, self.anno_file), 'r') as f:
            a = f.readlines()
        data_ = [a_.strip().split(' ') for a_ in a]

        self.data = {}
        for i, d in enumerate(data_):
            self.data[i] = {'img': d[0], 'Freq': d[1], 'label': d[2:]}

        self.label2no, self.no2label = {}, {}
        self.labels = []
        self.elabels = []
        for i in range(len(self.data)):
            ent = self.data[i]
            # 'img' をフルパスで置き換える
            filename = os.path.join(self.base, self.img_dir, ent['img']+'.jpg')

            # ファイルが存在しなかったらデータを削除
            if not os.path.exists(filename):
                del(self.data[i])
                continue
            else:
                # 存在したら書き換え
                ent['img'] = filename

            # 正解ラベルを作成: 後ろにあるデータが読み
            if len(ent['label']) > 1:
                ent['label'] = ent['label'][-1]
            #    ent['label'] = a[0]
            #    ent['yomi'] = a[1:]

            if isinstance(ent['label'], list):
                _label = ent['label'][0]
            else:
                _label = ent['label']
            if '（' in _label:
                d = _label.split('（')
                ent['label'] = d[0]
                ent['yomi'] = d[1].replace('）', '')
            else:
                ent['label'] = _label
                ent['yomi'] = _label

            # 検索容易性のためにラベルから番号を，番号からラベルを検索可能にする
            self.label2no[ent['label']] = i
            self.no2label[i] = ent['label']

            # ラベルのリストに追加
            self.labels.append(ent['label'])

            # WordNet 情報の付加
            wn_ids = wn.synsets(ent['label'], lang='jpn')
            ent['WordNet_id'] = [s.name() for s in wn_ids]
            ent['e_labels'] = [wnid.split('.')[0]
                               for wnid in ent['WordNet_id']]
            ent['definition'] = [s.definition() for s in wn_ids]
            ent['lexname'] = [s.lexname() for s in wn_ids]

        # self.labels = set(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem(self, entry):
        """画像とラベルを返す"""
        return self.data[entry]['img'], self.data[entry]['label']

    def __call__(self, ent):
        """ent が str か int かによって動作を変える"""

        if isinstance(ent, int):
            if (0 <= ent) and (ent < len(self.data)):
                img = self.data[ent]['img']
                label = self.data[ent]['label']
                return img, label
            else:
                return None
        if ent in self.labels:
            ent_ = self.data[self.label2no[ent]]
            img = ent_['img']
            label = ent_['label']
            return img, label
        else:
            return None

    def __iter__(self):
        self.n = 0
        return self  # .data[self.n]  # (self.n)

    def __next__(self):
        if self.n < len(self.data):
            self.n += 1
            return self(self.n)  # (self.n)
        else:
            raise StopIteration

    def show_an_image(self, entry):
        if isinstance(entry, int):
            ret = self.data[entry]
        else:
            ret = self.data[self.label2no[entry]]
        img_file = ret['img']
        label = ret['label']

        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{0}'.format(label), fontdict={
                     "fontproperties": fontprop}, fontsize=14)
        ax.set_axis_off()
        img = PILImage.open(img_file)
        if img.mode == 'L':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    def show_all_images(self, cols=6):
        """display all the images"""
        rows = int(self.__len__() / cols + 1)

        fig = plt.figure(figsize=(12, 2 * rows))

        i = 1
        for no in range(self.__len__()):
            img_file, label = self.data[no]['img'], self.data[no]['label']
            ax = fig.add_subplot(rows, cols, no+1)  # 縦，横，通し番号

            ax.set_title('{0}'.format(label), fontdict={
                "fontproperties": fontprop}, fontsize=14)
            ax.set_axis_off()
            img = PILImage.open(img_file)
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title('{0} {1}'.format(no, label), fontdict={
                         "fontproperties": fontprop}, fontsize=14)
            ax.axis(False)


class tlpaDataset():
    """TLPA
    藤田 他 (2000) 「失語症検査の開発」 音声言語医学, 41(2), 179-202.
    """

    def __init__(self):
        self.tlpa_base = 'ccap/data'
        self.img_dir = 'tlpa_images'
        self.tlpa_json = 'tlpa_data.json'

        self.tlpa_ = json.load(
            open(os.path.join(self.tlpa_base, self.tlpa_json), 'r'))
        del(self.tlpa_['Description'])  # 最後の json エントリは削除

        self.tlpa_bnc = {}
        i = 0
        self.labels = []
        for ent in self.tlpa_:

            # 色カードは除外
            if self.tlpa_[ent]['Cat'] != 'C':  # 色は除外
                self.tlpa_bnc[i] = self.tlpa_[ent]
                filename = os.path.join(self.tlpa_base,
                                        self.img_dir,
                                        '{0:03d}'.format(int(ent))+'.JPG')
                self.tlpa_bnc[i]['img'] = filename
                self.tlpa_bnc[i]['label'] = self.tlpa_[ent]['Name']
                self.labels.append(self.tlpa_bnc[i]['label'])
                i = i + 1
        self.data = self.tlpa_bnc
        self.label2index = {self.data[ent]['Name']: ent for ent in self.data}
        self.index2label = {ent: self.data[ent]['Name'] for ent in self.data}

    def __len__(self):
        """データサイズを返す"""
        return len(self.data)

    def __getitem__(self, index):
        """画像とラベルを返す"""
        return self.data[index]['img'], self.data[index]['Name']

    def __call__(self, ent):
        """呼び出された際に画像とラベルを返す"""
        if isinstance(ent, int):
            return self.data[ent]['img'], self.data[ent]['label']
        if ent in self.labels:
            index = self.label2index[ent]
            return self.data[index]['img'], self.data[index]['label']

    def show_an_image(self, element):
        img_file, label = self.__call__(element)

        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{0}'.format(label),
                     fontdict={"fontproperties": fontprop},
                     fontsize=14)
        ax.set_axis_off()
        img = PILImage.open(img_file)
        if img.mode == 'L':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    def show_all_images(self, cols=5):
        """display all the images"""
        rows = int(self.__len__() / cols + 1)
        fig = plt.figure(figsize=(12, 2 * rows))

        i = 1
        for no in range(self.__len__()):
            img_file, label = self.__call__(no)
            ax = fig.add_subplot(rows, cols, no+1)  # 縦，横，通し番号
            ax.set_title('{0}'.format(label),
                         fontdict={"fontproperties": fontprop},
                         fontsize=14)
            ax.set_axis_off()
            img = PILImage.open(img_file)
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title('{0} {1}'.format(no, label), fontdict={
                         "fontproperties": fontprop}, fontsize=14)
            # ax.axis(False)


class pntDataset():
    """
    - PNT: The Philadelphi Naming Test,
    - url: https://mrri.org/philadelphia-naming-test
    - GitHub: https://github.com/hanayik/Philadelphia-Naming-Test
    """

    def __init__(self):
        self.stims_url = 'https://raw.githubusercontent.com/hanayik/Philadelphia-Naming-Test/master/assets/stim.csv'
        #self.img_url = '/Users/asakawa/study/2020hanayik_Philadelphia-Naming-Test.git/assets/pics'
        #self.stims_txt = 'stim.csv'
        self.stims_txt = 'pnt_stim.csv'
        #self.stim_ja_txt = '2020pnt_stims_ja.txt'
        self.stim_ja_txt = 'pnt_stim_ja.txt'
        #self.base = '/Users/asakawa/study/2020ccap'
        self.base = 'ccap/data/'
        #self.img_dir = '/Users/asakawa/study/2020hanayik_Philadelphia-Naming-Test.git/assets/pics/'
        self.img_dir = 'pnt_pics'

        self.stim_file = os.path.join(self.base, self.stims_txt)
        if not os.path.exists(self.stim_file):
            self.stim_file = self.stims_url

        # データの入手
        self.pd = pd.read_csv(self.stim_file)

        # データ入手 by urllib
        # req = urllib.request.Request(pnt_url)
        # with urllib.request.urlopen(req) as response:
        #    pnt_csv = response.readlines()
        # pnt_json = pnt_pd.to_json()

        with open(os.path.join(self.base, self.stim_ja_txt), 'r') as f:
            ja_ = f.readlines()
        label_ja = {i: label.strip() for i, label in enumerate(ja_)}

        self.dict = self.pd.to_dict()
        self.pics = self.dict['PictureName']

        self.pics = [self.pics[entry].replace(' ', '') for entry in self.pics]

        self.data = {}
        self.labels = []
        for i in range(len(self.pics)):
            num = self.dict['OrderNum'][i]
            img = os.path.join(self.base, self.img_dir, self.pics[i]+'.png')
            self.data[i] = {'img': img,
                            'label': self.pics[i], 
                            'label_ja': label_ja[i]}
            self.labels.append(self.data[i]['label'])

        #self.labels = label.values()

        self.label2no = {self.data[no]['label']:no for no in range(self.__len__())}
        self.no2label = {no:label for label, no in self.label2no.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]['img'], self.data[index]['label']

    def __call__(self, index):
        if isinstance(index, int):
            return self.__getitem__(index)
        else:
            if index in self.labels:
                index_ = self.label2no[index]
                return self.__getitem__(index_)

    def show_an_image(self, element):
        img_file, label = self.__call__(element)

        fig = plt.figure(figsize=(4, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('{0}'.format(label),
                     fontdict={"fontproperties": fontprop},
                     fontsize=14)
        ax.set_axis_off()
        img = PILImage.open(img_file)
        if img.mode == 'L':
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)

    def show_all_images(self, cols=5):
        """display all the images"""
        rows = int(self.__len__() / cols + 1)
        fig = plt.figure(figsize=(14, 3 * rows))

        i = 1
        for no in range(self.__len__()):
            img_file, label = self.__call__(no)
            ax = fig.add_subplot(rows, cols, no+1)  # 縦，横，通し番号
            ax.set_title('{0}'.format(label),
                         fontdict={"fontproperties": fontprop},
                         fontsize=14)
            ax.set_axis_off()
            img = PILImage.open(img_file)
            if img.mode == 'L':
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.set_title('{0} {1}'.format(no, label), fontdict={
                         "fontproperties": fontprop}, fontsize=14)
            # ax.axis(False)
