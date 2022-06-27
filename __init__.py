# -*- coding: utf-8 -*-

__version__ = '0.2'
__author__ = 'Shin Asakawa'
__email__ = 'asakawa@ieee.org'
__license__ = 'MIT'
__copyright__ = 'Copyright 2020 {0}'.format(__author__)


from .ccap import snodgrassDataset
from .ccap import imagenetDataset
from .ccap import salaDataset
from .ccap import tlpaDataset
from .ccap import pntDataset
from . import mecab_settings
#from . import tlpa_o2p # this name seems to be obsolute
from . import multi_modal_dataset


from .ccap_w2v import ccap_w2v

"""
- date: 2020-0718
- memo: WordNet を colab で使用するためには nltk で wordnet をダウンロードする必要がある

```python
import nltk
nltk.download('wordnet')
nltk.download('omw')
from nltk.corpus import wordnet as wn
```

"""
