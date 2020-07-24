# ccap

- ccap is the abbreviation for the Computational Clinical Aphasia Project.
- CCAP was founded the early 2020 following the 2019CNPS that was held at Tokyo Women's Christian university.

# Usage

In order to use ccap on colab, you should do below:

```python
!pip install japanize-matplotlib
import japanize_matplotlib

import nltk
nltk.download('wordnet')
nltk.download('omw')

!git clone https://github.com/project-ccap/ccap.git

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
# This only needs to be done once per notebook.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Download a file based on its file ID.
#
# A file ID looks like: laggVyWshwcyP6kEI-y_W3P8D26sz
#https://drive.google.com/file/d/1LvoGyABdW3n3_EIs7PtntuQQiS2OM4jY/view?usp=sharing for Gdrive cis.twcu.ac.jp/GitHub_shared/ccap_imgs.tgz
file_id = '1LvoGyABdW3n3_EIs7PtntuQQiS2OM4jY'
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('ccap_imgs.tgz')
!tar xzf ccap_imgs.tgz

from ccap import tlpaDataset
tlpa = tlpaDataset()
```

Enjoy!

