# Data

The dataset is not stored in this repository (course rule). Each person downloads it once into this `data/` folder, which is git-ignored.

| | |
|---|---|
| **Dataset** | Medical Transcriptions (MTSamples) |
| **Source** | https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions |
| **Licence** | CC0: Public Domain (as listed on Kaggle) |
| **File** | `mtsamples.csv` (~17 MB, ~5,000 rows) |
| **Access** | Free Kaggle account. No credentialing or data use agreement. |

Expected location after download: `data/mtsamples.csv`

## Option 1: manual download (local)

1. Sign in to Kaggle and open the dataset link above.
2. Click **Download**, unzip, and move `mtsamples.csv` into this `data/` folder.

## Option 2: Kaggle API (Google Colab)

1. On Kaggle: **Settings > API > Create New Token**. This gives you a username and key.
2. In Colab: open the key icon (**Secrets**) and add `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Run from the repo root:

```python
import os
from google.colab import userdata

os.environ["KAGGLE_USERNAME"] = userdata.get("KAGGLE_USERNAME")
os.environ["KAGGLE_KEY"] = userdata.get("KAGGLE_KEY")

!kaggle datasets download -d tboyle10/medicaltranscriptions -p data --unzip
```

**Never commit your Kaggle token** (`kaggle.json` or the key itself). It is git-ignored, but keep it out of notebook cells and outputs too.

## Citation

The custodian does not request a specific citation. We cite it as:

> Boyle T. Medical Transcriptions [dataset]. Kaggle; 2018. Available from: https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions. Data originally scraped from mtsamples.com.
