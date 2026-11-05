import os
import urllib.request

URL = ("https://raw.githubusercontent.com/ARY2260/openpom/main/"
       "openpom/data/curated_datasets/curated_GS_LF_merged_4983.csv")
SAVE_AS = "data/molecules.csv"

os.makedirs("data", exist_ok=True)   # create the data folder if it's missing

print("Downloading...")
urllib.request.urlretrieve(URL, SAVE_AS)
print("Saved to", SAVE_AS)