from config import DATA_DIR
import pandas as pd

import sys
import pip._vendor.pkg_resources as pkg_resources
sys.modules['pkg_resources'] = pkg_resources

from data_profiling import ProfileReport
import os


datafile = 'processed_data.csv'

if datafile not in os.listdir(DATA_DIR):
    sys.exit("ERROR: Processed data file not present in data folder! Run src.data_parser.py")

df = pd.read_csv(DATA_DIR / datafile)

profile = ProfileReport(df, title="Raport CAN")
profile.to_file(DATA_DIR / "report.html")

print("Done! Report saved in data folder.")
