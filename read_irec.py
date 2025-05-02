import pandas as pd

wd = "/home/gmor/Nextcloud2/Beegroup/data/hypercadaster_ES"
cadaster_codes = ["08900"]

df = pd.read_pickle(f"{wd}/results/IREC_bcn_input.pkl")
df[["BuildingReference","YearOfConstruction"]].sort_values("YearOfConstruction").head(50)
