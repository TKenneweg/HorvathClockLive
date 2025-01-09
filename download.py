import GEOparse


gse = GEOparse.get_GEO(filepath="./data/GSE41037_family.soft.gz")


# SERIES_NAMES = ["GSE41037", "GSE15745", "GSE27317", "GSE27097", "GSE34035"]

# for series_name in SERIES_NAMES:
#     print("Downloading series", series_name)
#     gse = GEOparse.get_GEO(geo=series_name)
#     print("Done with", series_name)