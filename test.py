import pickle
path = "./data/GSE41037/GSM1007129.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)
    print(data)