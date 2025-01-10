import pickle 

pklpath = './data/GSE15745/GSM401538.pkl'

with open(pklpath, 'rb') as f:
    data = pickle.load(f)

print(data["age"])
print(len(data.keys()))
# for key in data.keys():
    # print(key, data)
