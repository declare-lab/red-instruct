import pickle as pk
import random
import json

data1 = pk.load(open('./data/data_blue_finetune.pkl','rb'))
data2 = pk.load(open('./data/data_helpful_finetune.pkl','rb'))
data3 = pk.load(open('./data/data_sgpt_only.pkl','rb'))
data_combined = data1+data2+data3

random.shuffle(data_combined)
with open("./data/data_combined.json", "w", encoding='utf-8') as f:
    json.dump(data_combined, f, ensure_ascii=False, indent=4)

print("Done...")
