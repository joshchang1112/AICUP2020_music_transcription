import json
import numpy as np

with open("answer.json") as f:
     result=json.load(f)
#print(result['1'])

for key, value in result.items():
    delete_idx = []
    for i in range(len(value)):
        if value[i][2] <= 0:
            delete_idx.append(i)
    
    #print(delete_idx)
    count = 0
    for idx in delete_idx:
        del value[idx-count]
        count +=1

with open('answer_2.json', 'w') as f_obj:
    json.dump(result, f_obj)

