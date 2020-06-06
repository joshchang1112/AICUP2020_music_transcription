import json
import numpy as np

with open("answer.json") as f:
     result=json.load(f)
#print(result['1'])

for key, value in result.items():
    delete_idx = []
    for i in range(len(value)):
        #print(i)
        if value[i][2] < 65:
            delete_idx.append(i)
        else:
            
            value[i][2] = int(round(np.log2(value[i][2]/440) * 12 + 69))
    #print(delete_idx)
    count = 0
    for idx in delete_idx:
        del value[idx-count]
        count +=1

with open('answer_2.json', 'w') as f_obj:
    json.dump(result, f_obj, indent = 4)

