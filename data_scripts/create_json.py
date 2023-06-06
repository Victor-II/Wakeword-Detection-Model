import os
import json
import random


def main(args):
    zeros = os.listdir(args.get('zero_label_dir'))
    ones = os.listdir(args.get('one_label_dir'))
    random.shuffle(zeros)
    random.shuffle(ones)
    percent = args.get('test_percent')
    data = []
    for zero in zeros[: int(0.5*len(zeros))]:
        data.append({
            "key": os.path.join(args.get('zero_label_dir'), zero),
            "label": 0
        })
    for one in ones[: int(0.5*len(ones))]:
        data.append({
            "key": os.path.join(args.get('one_label_dir'), one),
            "label": 1
        })
    random.shuffle(data)

    data_len = len(data)
    split_point = int(data_len - data_len / percent)

    with open(args.get('json_save_path') +"/"+ 'train.json','w') as file:
        
        for i in range(split_point):
            line = json.dumps(data[i])
            file.write(line + "\n")
    file.close()

    with open(args.get('json_save_path') +"/"+ 'test.json','w') as file:
        
        for i in range(split_point, data_len):
            line = json.dumps(data[i])
            file.write(line + "\n")
    file.close()

if __name__ == "__main__":
    
    args = {
        'zero_label_dir': 'C:/Users/Victor/Desktop/WakeWordData/label_zero',
        'one_label_dir': 'C:/Users/Victor/Desktop/WakeWordData/label_one',
        'json_save_path': 'C:/Users/Victor/Desktop/WakeWordData/json_20percent',
        'test_percent': 10
    }
    
    main(args)
