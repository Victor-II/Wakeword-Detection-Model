import os
import shutil

def main(args):
    data_dir = os.listdir(args.get('data_dir'))
    dest_dir = args.get('dest_dir')
    for file in data_dir:
        for i in range(args.get('copy_number')):
            file_path = os.path.join(args.get('data_dir'), file)
            shutil.copy(file_path, dest_dir)
            new_file_path = os.path.join(dest_dir, file)
            new_file_name = os.path.join(dest_dir, str(i) + "_" + file)
            os.rename(new_file_path, new_file_name)

if __name__ == "__main__":
    
    args = {
        'data_dir': 'C:/Users/Victor/Desktop/WakeWordData/wakewords',
        'dest_dir': 'C:/Users/Victor/Desktop/WakeWordData/label_one',
        'copy_number': 100
    }

    main(args)