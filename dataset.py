import shutil
import os
import random
import tqdm


DATA_DIR = "../../dataset/views"

for scene in tqdm.tqdm(os.listdir(DATA_DIR)[216:]):
    for tp in ["img", "semantics", "views"]:
        for mode in ['train', "test", 'val']:
            os.makedirs(os.path.join(DATA_DIR, scene,tp,mode),exist_ok=True)
    
    
    data_num = len(os.listdir(os.path.join(DATA_DIR,scene,'views')))  -3
    

    try:
        for i in range(data_num):
            a = random.randint(0,9)        
            if a < 7:
                shutil.move(
                    os.path.join(DATA_DIR,scene,'views',f'scene_{i}.json'),
                    os.path.join(DATA_DIR,scene,'views','train'),
                )

                shutil.move(
                    (os.path.join(DATA_DIR,scene,'img',f'scene_{i}.png')),
                    (os.path.join(DATA_DIR,scene,'img','train')),
                )
                shutil.move(
                    (os.path.join(DATA_DIR,scene,'semantics',f'scene_{i}.npy')),
                    (os.path.join(DATA_DIR,scene,'semantics','train')),
                )
            elif a == 7 or a == 8:
                shutil.move(
                    (os.path.join(DATA_DIR,scene,'views',f'scene_{i}.json')),
                    (os.path.join(DATA_DIR,scene,'views','test')),
                )

                shutil.move(
                    (os.path.join(DATA_DIR,scene,'img',f'scene_{i}.png')),
                    (os.path.join(DATA_DIR,scene,'img','test')),
                )

                shutil.move(
                    (os.path.join(DATA_DIR,scene,'semantics',f'scene_{i}.npy')),
                    (os.path.join(DATA_DIR,scene,'semantics','test')),
                )
            else:
                shutil.move(
                    (os.path.join(DATA_DIR,scene,'views',f'scene_{i}.json')),
                    (os.path.join(DATA_DIR,scene,'views','val')),
                )
                shutil.move(
                    (os.path.join(DATA_DIR,scene,'img',f'scene_{i}.png')),
                    (os.path.join(DATA_DIR,scene,'img','val')),
                )

                shutil.move(
                    (os.path.join(DATA_DIR,scene,'semantics',f'scene_{i}.npy')),
                    (os.path.join(DATA_DIR,scene,'semantics','val')),
                )
    except:
        continue