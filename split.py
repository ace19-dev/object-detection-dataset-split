import argparse
import os
import shutil

import pandas as pd
from sklearn.model_selection import GroupKFold


def split_train_val3():
    total_df = pd.read_csv(os.path.join(args.source_dir, 'train.csv'))
    print('total:\n', total_df.iloc[:, 1:].sum().sort_values(), '\n')

    NUM_FOLD = 5
    gkf = GroupKFold(n_splits=NUM_FOLD)

    results = []
    for idx, (train_idx, val_idx) in enumerate(gkf.split(total_df, total_df["car"],
                                                         groups=total_df['image name'].tolist())):
        train_fold = total_df.iloc[train_idx]
        val_fold = total_df.iloc[val_idx]

        print('\nsplit: ', idx)
        print('train_df:\n', train_fold.iloc[:, 1:].sum().sort_values())
        print('\nval_df:\n', val_fold.iloc[:, 1:].sum().sort_values())
        print("+++++++++++++++++++++++++++++++\n")

        results.append((train_fold, val_fold))

    for i, splited in enumerate(results):
        train_img = os.path.join(args.target_dir, 'train_fold' + str(i), 'images')
        train_json = os.path.join(args.target_dir, 'train_fold' + str(i), 'json')
        val_img = os.path.join(args.target_dir, 'val_fold' + str(i), 'images')
        val_json = os.path.join(args.target_dir, 'val_fold' + str(i), 'json')

        if not os.path.exists(train_img):
            os.makedirs(train_img)
        if not os.path.exists(train_json):
            os.makedirs(train_json)
        if not os.path.exists(val_img):
            os.makedirs(val_img)
        if not os.path.exists(val_json):
            os.makedirs(val_json)

        for idx, img in enumerate(splited[0]['image name']):
            shutil.copyfile(os.path.join(args.source_dir, 'images', img),
                            os.path.join(train_img, img))
            shutil.copyfile(os.path.join(args.source_dir, 'json', img.split('.')[0] + '.json'),
                            os.path.join(train_json, img.split('.')[0] + '.json'))
        for idx, img in enumerate(splited[1]['image name']):
            shutil.copyfile(os.path.join(args.source_dir, 'images', img),
                            os.path.join(val_img, img))
            shutil.copyfile(os.path.join(args.source_dir, 'json', img.split('.')[0] + '.json'),
                            os.path.join(val_json, img.split('.')[0] + '.json'))


def main(args):
    if args.target_dir is not None:
        if not os.path.exists(args.target_dir):
            os.makedirs(args.target_dir)

    split_train_val3()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir',
                        type=str,
                        default='/path/to/your_dataset',
                        help='Where is train image to load')
    parser.add_argument('--target_dir', type=str,
                        default='/path/to/target_dir',
                        help='Directory to save splited dataset')

    args = parser.parse_args()
    main(args)
