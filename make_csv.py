import os
import json
import glob
import pandas as pd

root_dir = '/home/ace19/dl_data/Arirang_Dataset'
root_dir = '/path/to/your_dataset_dir'

# train label path
train_labels = glob.glob(root_dir + '/your_dataset_dir/json/*')


def readJSON(path):
    with open(path) as f:
        data = json.load(f)

    return data


def getLabelName(dic):
    '''
    Parameters:
    -----------
    dic: result of readJSON

    Returns:
    --------
    pd.DataFrame
        dataframe with image_id as index and number of each objects contained in the image as values
    '''

    output = {}
    image_id = dic['features'][0]['properties']['image_id']

    for i in range(len(dic['features'])):
        label_name = dic['features'][i]['properties']['label_name']
        if output.get(label_name) == None:
            output[label_name] = 1
        else:
            output[label_name] += 1

    return pd.DataFrame(output, index=[image_id])


total_df = pd.DataFrame()
for path in train_labels:
    total_df = pd.concat([total_df, getLabelName(readJSON(path))], axis=0, join='outer')

total_df.fillna(0, inplace=True)
total_df.index.name = 'image name'

total_df.to_csv(os.path.join(root_dir, 'datasets', 'datasets.csv'))
