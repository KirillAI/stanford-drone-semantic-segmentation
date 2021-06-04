# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

IMG_WIDTH = 224
IMG_HEIGHT = 224

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    logger.info('get paths of input data')
    paths = []
    for video_path in Path(os.path.join(input_filepath, 'videos')).rglob('*.mov'):
        video_path = str(video_path)
        annotation_path = video_path.replace('videos', 'annotations')
        annotation_path = annotation_path.replace('video.mov', 'annotations.txt')
        paths.append((video_path, annotation_path))
    
    target_path_images = os.path.join(output_filepath, 'all', 'images')
    target_path_masks = os.path.join(output_filepath, 'all', 'masks')
    os.makedirs(target_path_images, exist_ok=True)
    os.makedirs(target_path_masks, exist_ok=True)

    video_id = 0
    dic_annotations = {}

    logger.info('creation of labels.csv')
    labels = []
    for video_path, annotation_path in paths:
        df = pd.read_csv(annotation_path, header=None, sep=' ', names=['TrackID', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'])
        df = df.drop(columns=['TrackID', 'lost', 'occluded', 'generated'])
        dic_annotations[video_path] = df
        labels += df['label'].to_list()

    le = LabelEncoder()
    le.fit(labels)
    mapping = list(zip(range(len(le.classes_)), le.classes_))
    pd.DataFrame(mapping, columns=['index', 'label']).to_csv(os.path.join(output_filepath, 'labels.csv'), index=False)

    logger.info('creation frames and masks')
    for video_path, annotation_path in paths:
        df = dic_annotations[video_path]
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()
        height, width = image.shape[:2]
        scale_height = IMG_HEIGHT / height
        scale_width = IMG_WIDTH / width

        count = 0
        while success:
            obj_frame = df.loc[df['frame'] == count]
            n_obj = len(obj_frame.index)
            if n_obj > 0:
                resized_image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                image_path = 'frame_{:05d}_{:05d}.png'.format(video_id, count)
                full_image_path = os.path.join(target_path_images, image_path)
                cv2.imwrite(full_image_path, resized_image)

                mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)).astype('uint8')
                for i in range(n_obj):
                    one_object = obj_frame.iloc[[i]]
                    xmin = int(one_object['xmin'] * scale_width)
                    xmax = int(one_object['xmax'] * scale_width)
                    ymin = int(one_object['ymin'] * scale_height)
                    ymax = int(one_object['ymax'] * scale_height)
                    label_value = le.transform(one_object['label'])[0] + 1
                    mask[ymin: ymax+1, xmin:xmax+1, ...] = label_value
                mask_path = 'mask_{:05d}_{:05d}.png'.format(video_id, count)
                full_mask_path = os.path.join(target_path_masks, mask_path)
                cv2.imwrite(full_mask_path, mask)
            success, image = vidcap.read()
            count += 1
        video_id += 1

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
