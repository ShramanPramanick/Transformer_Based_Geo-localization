import os
import re
from typing import Dict, Union
from io import BytesIO
import random
from pathlib import Path
import pandas as pd
from PIL import Image
import torchvision
import torch
import msgpack
from argparse import Namespace
import json
import yaml
import numpy as np
import scipy.io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seg_channels', help="Number of Segmentation Channels", type=int)
parser.add_argument('--scene_of_interest', help="indoor - 0, natural - 1, urban - 2, all - 3", type=int)
parser.add_argument('--cuda_base', help="in form cuda:x")
parser.add_argument('--device_ids', help='delimited list input',
                    type=lambda s: [int(item) for item in s.split(',')])
parser.add_argument('--no_of_scenes', help="specify the number of scenes used (3/16/365")
parser.add_argument('--epochs', help="number of epochs", type=int)

args = parser.parse_args()

seg_channels = args.seg_channels
scene_of_interest = args.scene_of_interest
cuda_base = args.cuda_base
device_ids = args.device_ids
num_epochs = args.epochs
scenes = args.no_of_scenes


class MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str,
        seg_path: str,
        segment_file_path:str,
        scene_of_interest: int,
        seg_channels:int,
        percent_seg_pixels:int,
        target_mapping: Dict[str, int],
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        transformation_seg=None,
        shuffle=True,
        meta_path=None,
        cache_size=6 * 4096,
        lat_key="LAT",
        lon_key="LON",
        scene_key ="S3_Label",
        scene_key_2 = "S" + scenes + "_Label" 
    ):

        super(MsgPackIterableDatasetMultiTargetWithDynLabels, self).__init__()
        self.path = path
        self.seg_path = seg_path
        self.segment_file_path = segment_file_path
        self.scene_of_interest = scene_of_interest # indoor - 0, natural - 1, urban - 2
        self.seg_channels = seg_channels
        self.percent_seg_pixels = percent_seg_pixels
        self.cache_size = cache_size
        self.transformation = transformation
        self.transformation_seg = transformation_seg
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.meta_path = meta_path
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.meta = self.meta.astype({lat_key: "float32", lon_key: "float32", scene_key: "float32", scene_key_2: "float32"})
            self.lat_key = lat_key
            self.lon_key = lon_key
            self.scene_key = scene_key
            self.scene_key_2 = scene_key_2

            # filter the file names based on the scene kind
            if self.scene_of_interest == 0 or self.scene_of_interest == 1 or self.scene_of_interest == 2: 
                
                self.meta = self.meta.loc[self.meta[self.scene_key] == self.scene_of_interest]
                self.meta = self.meta.astype({lat_key: "float32", lon_key: "float32", scene_key: "float32", scene_key_2: "float32"})
            # otherwise keep all scene kinds

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)

    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value
        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))

        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        img = self.transformation(img)
        
        if self.meta_path is None:
            return img, x["target"]
        else:
            _id = x[self.key_img_id].decode("utf-8") # image name
            meta = self.meta.loc[_id]

            img_file = self.seg_path + _id
            seg_file = os.path.join(img_file.replace('.jpg', '.png'))

        if os.path.exists(seg_file): 

            img = Image.open(BytesIO(x[self.key_img_encoded]))

            if img.mode != "RGB":
                img = img.convert("RGB")

            if img.width > 320 and img.height > 320:
                img = torchvision.transforms.Resize(320)(img)

            # apply all user specified image transformations
            img = self.transformation(img)
            if self.meta_path is None:
                return img, x["target"]
            else:
                _id = x[self.key_img_id].decode("utf-8") # image name
                meta = self.meta.loc[_id]

                img_file = self.seg_path + _id
                seg_file = os.path.join(img_file.replace('.jpg', '.png'))

                with Image.open(seg_file) as img_seg:
                    # img_seg = np.array(im)

                    if img_seg.mode != "RGB":
                        img_seg = img_seg.convert("RGB")

                    if img_seg.width > 320 and img_seg.height > 320:
                        img_seg = torchvision.transforms.Resize(320)(img_seg)   

                    img_seg = self.transformation_seg(img_seg)

                    # if want 150-D binary channels
                    if self.seg_channels == 150:
                        csv_file = os.path.join(img_file.replace('.jpg', '.csv'))
                    
                        df = pd.read_csv(csv_file) #os.path.join(root, csv_file))
                        class_idx = df['Class Index']  

                        segment_file = scipy.io.loadmat(self.segment_file_path)
                        #'../../segmentation/CSAIL_semantic_segmentation/semantic_cagtegories.mat')
                        color_categories = segment_file['color_categories']

                        seg_channels = np.zeros((np.array(img_seg).shape[0], np.array(img_seg).shape[1], 150), dtype=bool)
                        for c in range(class_idx.shape[0]):
                            RGB_idx = color_categories[class_idx[c]-1,:]
                            mask = (np.array(img_seg) == RGB_idx).all(-1)
                            # print('%')
                            # print((100*np.sum(mask)) / (np.array(img_seg).shape[0] * np.array(img_seg).shape[1]))
                            if (100*np.sum(mask)) / (np.array(img_seg).shape[0] * np.array(img_seg).shape[1]) > self.percent_seg_pixels: # only use the channel if the segmented object takes up more than the threshold of all pixels
                                seg_channels[:,:, class_idx[c]-1] = mask
                    
                    
                        img_seg = seg_channels

                        scipy.io.savemat('img_seg2.mat', {'img_seg':img_seg})
                    img_seg = torch.from_numpy(np.array(img_seg))

                return img, img_seg, x["target"], meta[self.lat_key], meta[self.lon_key], meta[self.scene_key_2]


    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        if len(self.target_mapping[_id]) == 1:
                            x["target"] = self.target_mapping[_id][0]
                        else:
                            x["target"] = self.target_mapping[_id]
                    except KeyError:
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length

class MultiPartitioningClassifier():
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams

    def train_dataloader(self):

        with open(self.hparams.train_label_mapping, "r") as f:
            # target_mapping = json.load(f) # dictionary - name of the image and 
                                          # three labels: S2 class_label for 
                                          # 50_5000, 50_2000, 50_1000 - set based on latitude and longitude

            target_mapping_tmp = json.load(f) # lenght: 25600

            # filter the target_mapping by images with specific scene kind 
            if scene_of_interest == 0 or scene_of_interest == 1 or scene_of_interest == 2: 

                # TODO: read the csv file again - can we make this more efficient to not read the file again for each batch?
                meta_path=self.hparams.train_meta_path    
                meta = pd.read_csv(meta_path, index_col=False)
                scene_key="S3_Label"

                meta = meta.loc[meta[scene_key] == scene_of_interest] 
                # meta.iloc[:, 0] is the first column of the file containing the image file names
                target_mapping = {k: target_mapping_tmp[k] for k in meta.iloc[:, 0] if k in target_mapping_tmp}
            else:
                target_mapping = target_mapping_tmp

        # TODO: remove random transforms for now to make sure that RGB and segmented images match up
        tfm = torchvision.transforms.Compose(
                        [
                        torchvision.transforms.RandomAffine(degrees=(0,15)),
                        torchvision.transforms.ColorJitter(),
                        torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.TenCrop((224, 224)),
                        torchvision.transforms.Lambda(
                                                     lambda crops: torch.stack([crop for crop in crops])
                                                     ),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                                            )
                

        # TODO: not using this now, figure out what transformations make sense for segmented images
        tfm_seg = torchvision.transforms.Compose(
                        [
                        #torchvision.transforms.RandomAffine(degrees=(0,15)),
                        #torchvision.transforms.ColorJitter(),
                        #torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.TenCrop((224, 224)),
                        torchvision.transforms.Lambda(
                                                     lambda crops: torch.stack([crop for crop in crops])
                                                     ),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                                            )


        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_train_dir,
            seg_path=self.hparams.msgpack_train_seg_dir,
            segment_file_path=self.hparams.segment_file_path,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=True,
            transformation=tfm,
            transformation_seg = tfm_seg,
            meta_path=self.hparams.train_meta_path,
            scene_of_interest=scene_of_interest,
            seg_channels=seg_channels,
            percent_seg_pixels=self.hparams.percent_seg_pixels,
            cache_size=1024,            
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
            drop_last = True,
        )
        return dataloader

    def val_dataloader(self):

        with open(self.hparams.val_label_mapping, "r") as f:
            target_mapping_tmp = json.load(f) # lenght: 25600

            # filter the target_mapping by images with specific scene kind 
            if scene_of_interest == 0 or scene_of_interest == 1 or scene_of_interest == 2: 

                # TODO: read the csv file again - can we make this more efficient to not read the file again for each batch?
                meta_path=self.hparams.val_meta_path    
                meta = pd.read_csv(meta_path, index_col=False)
                scene_key="S3_Label"

                meta = meta.loc[meta[scene_key] == scene_of_interest] 
                # meta.iloc[:, 0] is the first column of the file containing the image file names
                target_mapping = {k: target_mapping_tmp[k] for k in meta.iloc[:, 0] if k in target_mapping_tmp}
            else:
                target_mapping = target_mapping_tmp

        tfm = torchvision.transforms.Compose(
                        [
                        #torchvision.transforms.RandomAffine(degrees=(0,15)),
                        #torchvision.transforms.ColorJitter(),
                        #torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.TenCrop((224, 224)),
                        torchvision.transforms.Lambda(
                                                     lambda crops: torch.stack([crop for crop in crops])
                                                     ),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                                            )
         
        # TODO: not using this now, figure out what transformations make sense for segmented images
        tfm_seg = torchvision.transforms.Compose(
                        [
                        #torchvision.transforms.RandomAffine(degrees=(0,15)),
                        #torchvision.transforms.ColorJitter(),
                        #torchvision.transforms.RandomHorizontalFlip(),
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Resize((256, 256)),
                        torchvision.transforms.TenCrop((224, 224)),
                        torchvision.transforms.Lambda(
                                                     lambda crops: torch.stack([crop for crop in crops])
                                                     ),
                        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                                            )


        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.msgpack_val_dir,
            seg_path=self.hparams.msgpack_val_seg_dir,
            segment_file_path=self.hparams.segment_file_path,
            target_mapping=target_mapping,
            key_img_id=self.hparams.key_img_id,
            key_img_encoded=self.hparams.key_img_encoded,
            shuffle=True,
            transformation=tfm,
            transformation_seg = tfm_seg,
            meta_path=self.hparams.val_meta_path,
            scene_of_interest=scene_of_interest,
            seg_channels=seg_channels,
            percent_seg_pixels=self.hparams.percent_seg_pixels,
            cache_size=1024,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers_per_loader,
            pin_memory=True,
            drop_last=True,
        )
        
        return dataloader

def main():    
    with open('../config/base_config.yml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
   
    tmp_model = MultiPartitioningClassifier(hparams=Namespace(**model_params))
    
    train_data_loader = tmp_model.train_dataloader()

    val_data_loader = tmp_model.val_dataloader()


    #     ############### save and visualize image for debugging ################  
    # iterate through the dataset
 
    it = iter(train_data_loader)
    first_batch = next(it)

    image_it = first_batch[0] # RGB image
    seg_image_it = first_batch[1] # segmented image

    S2_labels_it = first_batch[2] 

    print(image_it.shape)
    print(seg_image_it.shape)
    lat_it = first_batch[3]
    lon_it = first_batch[4]
    scene_it = first_batch[5]


if __name__ == "__main__":
    main()
