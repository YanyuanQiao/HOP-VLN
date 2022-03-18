import csv
import numpy as np
import base64
import os
import h5py
import functools
import pdb
from ipdb import set_trace
csv.field_size_limit(500 * 1024 * 1024)
class Feature:
    def __init__(self, feature_store, panoramic, max_load=-1):
        self.feature_store = feature_store
        self.image_h, self.image_w, self.vfov, self.features = None, None, None, None
        self.panoramic = panoramic
        self.max_load = max_load
        self._load()

    def _load(self):
        print('Loading image features from %s' % str(self.feature_store))
        if self.feature_store == 'img_features/ResNet-152-imagenet.tsv':
            self.features, self.image_h, self.image_w, self.vfov = self.__loadResNet(self.feature_store)
            self.rollout = self.rollout_single
        elif self.feature_store == 'img_features/bottom_up':
            self.features, self.image_h, self.image_w, self.vfov = \
                self.__loadBottomUp(self.feature_store)
            self.rollout = self.rollout_single
        elif self.feature_store == 'img_features/ResNet-152-imagenet.tsv+img_features/bottom_up':
            features_resnet, self.image_h, self.image_w, self.vfov = \
                self.__loadResNet(self.feature_store.split('+')[0])
            features_bottom, _, _, _ = \
                self.__loadBottomUp(self.feature_store.split('+')[1])
            self.features = features_bottom
            for key in features_bottom.keys():
                self.features[key] = np.hstack([features_resnet[key],
                                                features_bottom[key]])
            self.rollout = self.rollout_single
        elif self.feature_store == 'img_features/ResNet-152-imagenet.tsv+img_features/bottom_up+bbox':
            features_resnet, self.image_h, self.image_w, self.vfov = \
                self.__loadResNet(self.feature_store.split('+')[0])
            features_bottom, _, _, _ = \
                self.__loadBottomUp(self.feature_store.split('+')[1])
            self.features = features_bottom
            for key in features_bottom.keys():
                self.features[key] = np.hstack([features_resnet[key],
                                                features_bottom[key]])
            self.rollout = functools.partial(self.rollout_with_bbox, self.feature_store.split('+')[1])
        else:
            print('Image features not provided')
            self.rollout = (lambda a,b,c: None) if not self.features else self.rollout_single
            self.features, self.image_h, self.image_w, self.vfov = None, 480, 640, 60

    def __loadResNet(self, feature_store):
        tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h', 'vfov', 'features']
        features, image_h, image_w, vfov = {}, 480, 640, 60
        read_num = 0
        while (read_num < 20):
            print('read_num %d' % (read_num))
            try:
                with open(feature_store, "r+") as tsv_in_file:
                    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=tsv_fieldnames)
                    for item in reader:
                        image_h = int(item['image_h'])
                        image_w = int(item['image_w'])
                        vfov = int(item['vfov'])
                        long_id = item['scanId'] + '_' + item['viewpointId']
                        features[long_id] = np.frombuffer(base64.b64decode(item['features']), dtype=np.float32).reshape((36, 2048))
                        if self.max_load > 0 and len(features) >= self.max_load:
                            break
                break
            except OSError:
                read_num += 1

        return features, image_h, image_w, vfov

    def __loadBottomUp(self, feature_store):
        # tsv_fieldnames = ['scanId', 'viewpointId', 'image_w', 'image_h',
        #                   'num_boxes', 'features', 'cls_prob', 'captions']
        scanIds = os.listdir(feature_store)
        temp_folder = os.path.join(feature_store, scanIds[0])
        temp_fname = os.path.join(temp_folder, os.listdir(temp_folder)[0])
        with h5py.File(temp_fname, 'r') as f:
            image_h = int(f['0']['image_h'].value)
            image_w = int(f['0']['image_w'].value)
            view_size = len(f)  # 36
            feature_size = (f['0']['features'].value).shape[1]  # 2048
        vfov = 60
        features = {}
        for scanId in scanIds:
            folder = os.path.join(feature_store, scanId)
            viewpointIds_h5 = os.listdir(folder)
            for viewpointId_h5 in viewpointIds_h5:
                fname = os.path.join(folder, viewpointId_h5)
                with h5py.File(fname, "r") as viewpoint:
                    assert len(viewpoint.keys()) == 36
                    long_id = scanId + '_' + viewpointId_h5[:-3]  # rstrip('.h5')
                    temp = np.zeros((view_size, feature_size))
                    for image_id in range(36):
                        item = viewpoint[str(image_id)]
                        temp[image_id, :] = np.mean(item['features'].value, 0)
                    features[long_id] = temp

                if self.max_load > 0 and len(features) >= self.max_load:
                    break
        return features, image_h, image_w, vfov

    def rollout_single(self, scanId, viewpointId, viewIndex):
        long_id = scanId + '_' + viewpointId
        feature = self.features[long_id]
        feature = (feature[viewIndex, :], feature)        
        return feature

    @functools.lru_cache(maxsize=20000)
    def rollout_with_bbox(self, feature_store, scanId, viewpointId, viewIndex):
        long_id = scanId + '_' + viewpointId
        fname = os.path.join(feature_store, long_id+'.h5')
        with h5py.File(fname, "r") as viewpoint:
            assert len(viewpoint.keys()) == 36
            item = viewpoint[str(viewIndex)]
            features = item['features'].value
        return features

class Feature_bnb:
    def __init__(self, feature_store, max_load=-1):
        self.feature_store = feature_store
        self.features = None
        self.max_load = max_load
        self._load()

    def _load(self):
        #set_trace()
        print('Loading image features from %s' % str(self.feature_store))
        self.features = self.__loadResNet_bnb(self.feature_store)
        self.rollout = self.rollout_single_bnb

    def __loadResNet_bnb(self, feature_store):
        features = {}

        bnb=np.load(feature_store,  allow_pickle=True)
        bnbimg = bnb['data']
        read_num = 0
        try:
            for item in bnbimg:
                picid = item['picid']
                features[picid] = item['features']
        except OSError:
            read_num += 1
        return features

    def rollout_single_bnb(self, picid):

        feature = self.features[picid]
   
        return feature
