# -*- coding: utf-8 -*-

"""cityscapes fine and coarse dataset
.
├── gtFine_trainvaltest
│   ├── gtCoarse
│   │   ├── train
│   │   ├── train_extra
│   │   └── val
│   └── gtFine
│       ├── test
│       ├── train
│       └── val
└── leftImg8bit_trainvaltest
    └── leftImg8bit
        ├── test
        ├── train
        ├── train_extra
        └── val

"""
import os
from PIL import Image
import glob
import mxnet
from mxnet import cpu
import numpy as np
from ..segbase import SegmentationDataset

class CityscapesSegmentation(SegmentationDataset):
    NUM_CLASS=19
    foreground_class_ids = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    
    def __init__(self,root=os.path.expanduser('~/.mxnet/datasets/cityscapes'),
                 split='train',mode=None,transform=None,is_fine=True):
        super().__init__(root,split,mode,transform)
        fine_str='gtFine' if is_fine else 'gtCoarse'
        mask_dir=os.path.join(root,'gtFine_trainvaltest',fine_str,split)
        image_dir=os.path.join(root,'leftImg8bit_trainvaltest','leftImg8bit',split)
        
        glob_images=glob.glob(os.path.join(image_dir,'*','*leftImg8bit.png'))
        glob_annotations=glob.glob(os.path.join(mask_dir,'*','*labelIds.png'))
        glob_images.sort()
        glob_annotations.sort()
        print('%s glob images'%split,len(glob_images))
        print('%s glob annotations'%split,len(glob_annotations))
        assert len(glob_images)==len(glob_annotations),'image number %d != annotations number %d'%(len(glob_images),len(glob_annotations))
        
        self.images=glob_images
        self.masks=glob_annotations
        
    def __getitem__(self,index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        
        target = Image.open(self.masks[index])
        # synchrosized transform
        if self.mode == 'train' or self.mode == 'train_extra':
            img, target = self._sync_transform(img, target)
        elif self.mode == 'val':
            img, target = self._val_sync_transform(img, target)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        
        target = self._mask_transform(target)
        return img, target
    
    def _mask_transform(self, target):
        mask = np.zeros_like(target,dtype='int32')+255
        for idx,class_id in enumerate(self.foreground_class_ids):
            mask[target==class_id] = idx
    
        return mxnet.ndarray.array(mask, cpu(0))
    
    def __len__(self):
        return len(self.images)
    
    @property
    def classes(self):
        """Category names."""
        return ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 
                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 
                'motorcycle', 'bicycle')
        
class CityscapesFineSegmentation(CityscapesSegmentation):
    def __init__(self,root=os.path.expanduser('~/.mxnet/datasets/cityscapes'),
                 split='train',mode=None,transform=None):
        super().__init__(root=root,split=split,mode=mode,transform=transform,is_fine=True)
        
class CityscapesCoarseSegmentation(CityscapesSegmentation):
    def __init__(self,root=os.path.expanduser('~/.mxnet/datasets/cityscapes'),
                 split='train',mode=None,transform=None):
        super().__init__(root=root,split=split,mode=mode,transform=transform,is_fine=False)