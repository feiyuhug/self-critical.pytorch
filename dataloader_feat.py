from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random

import torch
import torch.utils.data as data

from scipy.misc import imread, imresize
import multiprocessing
from torchvision import transforms as trn
preprocess = trn.Compose([
        #trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_img_data(file_path, img_size):
    img = imread(file_path)
    img = imresize(img, (img_size, img_size))

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

    img = preprocess(torch.from_numpy(img.transpose(2, 0, 1).astype('float32') / 255.0)).numpy()
    return img

def get_npy_data(ix, fc_file, att_file, use_att):
    if use_att == True:
        return [np.load(fc_file), np.load(att_file)['feat']]
    else:
        return [np.load(fc_file), np.zeros((1, 1, 1))]


class DataLoader(data.Dataset):
    def reset_iterator(self, split, batch_size=None):
        self.iterators[split] = 0
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, \
                                                    split == 'train', batch_size=batch_size)

    def syn_iterator_all(self):
        for split in self.iterators.keys() :
            del self._prefetch_process[split]
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.seq_per_img = opt.seq_per_img
        self.use_att = getattr(opt, 'use_att', True)
        self.use_img = getattr(opt, 'use_img', 1)
        self.img_fold = getattr(opt, 'img_fold', 'data/images')
        self.img_size = getattr(opt, 'img_size', 256)
        self.use_fc = getattr(opt, 'use_fc', 1)
        self.use_topic = getattr(opt, 'use_topic', 1)

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file: ', opt.input_json)
        self.info = json.load(open(self.opt.input_json))
        self.ix_to_word = self.info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        # open the hdf5 file
        print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_label_h5, \
              opt.input_image_h5, opt.input_topic_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')
        if self.use_img != 0 :
            self.h5_image_file = h5py.File(self.opt.input_image_h5)
        if self.use_topic != 0 :
            self.h5_topic_file = h5py.File(self.opt.input_topic_h5)
        self.input_fc_dir = self.opt.input_fc_dir
        self.input_att_dir = self.opt.input_att_dir

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        # extract image size from dataset
        '''
        if self.use_img != 0 :
            images_size = self.h5_image_file['images'].shape
            assert len(images_size) == 4, 'images should be a 4D tensor'
            assert images_size[2] == images_size[3], 'width and height must match'
            self.num_images = images_size[0]
            self.num_channels = images_size[1]
            self.max_image_size = images_size[2]
            print('read %d images of size %dx%dx%d' %(self.num_images,
                        self.num_channels, self.max_image_size, self.max_image_size))
        '''
        # separate out indexes for each of the provided splits
        self.split_ix = {'train': [], 'val': [], 'test': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)
            elif opt.train_only == 0:  # restval
                self.split_ix['train'].append(ix)

        print('assigned %d images to split train' % len(self.split_ix['train']))
        print('assigned %d images to split val' % len(self.split_ix['val']))
        print('assigned %d images to split test' % len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {}  # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split == 'train')
            # Terminate the child process when the parent exists

        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]

        import atexit
        atexit.register(cleanup)

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        fc_batch = []  # np.ndarray((batch_size * seq_per_img, self.opt.fc_feat_size), dtype = 'float32')
        att_batch = []  # np.ndarray((batch_size * seq_per_img, 14, 14, self.opt.att_feat_size), dtype = 'float32')
        img_batch = []
        topics_batch = []
        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype='float32')

        wrapped = False

        infos = []
        gts = []
        tmp = self._prefetch_process[split].split_loader.next()
        for i in range(batch_size):
            import time
            #t_start = time.time()
            # fetch image
            if self.use_img != 0 :
                ix_, img = tmp[i]
                #raw_image = self.h5_image_file['images'][ix, :, :, :]
                #img = preprocess(torch.from_numpy(raw_image.astype('float32')/255.0)).numpy()
                img_batch.append(img)
            else :
                ix_, tmp_fc, tmp_att = tmp[i]
                fc_batch += [tmp_fc] * seq_per_img
                att_batch += [tmp_att] * seq_per_img
            ix, tmp_wrapped = self._prefetch_process[split].get()
            assert ix_ == ix
            if self.use_topic != 0 :
                topics = self.h5_topic_file['topics'][ix, :]
                topics_batch += [topics] * seq_per_img
            # fetch raw image


            # fetch the sequence labels
            ix1 = self.label_start_ix[ix] - 1  # label_start_ix starts from 1
            ix2 = self.label_end_ix[ix] - 1
            ncap = ix2 - ix1 + 1  # number of captions available for this image
            assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

            if ncap < seq_per_img:
                # we need to subsample (with replacement)
                seq = np.zeros([seq_per_img, self.seq_length], dtype='int')
                for q in range(seq_per_img):
                    ixl = random.randint(ix1, ix2)
                    seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
            else:
                ixl = random.randint(ix1, ix2 - seq_per_img + 1)
                seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

            label_batch[i * seq_per_img: (i + 1) * seq_per_img, 1: self.seq_length + 1] = seq

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])

            # record associated info as well
            info_dict = {}
            info_dict['image_id'] = self.info['images'][ix]['image_id']
            infos.append(info_dict)
            # print(i, time.time() - t_start)

        # generate mask
        #t_start = time.time()
        nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, label_batch)))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        # print('mask', time.time() - t_start)

        data = {}
        if self.use_img != 0 :
            data['img'] = np.stack(img_batch)
        else :
            data['fc_feats'] = np.stack(fc_batch)
            data['att_feats'] = np.stack(att_batch)
        if self.use_topic != 0 :
            data['topics'] = np.stack(topics_batch)
        data['labels'] = label_batch
        data['gts'] = gts
        data['masks'] = mask_batch
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index  # self.split_ix[index]
        if self.use_img != 0 :
            img = get_img_data(os.path.join(self.img_fold, str(self.info['images'][ix]['image_id']) + '.jpg'),\
                            self.img_size)
            return [ix, img]
        else :
            feats = get_npy_data(ix, \
                                os.path.join(self.input_fc_dir, str(self.info['images'][ix]['image_id']) + '.npy'),
                                os.path.join(self.input_att_dir, str(self.info['images'][ix]['image_id']) + '.npz'),
                                self.use_att
                                )
            return [ix] + feats

    def __len__(self):
        return len(self.info['images'])


class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""

    def __init__(self, split, dataloader, if_shuffle=False, batch_size=None):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle
        if batch_size :
            self.reset(batch_size)
        else :
            self.reset(self.dataloader.batch_size)
    # Add more in the queue
    def reset(self, batch_size):
        """
        Two cases:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 0, the merge is done in DataLoader class
        #print('cpu count: %d'%(multiprocessing.cpu_count()))
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                                 batch_size=batch_size,
                                                 sampler=self.dataloader.split_ix[self.split][
                                                         self.dataloader.iterators[self.split]:],
                                                 shuffle=False,
                                                 pin_memory=False,
                                                 num_workers = 8,
                                                 collate_fn=lambda x: x))

    def _get_next_minibatch_inds(self):
        if not hasattr(self, 'split_loader'):
            self.reset()
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped

    def get(self):
        ix, wrapped = self._get_next_minibatch_inds()
        if wrapped :
            print('loader reset...')
            self.reset()
        return ix, wrapped
