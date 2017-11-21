from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils

import sys
from coco_caption.pycxtools.coco import COCO
from coco_caption.pycxevalcap.eval import COCOEvalCap
import hashlib

def language_eval(dataset, preds, model_id, split):
    annFile = '../neuraltalk2.pytorch/data/dataset/val_ref.json'

    encoder.FLOAT_REPR = lambda o: format(o, '.3f')

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    cache_path_pred = os.path.join('eval_results/', model_id + '_' + split + '_pred.json')
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '.json')

    coco = COCO(annFile)
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt = [p for p in preds if int(int(hashlib.sha256(p['image_id']).hexdigest(), 16) % sys.maxint) in valids]
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))
    json.dump(preds_filt, open(cache_path_pred, 'w')) # serialize to temporary json file. Sigh, COCO API...

    cocoRes = coco.loadRes(cache_path_pred)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # create output dictionary
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score

    imgToEval = cocoEval.imgToEval
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[int(int(hashlib.sha256(image_id).hexdigest(), 16) % sys.maxint)]['caption'] = caption
    with open(cache_path, 'w') as outfile:
        json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)

    return out

def eval_split(cnn_model, model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    use_img = eval_kwargs.get('use_img', 0)
    img_csize = eval_kwargs.get('img_csize', 224)
    use_topic = eval_kwargs.get('use_topic', 0)
    use_fc = eval_kwargs.get('use_fc', 0)
    use_att = eval_kwargs.get('use_att', 0)
    gpu_num = eval_kwargs.get('gpu_num', 1)
    # Make sure in the evaluation mode
    if use_img != 0 :
        cnn_model.eval()
    model.eval()

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    loader.reset_iterator(split)
    while True:
        # Load data from train split (0)
        data = loader.get_batch(split)
        if use_img != 0 :
            data['img'] = utils.prepro_images(data['img'], img_csize, False)

        n = n + loader.batch_size

        #evaluate loss if we have the labels
        loss = 0
        topics = None
        fc_feats = None
        att_feats = None
        if use_img != 0 :
            if use_topic != 0 :
                tmp = [data['img'], data['topics'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                images, topics, labels, masks = tmp
            else :
                tmp = [data['img'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                images, labels, masks = tmp
            att_feats = cnn_model(images).permute(0, 2, 3, 1)
            fc_feats = att_feats.mean(2).mean(1)
            if not use_att:
                att_feats = Variable(torch.FloatTensor(1,1,1,1).cuda())
            if use_topic == 0 :
                topics = Variable(torch.FloatTensor(1,1,1,1).cuda())
            if use_fc == 0 :
                fc_feats = Variable(torch.FloatTensor(1,1,1,1).cuda())
            att_feats = att_feats.unsqueeze(1).expand(*((att_feats.size(0), loader.seq_per_img,) \
                                                        + att_feats.size()[1:])).contiguous().view(*((att_feats.size(0) * loader.seq_per_img,) + att_feats.size()[1:]))
            fc_feats = fc_feats.unsqueeze(1).expand(*((fc_feats.size(0), loader.seq_per_img,) \
                                                      + fc_feats.size()[1:])).contiguous().view(*((fc_feats.size(0) * loader.seq_per_img,) + fc_feats.size()[1:]))
        else :
            if use_topic != 0 and use_fc != 0 :
                tmp = [data['fc_feats'], data['att_feats'], data['topics'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                fc_feats, att_feats, topics, labels, masks = tmp
            elif use_topic != 0 :
                tmp = [data['att_feats'], data['topics'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                att_feats, topics, labels, masks = tmp
            elif use_fc != 0 :
                tmp = [data['fc_feats'], data['att_feats'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                fc_feats, att_feats, labels, masks = tmp
            else :
                tmp = [data['att_feats'], data['labels'], data['masks']]
                tmp = [Variable(torch.from_numpy(_), volatile=True).cuda() for _ in tmp]
                att_feats, labels, masks = tmp

        # forward the model to get loss
        if data.get('labels', None) is not None:
            if use_topic:
                loss = crit(model(fc_feats, att_feats, topics, labels), labels[:,1:], masks[:,1:]).data[0]
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1
            else:
                loss = crit(model(fc_feats, att_feats, labels), labels[:,1:], masks[:,1:]).data[0]
                loss_sum = loss_sum + loss
                loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        att_feats = att_feats.data.cpu().numpy()[np.arange(loader.batch_size) * loader.seq_per_img]
        att_feats = Variable(torch.from_numpy(att_feats), volatile=True).cuda()
        if use_fc != 0 :
            fc_feats = fc_feats.data.cpu().numpy()[np.arange(loader.batch_size) * loader.seq_per_img]
            fc_feats = Variable(torch.from_numpy(fc_feats), volatile=True).cuda()
        if use_topic != 0 :
            topics = topics.data.cpu().numpy()[np.arange(loader.batch_size) * loader.seq_per_img]
            topics = Variable(torch.from_numpy(topics), volatile=True).cuda()
        # forward the model to also get generated samples for each image
        if use_topic:
            seq, _ = model.sample(fc_feats, att_feats, topics, eval_kwargs)
        else:
            seq, _ = model.sample(fc_feats, att_feats, eval_kwargs)

        #set_trace()
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            sent = ''.join(sent.strip().split())
            if sent == '' :
                sent = ' '
            entry = {'caption': sent, 'image_id': data['infos'][k]['image_id']}
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['image_id']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose:
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None
    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, eval_kwargs['id'], split)

    # Switch back to training mode
    if use_img != 0 :
        cnn_model.train()
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats


