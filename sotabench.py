import argparse
import os
import os.path as osp
import shutil
import tempfile
import urllib.request
import json

from sotabencheval.object_detection import COCOEvaluator

import copy
import mmcv
import torch
import torch.distributed as dist
torch.manual_seed(0)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, load_checkpoint

from mmdet.apis import init_dist
from mmdet.core import coco_eval, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

# Extract val2017 zip
from torchbench.utils import extract_archive
image_dir_zip = osp.join('./.data/vision/coco', 'val2017.zip')
extract_archive(from_path=image_dir_zip, to_path='./.data/vision/coco')

from pathlib import Path

if not os.path.isdir(Path.home() / '.cache/torch'):
    os.mkdir(Path.home() / '.cache/torch')

def xyxy2xywh(bbox):
    _bbox = bbox.tolist()
    return [
        _bbox[0],
        _bbox[1],
        _bbox[2] - _bbox[0] + 1,
        _bbox[3] - _bbox[1] + 1,
    ]


def proposal2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        bboxes = results[idx]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['image_id'] = img_id
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['category_id'] = 1
            json_results.append(data)
    return json_results


def det2json(dataset, results):
    json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        try:
            result = results[idx]
        except IndexError:
            break
        for label in range(len(result)):
            bboxes = result[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                json_results.append(data)
    return json_results


def segm2json(dataset, results):
    bbox_json_results = []
    segm_json_results = []
    for idx in range(len(dataset)):
        img_id = dataset.img_ids[idx]
        try:
            det, seg = results[idx]
        except IndexError:
            break
        for label in range(len(det)):
            # bbox results
            bboxes = det[label]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = dataset.cat_ids[label]
                bbox_json_results.append(data)

            # segm results
            # some detectors use different score for det and segm
            if len(seg) == 2:
                segms = seg[0][label]
                mask_score = seg[1][label]
            else:
                segms = seg[label]
                mask_score = [bbox[4] for bbox in bboxes]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['score'] = float(mask_score[i])
                data['category_id'] = dataset.cat_ids[label]
                segms[i]['counts'] = segms[i]['counts'].decode()
                data['segmentation'] = segms[i]
                segm_json_results.append(data)
    return bbox_json_results, segm_json_results


def cached_results2json(dataset, results, out_file):
    result_files = dict()
    if isinstance(results[0], list):
        json_results = det2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        mmcv.dump(json_results, result_files['bbox'])
    elif isinstance(results[0], tuple):
        json_results = segm2json(dataset, results)
        result_files['bbox'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'bbox')
        result_files['segm'] = '{}.{}.json'.format(out_file, 'segm')
        mmcv.dump(json_results[0], result_files['bbox'])
        mmcv.dump(json_results[1], result_files['segm'])
    elif isinstance(results[0], np.ndarray):
        json_results = proposal2json(dataset, results)
        result_files['proposal'] = '{}.{}.json'.format(out_file, 'proposal')
        mmcv.dump(json_results, result_files['proposal'])
    else:
        raise TypeError('invalid type of results')
    return result_files

def single_gpu_test(model, data_loader, show=False, evaluator=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))                    
        
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if i == 0:
            temp_result_files = cached_results2json(copy.deepcopy(dataset), copy.deepcopy(results), 'temp_results.pkl')
            anns = json.load(open(temp_result_files['bbox']))
            evaluator.add(anns)
            from sotabencheval.object_detection.utils import get_coco_metrics
            print(evaluator.batch_hash)
            print(evaluator.cache_exists)
            if evaluator.cache_exists:
                return results, True
        
        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
            
    return results, False


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results

def evaluate_model(model_name, paper_arxiv_id, weights_url, weights_name, paper_results, config):
    print('---')
    print('Now Evaluating %s' % model_name)
    evaluator = COCOEvaluator(
    root='./.data/vision/coco',
    model_name=model_name,
    paper_arxiv_id=paper_arxiv_id,
    paper_results=paper_results)

    out = 'results.pkl'
    launcher = 'none'

    if out is not None and not out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(config)
    cfg.data.test['ann_file'] = './.data/vision/coco/annotations/instances_val2017.json'
    cfg.data.test['img_prefix'] = './.data/vision/coco/val2017/'

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
        
    local_checkpoint, _ = urllib.request.urlretrieve(
        weights_url,
         '%s/.cache/torch/%s' % (str(Path.home()), weights_name))

    print(local_checkpoint)
    
    # '/home/ubuntu/GCNet/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth'
    checkpoint = load_checkpoint(model, local_checkpoint, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    evaluator.reset_time()
        
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs, cache_exists = single_gpu_test(model, data_loader, False, evaluator)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    if cache_exists:
        print('Cache exists: %s' % (evaluator.batch_hash))
        evaluator.save()
    
    else:
        from mmdet.core import results2json

        rank, _ = get_dist_info()
        if out and rank == 0:
            print('\nwriting results to {}'.format(out))
            mmcv.dump(outputs, out)
            eval_types = ['bbox']
            if eval_types:
                print('Starting evaluate {}'.format(' and '.join(eval_types)))
                if eval_types == ['proposal_fast']:
                    result_file = out
                else:
                    if not isinstance(outputs[0], dict):
                        result_files = results2json(dataset, outputs, out)
                    else:
                        for name in outputs[0]:
                            print('\nEvaluating {}'.format(name))
                            outputs_ = [out[name] for out in outputs]
                            result_file = out + '.{}'.format(name)
                            result_files = results2json(dataset, outputs_,
                                                        result_file)
        anns = json.load(open(result_files['bbox']))
        evaluator.detections = []
        evaluator.add(anns)
        evaluator.save()

model_configs = []


# libra r-cnn

model_configs.append(
    {'model_name': 'Libra R-CNN (Faster R-CNN, ResNet-50-FPN)', 
     'paper_arxiv_id': '1904.02701',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/libra_rcnn/libra_faster_rcnn_r50_fpn_1x_20190610-bf0ea559.pth',
     'weights_name': 'libra_faster_rcnn_r50_fpn_1x_20190610-bf0ea559.pth',
     'config': './configs/libra_rcnn/libra_faster_rcnn_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Libra R-CNN (Faster R-CNN, ResNet-101-FPN)', 
     'paper_arxiv_id': '1904.02701',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/libra_rcnn/libra_faster_rcnn_r101_fpn_1x_20190525-94e94051.pth',
     'weights_name': 'libra_faster_rcnn_r101_fpn_1x_20190525-94e94051.pth',
     'config': './configs/libra_rcnn/libra_faster_rcnn_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Libra R-CNN (Faster R-CNN, ResNeXt-101 64x4d-FPN)', 
     'paper_arxiv_id': '1904.02701',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x_20190525-359c134a.pth',
     'weights_name': 'libra_faster_rcnn_x101_64x4d_fpn_1x_20190525-359c134a.pth',
     'config': './configs/libra_rcnn/libra_faster_rcnn_x101_64x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Libra R-CNN (RetinaNet, ResNet-50-FPN)', 
     'paper_arxiv_id': '1904.02701',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/libra_rcnn/libra_retinanet_r50_fpn_1x_20190525-ead2a6bb.pth',
     'weights_name': 'libra_retinanet_r50_fpn_1x_20190525-ead2a6bb.pth',
     'config': './configs/libra_rcnn/libra_retinanet_r50_fpn_1x.py'}
)

# mask scoring r-cnn

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNet-50-FPN, LR 1x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth',
     'weights_name': 'ms_rcnn_r50_caffe_fpn_1x_20190624-619934b5.pth',
     'config': './configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNet-50-FPN, LR 2x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r50_caffe_fpn_2x_20190525-a07be31e.pth',
     'weights_name': 'ms_rcnn_r50_caffe_fpn_2x_20190525-a07be31e.pth',
     'config': './configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNet-101-FPN, LR 1x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_r101_caffe_fpn_1x_20190624-677a5548.pth',
     'weights_name': 'ms_rcnn_r101_caffe_fpn_1x_20190624-677a5548.pth',
     'config': './configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNet-101-FPN, LR 2x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_r101_caffe_fpn_2x_20190525-4aee1528.pth',
     'weights_name': 'ms_rcnn_r101_caffe_fpn_2x_20190525-4aee1528.pth',
     'config': './configs/ms_rcnn/ms_rcnn_r101_caffe_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNeXt-101 64x4d-FPN, LR 1x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/ms-rcnn/ms_rcnn_x101_64x4d_fpn_1x_20190628-dec32bda.pth',
     'weights_name': 'ms_rcnn_x101_64x4d_fpn_1x_20190628-dec32bda.pth',
     'config': './configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask Scoring R-CNN (ResNeXt-101 64x4d-FPN, LR 2x)', 
     'paper_arxiv_id': '1903.00241',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ms-rcnn/ms_rcnn_x101_64x4d_fpn_2x_20190525-c044c25a.pth',
     'weights_name': 'ms_rcnn_x101_64x4d_fpn_2x_20190525-c044c25a.pth',
     'config': './configs/ms_rcnn/ms_rcnn_x101_64x4d_fpn_1x.py'}
)

# Weight Standardization

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNet-50-FPN, GN+WS, LR 1x)', 
     'paper_arxiv_id': '1903.10520',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/faster_rcnn_r50_fpn_gn_ws_1x_20190418-935d00b6.pth',
     'weights_name': 'faster_rcnn_r50_fpn_gn_ws_1x_20190418-935d00b6.pth',
     'config': './configs/gn+ws/faster_rcnn_r50_fpn_gn_ws_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, GN+WS, LR 2x)', 
     'paper_arxiv_id': '1903.10520',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r50_fpn_gn_ws_2x_20190419-9ec97bbb.pth',
     'weights_name': 'mask_rcnn_r50_fpn_gn_ws_2x_20190419-9ec97bbb.pth',
     'config': './configs/gn+ws/mask_rcnn_r50_fpn_gn_ws_2x.py'}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNet-50-FPN, GN+WS, LR 20-23-24e)', 
     'paper_arxiv_id': '1903.10520',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_r50_fpn_gn_ws_20_23_24e_20190425-1d9e499e.pth',
     'weights_name': 'mask_rcnn_r50_fpn_gn_ws_20_23_24e_20190425-1d9e499e.pth',
     'config': './configs/gn+ws/mask_rcnn_r50_fpn_gn_ws_20_23_24e.py'}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (ResNeXt-101 32x4d-FPN, GN+WS, LR 1x)', 
     'paper_arxiv_id': '1903.10520',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ws/mask_rcnn_x101_32x4d_fpn_gn_ws_20_23_24e_20190425-1ff3e5b2.pth',
     'weights_name': 'mask_rcnn_x101_32x4d_fpn_gn_ws_20_23_24e_20190425-1ff3e5b2.pth',
     'config': './configs/gn+ws/mask_rcnn_x101_32x4d_fpn_gn_ws_2x.py'}
)


# SSD

model_configs.append(
    {'model_name': 'SSD300', 
     'paper_arxiv_id': '1512.02325',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth',
     'weights_name': 'ssd300_coco_vgg16_caffe_120e_20181221-84d7110b.pth',
     'config': './configs/ssd300_coco.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'SSD512', 
     'paper_arxiv_id': '1512.02325',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth',
     'weights_name': 'ssd512_coco_vgg16_caffe_120e_20181221-d48b0be8.pth',
     'config': './configs/ssd512_coco.py',
    'paper_results': None}
)

## Hybrid Task Cascade (HTC)

model_configs.append(
    {'model_name': 'HTC (ResNet-50-FPN, 1x LR)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_1x_20190408-878c1712.pth',
     'weights_name': 'htc_r50_fpn_1x_20190408-878c1712.pth',
     'config': './configs/htc/htc_r50_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'HTC (ResNet-50-FPN, 20e LR)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r50_fpn_20e_20190408-c03b7015.pth',
     'weights_name': 'htc_r50_fpn_20e_20190408-c03b7015.pth',
     'config': './configs/htc/htc_r50_fpn_20e.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'HTC (ResNet-101-FPN)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_r101_fpn_20e_20190408-a2e586db.pth',
     'weights_name': 'htc_r101_fpn_20e_20190408-a2e586db.pth',
     'config': './configs/htc/htc_r101_fpn_20e.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'HTC (ResNeXt-101 32x4d-FPN)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_32x4d_fpn_20e_20190408-9eae4d0b.pth',
     'weights_name': 'htc_x101_32x4d_fpn_20e_20190408-9eae4d0b.pth',
     'config': './configs/htc/htc_x101_32x4d_fpn_20e_16gpu.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'HTC (ResNeXt-101 64x4d-FPN)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_x101_64x4d_fpn_20e_20190408-497f2561.pth',
     'weights_name': 'htc_x101_64x4d_fpn_20e_20190408-497f2561.pth',
     'config': './configs/htc/htc_x101_64x4d_fpn_20e_16gpu.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'HTC (ResNeXt-101 64x4d-FPN, DCN, multi-scale)', 
     'paper_arxiv_id': '1901.07518',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
     'weights_name': 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
     'config': './configs/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e.py',
    'paper_results': None}
)

## Cascade Mask R-CNN

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNet-50-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth',
     'weights_name': 'cascade_mask_rcnn_r50_fpn_1x_20181123-88b170c9.pth',
     'config': './configs/cascade_mask_rcnn_r50_fpn_1x.py',
    'paper_results': {'box AP': 0.413, 'AP50': 0.594, 'AP75': 0.453, 'APS': 0.232, 'APM': 0.438, 'APL': 0.558}}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNet-50-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r50_fpn_20e_20181123-6e0c9713.pth',
     'weights_name': 'cascade_mask_rcnn_r50_fpn_20e_20181123-6e0c9713.pth',
     'config': './configs/cascade_mask_rcnn_r50_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNet-101-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_1x_20181129-64f00602.pth',
     'weights_name': 'cascade_mask_rcnn_r101_fpn_1x_20181129-64f00602.pth',
     'config': './configs/cascade_mask_rcnn_r101_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNet-101-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_r101_fpn_20e_20181129-cb85151d.pth',
     'weights_name': 'cascade_mask_rcnn_r101_fpn_20e_20181129-cb85151d.pth',
     'config': './configs/cascade_mask_rcnn_r101_fpn_1x.py',
    'paper_results': {'box AP': 0.433, 'AP50': 0.617, 'AP75': 0.472, 'APS': 0.242, 'APM': 0.463, 'APL': 0.582}}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNeXt-101 32x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_1x_20181218-1d944c89.pth',
     'weights_name': 'cascade_mask_rcnn_x101_32x4d_fpn_1x_20181218-1d944c89.pth',
     'config': './configs/cascade_mask_rcnn_x101_32x4d_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNeXt-101 32x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_32x4d_fpn_20e_20181218-761a3473.pth',
     'weights_name': 'cascade_mask_rcnn_x101_32x4d_fpn_20e_20181218-761a3473.pth',
     'config': './configs/cascade_mask_rcnn_x101_32x4d_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNeXt-101 64x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_1x_20190501-827e0a70.pth',
     'weights_name': 'cascade_mask_rcnn_x101_64x4d_fpn_1x_20190501-827e0a70.pth',
     'config': './configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py',
    'paper_results': None}
)

model_configs.append(
    {'model_name': 'Cascade Mask R-CNN (ResNeXt-101 64x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_mask_rcnn_x101_64x4d_fpn_20e_20181218-630773a7.pth',
     'weights_name': 'htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth',
     'config': './configs/cascade_mask_rcnn_x101_64x4d_fpn_1x.py',
    'paper_results': {'box AP': 0.458, 'AP50': 0.641, 'AP75': 0.503, 'APS': 0.272, 'APM': 0.495, 'APL': 0.601}}
)

## Cascade R-CNN

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNet-50-FPN+)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth',
     'weights_name': 'cascade_rcnn_r50_fpn_1x_20190501-3b6211ab.pth',
     'config': './configs/cascade_rcnn_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNet-50-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth',
     'weights_name': 'cascade_rcnn_r50_fpn_20e_20181123-db483a09.pth',
     'config': './configs/cascade_rcnn_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNet-101-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth',
     'weights_name': 'cascade_rcnn_r101_fpn_1x_20181129-d64ebac7.pth',
     'config': './configs/cascade_rcnn_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNet-101-FPN+, cascade)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth',
     'weights_name': 'cascade_rcnn_r101_fpn_20e_20181129-b46dcede.pth',
     'config': './configs/cascade_rcnn_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNeXt-101 32x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth',
     'weights_name': 'cascade_rcnn_x101_32x4d_fpn_1x_20190501-af628be5.pth',
     'config': './configs/cascade_rcnn_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNeXt-101 32x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth',
     'weights_name': 'cascade_rcnn_x101_32x4d_fpn_2x_20181218-28f73c4c.pth',
     'config': './configs/cascade_rcnn_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNeXt-101 64x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth',
     'weights_name': 'cascade_rcnn_x101_64x4d_fpn_1x_20181218-e2dc376a.pth',
     'config': './configs/cascade_rcnn_x101_64x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Cascade R-CNN (ResNeXt-101 64x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1906.09756',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth',
     'weights_name': 'cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth',
     'config': './configs/cascade_rcnn_x101_64x4d_fpn_1x.py'}
)

## RetinaNet

model_configs.append(
    {'model_name': 'RetinaNet (ResNet-50-FPN+)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r50_fpn_1x_20181125-7b0c2548.pth',
     'weights_name': 'retinanet_r50_fpn_1x_20181125-7b0c2548.pth',
     'config': './configs/retinanet_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNet-50-FPN, 20e LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/retinanet_r50_fpn_2x_20190616-75574209.pth',
     'weights_name': 'retinanet_r50_fpn_2x_20190616-75574209.pth',
     'config': './configs/retinanet_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNet-101-FPN, 1x LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_1x_20181129-f016f384.pth',
     'weights_name': 'retinanet_r101_fpn_1x_20181129-f016f384.pth',
     'config': './configs/retinanet_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNet-101-FPN+, cascade)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_r101_fpn_2x_20181129-72c14526.pth',
     'weights_name': 'retinanet_r101_fpn_2x_20181129-72c14526.pth',
     'config': './configs/retinanet_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNeXt-101 32x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_1x_20190501-967812ba.pth',
     'weights_name': 'retinanet_x101_32x4d_fpn_1x_20190501-967812ba.pth',
     'config': './configs/retinanet_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNeXt-101 32x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_32x4d_fpn_2x_20181218-8596452d.pth',
     'weights_name': 'retinanet_x101_32x4d_fpn_2x_20181218-8596452d.pth',
     'config': './configs/retinanet_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNeXt-101 64x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_1x_20181218-a0a22662.pth',
     'weights_name': 'retinanet_x101_64x4d_fpn_1x_20181218-a0a22662.pth',
     'config': './configs/retinanet_x101_64x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'RetinaNet (ResNeXt-101 64x4d-FPN, 20e LR)', 
     'paper_arxiv_id': '1708.02002',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/retinanet_x101_64x4d_fpn_2x_20181218-5e88d045.pth',
     'weights_name': 'retinanet_x101_64x4d_fpn_2x_20181218-5e88d045.pth',
     'config': './configs/retinanet_x101_64x4d_fpn_1x.py'}
)

## Faster R-CNN

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNet-50-FPN, 1x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth',
     'weights_name': 'faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth',
     'config': './configs/faster_rcnn_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNet-50-FPN, 2x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r50_fpn_2x_20181010-443129e1.pth',
     'weights_name': 'faster_rcnn_r50_fpn_2x_20181010-443129e1.pth',
     'config': './configs/faster_rcnn_r50_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNet-101-FPN, 1x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_1x_20181129-d1468807.pth',
     'weights_name': 'faster_rcnn_r101_fpn_1x_20181129-d1468807.pth',
     'config': './configs/faster_rcnn_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNet-101-FPN, 2x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth',
     'weights_name': 'faster_rcnn_r101_fpn_2x_20181129-73e7ade7.pth',
     'config': './configs/faster_rcnn_r101_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNeXt-101 32x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth',
     'weights_name': 'faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth',
     'config': './configs/faster_rcnn_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNeXt-101 32x4d-FPN, 2x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_32x4d_fpn_2x_20181218-0ed58946.pth',
     'weights_name': 'faster_rcnn_x101_32x4d_fpn_2x_20181218-0ed58946.pth',
     'config': './configs/faster_rcnn_x101_32x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNeXt-101 64x4d-FPN, 1x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth',
     'weights_name': 'faster_rcnn_x101_64x4d_fpn_1x_20181218-c9c69c8f.pth',
     'config': './configs/faster_rcnn_x101_64x4d_fpn_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (ResNeXt-101 64x4d-FPN, 2x LR)', 
     'paper_arxiv_id': '1506.01497',
     'weights_url': 'https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth',
     'weights_name': 'faster_rcnn_x101_64x4d_fpn_2x_20181218-fe94f9b8.pth',
     'config': './configs/faster_rcnn_x101_64x4d_fpn_1x.py'}
)

## HRNet

model_configs.append(
    {'model_name': 'Faster R-CNN (HRNetV2p-W18, 1x LR)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w18_1x_20190522-e368c387.pth',
     'weights_name': 'faster_rcnn_hrnetv2p_w18_1x_20190522-e368c387.pth',
     'config': './configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py',
     'paper_results': {'box AP': 0.362, 'AP50': 0.573, 'AP75': 0.393, 'APS': 0.207, 'APM': 0.39, 'APL': 0.468}}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (HRNetV2p-W18)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w18_2x_20190810-9c8615d5.pth',
     'weights_name': 'faster_rcnn_hrnetv2p_w18_2x_20190810-9c8615d5.pth',
     'config': './configs/hrnet/faster_rcnn_hrnetv2p_w18_1x.py'}
)

model_configs.append(
    {'model_name': 'Faster R-CNN (HRNetV2p-W32, 1x LR)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth',
     'weights_name': 'faster_rcnn_hrnetv2p_w32_1x_20190522-d22f1fef.pth',
     'config': './configs/hrnet/faster_rcnn_hrnetv2p_w32_1x.py',
     'paper_results': {'box AP': 0.396, 'AP50': 0.61, 'AP75': 0.433, 'APS': 0.237, 'APM': 0.425, 'APL': 0.505}} 
)

model_configs.append(
    {'model_name': 'Faster R-CNN (HRNetV2p-W32)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/faster_rcnn_hrnetv2p_w32_2x_20190810-24e8912a.pth',
     'weights_name': 'faster_rcnn_hrnetv2p_w32_2x_20190810-24e8912a.pth',
     'config': './configs/hrnet/faster_rcnn_hrnetv2p_w32_1x.py'} 
)

#maskrcnn

model_configs.append(
    {'model_name': 'Mask R-CNN (HRNetV2p-W18, 1x LR)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w18_1x_20190522-c8ad459f.pth',
     'weights_name': 'mask_rcnn_hrnetv2p_w18_1x_20190522-c8ad459f.pth',
     'config': './configs/hrnet/mask_rcnn_hrnetv2p_w18_1x.py',
     'paper_results': {'box AP': 0.371, 'APS': 0.219, 'APM': 0.395, 'APL': 0.479}}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (HRNetV2p-W18)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w18_2x_20190810-1e4747eb.pth',
     'weights_name': 'mask_rcnn_hrnetv2p_w18_2x_20190810-1e4747eb.pth',
     'config': './configs/hrnet/mask_rcnn_hrnetv2p_w18_1x.py'}
)

model_configs.append(
    {'model_name': 'Mask R-CNN (HRNetV2p-W32, 1x LR)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w32_1x_20190522-374aaa00.pth',
     'weights_name': 'mask_rcnn_hrnetv2p_w32_1x_20190522-374aaa00.pth',
     'config': './configs/hrnet/mask_rcnn_hrnetv2p_w32_1x.py',
     'paper_results': {'box AP': 0.409, 'APS': 0.245, 'APM': 0.439, 'APL': 0.522}}
) 

model_configs.append(
    {'model_name': 'Mask R-CNN (HRNetV2p-W32)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/mask_rcnn_hrnetv2p_w32_2x_20190810-773eca75.pth',
     'weights_name': 'mask_rcnn_hrnetv2p_w32_2x_20190810-773eca75.pth',
     'config': './configs/hrnet/mask_rcnn_hrnetv2p_w32_1x.py'}
) 

#cascadercnn

model_configs.append(
    {'model_name': 'Cascade R-CNN (HRNetV2p-W32)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
     'weights_name': 'cascade_rcnn_hrnetv2p_w32_20e_20190522-55bec4ee.pth',
     'config': './configs/hrnet/cascade_rcnn_hrnetv2p_w32_20e.py'}
)

#cascademaskkrcnn

model_configs.append( 
    {'model_name': 'Mask R-CNN (HRNetV2p-W32, cascade)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth',
     'weights_name': 'cascade_mask_rcnn_hrnetv2p_w32_20e_20190810-76f61cd0.pth',
     'config': './configs/hrnet/cascade_mask_rcnn_hrnetv2p_w32_20e.py'}
)

#htc

model_configs.append(
    {'model_name': 'HTC (HRNetV2p-W32)', 
     'paper_arxiv_id': '1904.04514',
     'weights_url': 'https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection/models/hrnet/htc_hrnetv2p_w32_20e_20190810-82f9ef5a.pth',
     'weights_name': 'htc_hrnetv2p_w32_20e_20190810-82f9ef5a.pth',
     'config': './configs/hrnet/htc_hrnetv2p_w32_20e.py'}
)


import torch.distributed as dist
dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    
for model_config in model_configs:
    if 'paper_results' not in model_config:
        model_config['paper_results'] = None
    
    evaluate_model(model_name=model_config['model_name'], 
                   paper_arxiv_id=model_config['paper_arxiv_id'],
                   weights_url=model_config['weights_url'],
                   weights_name=model_config['weights_name'],
                   paper_results=model_config['paper_results'],
                   config=model_config['config'])
