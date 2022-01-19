import argparse
import os
import warnings

import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
import os.path as osp
from mmcls.apis import multi_gpu_test, single_gpu_test
from mmcls.datasets import build_dataloader, build_dataset
from mmcls.models import build_classifier
from custom.utils.cm_funcs.confusion_matrix import plot_cm
from mmcls.datasets.scaled_mnist import read_image_file,read_label_file
from mmcls.datasets.utils import  rm_suffix
import ipdb
# TODO import `wrap_fp16_model` from mmcv and delete them from mmcls
try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('wrap_fp16_model from mmcls will be deprecated.'
                  'Please install mmcv>=1.1.4.')
    from mmcls.core import wrap_fp16_model


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')

    parser.add_argument('--ratio-w', help='ratio-w',default='1.0')
    parser.add_argument('--ratio-h', help='ratio-h',default='1.0')




    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., '
        '"accuracy", "precision", "recall", "f1_score", "support" for single '
        'label dataset, and "mAP", "CP", "CR", "CF1", "OP", "OR", "OF1" for '
        'multi-label dataset')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--metric-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be parsed as a dict metric_options for dataset.evaluate()'
        ' function.')
    parser.add_argument(
        '--show-options',
        nargs='+',
        action=DictAction,
        help='custom options for show_result. key-value pair in xxx=yyy.'
        'Check available options in `model.show_result`.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)


    if args.ratio_w is not None:
        cfg.data.val.ratio_w= args.ratio_w
        cfg.data.test.ratio_w= args.ratio_w

    if args.ratio_w is not None:
        cfg.data.val.ratio_h= args.ratio_h
        cfg.data.test.ratio_h= args.ratio_h

    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        round_up=False)

    # build the model and load checkpoint
    model = build_classifier(cfg.model)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        CLASSES = checkpoint['meta']['CLASSES']
    else:
        from mmcls.datasets import ImageNet
        warnings.simplefilter('once')
        warnings.warn('Class names are not saved in the checkpoint\'s '
                      'meta data, use imagenet by default.')
        CLASSES = ImageNet.CLASSES
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        model.CLASSES = CLASSES
        show_kwargs = {} if args.show_options is None else args.show_options
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  **show_kwargs)
    # outputs :a list containing 10000 ndarrays, every ndarray size is (10,) such as:
    # array([8.67471826e-05, 1.17470045e-04, 2.08544172e-03, 2.25347801e-04,
    #        1.19258320e-05, 2.03944765e-05, 4.21342884e-06, 9.97160912e-01,
    #        1.35557348e-04, 1.51961009e-04], dtype=float32)
    # outputs is prediction scores

    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()

    if rank == 0:
        if args.metrics != None:
            results= dataset.evaluate(outputs, args.metrics,
                                       args.metric_options)
            for k, v in results.items():
                print(f'\n{k} : {v:.2f}')
            scores = np.vstack(outputs)
            # pred_score = np.max(scores, axis=1)
            pred_labels = np.argmax(scores, axis=1)
            # pred_class = [CLASSES[lb] for lb in pred_labels]
            resources = {
                'test_image_file':
                    ('t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
                'test_label_file':
                    ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
            }

            data_prefix='/media/n/SanDiskSSD/HardDisk/data/MNIST/raw'
            # os.path = osp
            test_image_file = osp.join(
                data_prefix, rm_suffix(resources['test_image_file'][0]))
            test_label_file = osp.join(
                data_prefix, rm_suffix(resources['test_label_file'][0]))
            test_set = (read_image_file(test_image_file),
                        read_label_file(test_label_file))
            # test_set[0].shape    torch.Size([10000, 28, 28])
            # test_set[1].shape     torch.Size([10000])
            imgs, gt_labels = test_set
            plot_cm(gt_labels,pred_labels)




        else:   #  plot confusion matrix, if no metrics
            warnings.warn('Evaluation metrics are not specified.')
            scores = np.vstack(outputs)
            pred_score = np.max(scores, axis=1)
            pred_label = np.argmax(scores, axis=1)
            pred_class = [CLASSES[lb] for lb in pred_label]
            results = {
                'pred_score': pred_score,
                'pred_label': pred_label,
                'pred_class': pred_class
            }
            if not args.out:
                print('\nthe predicted result for the first element is '
                      f'pred_score = {pred_score[0]:.2f}, '
                      f'pred_label = {pred_label[0]} '
                      f'and pred_class = {pred_class[0]}. '
                      'Specify --out to save all results to files.')
    if args.out and rank == 0:
        print(f'\nwriting results to {args.out}')
        mmcv.dump(results, args.out)


if __name__ == '__main__':
    main()
