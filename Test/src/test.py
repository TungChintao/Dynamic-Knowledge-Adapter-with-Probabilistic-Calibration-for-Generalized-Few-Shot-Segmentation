import os
import cv2
import time
import numpy as np
import argparse
from typing import Tuple
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP

from .aux_classifier import Classifier

from .model.pspnet_aux import get_model
# from .model.unet_aux import get_model

from .utils import (fast_intersection_and_union, setup_seed, ensure_dir,
                    resume_random_state, find_free_port, setup, cleanup, get_cfg, adjust_background_to_highest_class)

from .dataset.data import get_val_loader
from .dataset.classes import classId2className, update_novel_classes
from .meter import AverageMeter, intersectionAndUnionGPU
from .dataset.data_oem import BaseData

import ttach as tta
from ttach.base import Merger, Compose
from .process import remove_small_objects_per_class, open_close_per_class


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    return get_cfg(parser)


def valid_mean(t, valid_pixels, dim):
    s = (valid_pixels * t).sum(dim=dim)
    return s / (valid_pixels.sum(dim=dim) + 1e-10)


def main_worker(rank: int, world_size: int, args: argparse.Namespace) -> None:
    print(f"==> Running evaluation script")
    setup(args, rank, world_size)
    setup_seed(args.manual_seed)

    # ========== Data  ==========
    val_loader = get_val_loader(args)

    # ========== Model  ==========
    print("=> Creating the model")
    model = get_model(args).to(rank)
    # for i in model.state_dict():
    #    print(i)
    # print('================')
    if args.pretrained is not None:
        assert os.path.isfile(args.pretrained), args.pretrained
        checkpoint = torch.load(args.pretrained)
        # parameter_names = checkpoint['state_dict'].keys()
        # for name in parameter_names:
        #    print(name)
        # print('================================================')
        # return 0
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> Loaded weight '{}'".format(args.pretrained))
    else:
        print("=> Not loading anything")
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    # ========== Test  ==========
    validate(args=args, val_loader=val_loader, model=model)
    cleanup()


def validate(args: argparse.Namespace, val_loader: torch.utils.data.DataLoader, model: DDP) -> Tuple[
    torch.tensor, torch.tensor]:
    print('\n==> Start testing...', flush=True)
    base_novel_classes = classId2className
    random_state = setup_seed(args.manual_seed, return_old_state=True)
    device = torch.device('cuda:{}'.format(dist.get_rank()))
    model.eval()

    c = model.module.bottleneck_dim
    h = int(args.image_size / 8)
    w = int(args.image_size / 8)

    # =========== Test-Time Augmentation Setting ========= #
    # tta_transforms = tta.aliases.d4_transform() # 2*4=8
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Rotate90(angles=[0, 90, 180, 270]),
        # tta.Scale(scales=[0.5, 1, 1.25]),
    ])  # 2*2*4=16
    merge_mode = 'mean'

    # ========== Perform the runs  ==========
    # The order of classes in the following tensors is the same as the order of classifier (novels at last)
    cls_intersection = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_union = torch.zeros(args.num_classes_tr + args.num_classes_val)
    cls_target = torch.zeros(args.num_classes_tr + args.num_classes_val)

    runtime = 0

    # get support images and labels
    spprt_imgs, s_label = val_loader.dataset.get_support_set()  # Get the support set
    spprt_imgs = spprt_imgs.to(device, non_blocking=True)
    s_label = s_label.to(device, non_blocking=True)

    features_s, gt_s = None, None
    with torch.no_grad():
        features_s = model.module.LRANet(model.module.extract_features(spprt_imgs)).detach().view(
            (args.num_classes_val, args.shot, c, h, w))
        gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))

    # ========== Begin: fine tune the aux network ================

    # set meter
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    # temp cls
    aux_cls = nn.Conv2d(args.bottleneck_dim, 4, kernel_size=1, bias=True).cuda()
    # print(aux_cls.weight.shape)
    ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
    ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_novel_classes, shot, 1, h, w]

    # Computing prototypes
    novel_weight = torch.zeros((features_s.size(2), 4), device=features_s.device)
    for cls in range(8, 12):
        novel_mask = (ds_gt_s == cls)
        novel_prototype = valid_mean(features_s, novel_mask, (0, 1, 3, 4))  # [c,]
        novel_weight[:, cls - 8] = novel_prototype

    novel_weight /= novel_weight.norm(dim=0).unsqueeze(0) + 1e-10
    assert torch.isnan(novel_weight).sum() == 0, novel_weight
    novel_weight = novel_weight.transpose(0, 1).unsqueeze(2).unsqueeze(2)
    novel_bias = torch.zeros((4,), device=features_s.device)
    # print(novel_weight.shape)
    aux_cls.weight = nn.Parameter(novel_weight)
    aux_cls.bias = nn.Parameter(novel_bias)

    # set config
    aux_params = [v for k, v in model.named_parameters() if 'aux' in k]
    aux_optimizer = torch.optim.SGD([{"params": aux_cls.parameters()},
                                     {"params": aux_params, "lr": args.emd_lr}], lr=args.aux_lr, momentum=args.momentum,
                                     weight_decay=args.weight_decay)

    aux_criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    # temp_label = [(label > 0) * 1 for label in s_label]
    temp_label = torch.where((s_label >= 8) & (s_label <= 11), s_label - 8, torch.tensor(255).cuda())

    for iteration in range(args.aux_iter):

        # fine tune
        for img, target in zip(spprt_imgs, temp_label):
            # print(np.unique(target.cpu()).tolist())
            # continue
            img = img.unsqueeze(0)
            target = target.unsqueeze(0)
            # print(img.shape)
            # print(target.shape)
            output_f = model.module.extract_features(img).detach()
            output_aux_f = model.module.LRANet(output_f)

            output = aux_cls(output_aux_f)
            output = F.interpolate(output, size=(img.size()[2], img.size()[3]), mode='bilinear', align_corners=True)
            # print(output.shape)
            aux_loss = aux_criterion(output, target.long())

            aux_optimizer.zero_grad()
            aux_loss.backward()
            aux_optimizer.step()

            # print(sum(sum(sum(sum(model.module.aux_d.weight)))))
            output = output.max(1)[1]
            intersection, union, _ = intersectionAndUnionGPU(output, target, 4, args.ignore_label)
            intersection, union, _ = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union)

            n = img.size(0)
            loss_meter.update(aux_loss.item(), n)

        # evaluation
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        mIoU = np.mean(iou_class)

        print('Fine tune loss/mIoU at iter [{}/{}]: {:.4f}/{:.4f}.'.format(iteration, args.aux_iter, loss_meter.val,
                                                                           mIoU))

    # ========== End: fine tune the aux network ================

    # ========== Relabeling ====================================
    base_weight = model.module.classifier.weight.detach().clone()
    # print(base_weight)
    base_weight = base_weight.permute(*torch.arange(base_weight.ndim - 1, -1, -1))
    base_bias = None  # model.module.classifier.bias.detach().clone()

    e_weight = model.module.aux_e.weight.detach().clone()
    e_bias = model.module.aux_e.bias.detach().clone()
    m_weight = model.module.aux_m.weight.detach().clone()
    m_bias = model.module.aux_m.bias.detach().clone()
    d_weight = model.module.aux_d.weight.detach().clone()
    d_bias = model.module.aux_d.bias.detach().clone()

    rl_classifier = Classifier(args, base_weight, base_bias, e_weight, e_bias, m_weight, m_bias, d_weight, d_bias, n_tasks=1)
    novel_weight = aux_cls.weight.detach().clone()
    novel_weight = novel_weight.permute(*torch.arange(novel_weight.ndim - 1, -1, -1))
    novel_bias = aux_cls.bias.detach().clone()
    rl_classifier.init_novel_cls(novel_weight, novel_bias)


    with torch.no_grad():
        features_s = model.module.extract_features(spprt_imgs).detach().view(
            (args.num_classes_val, args.shot, c, h, w))
        gt_s = s_label.view((args.num_classes_val, args.shot, args.image_size, args.image_size))


    rl_loader = BaseData(args)
    print("Re-labeling for base...")
    rl_episodes = len(rl_loader)  # The number of images in the query set
    for _ in tqdm(range(rl_episodes), leave=True):
        with torch.no_grad():
            try:
                loader_output = next(iter_loader)
            except (UnboundLocalError, StopIteration):
                iter_loader = iter(rl_loader)
                loader_output = next(iter_loader)
            rl_img  = loader_output

        rl_img = rl_img.to(device, non_blocking=True).unsqueeze(0)

        features_rl = model.module.extract_features(rl_img).detach().unsqueeze(1)
        # print(features_rl.shape)
        rl_classifier.optimize(features_s, features_rl, gt_s)
        # print(features_rl.shape)

    #print("Re-labeling for novel...")
    #for _, rl_img in tqdm(enumerate(spprt_imgs), total=len(spprt_imgs), leave=True):
    #    rl_img = rl_img.unsqueeze(0)
    #    features_rl = model.module.LRANet(model.module.extract_features(rl_img)).detach().unsqueeze(1)
        # print(features_rl.shape)
    #    rl_classifier.optimize(features_s, features_rl, gt_s)
    # ==========================================================
    print("Testing...")

    # ==========================================================

    nb_episodes = len(val_loader)  # The number of images in the query set
    for _ in tqdm(range(nb_episodes), leave=True):
        t0 = time.time()
        with torch.no_grad():
            try:
                loader_output = next(iter_loader)
            except (UnboundLocalError, StopIteration):
                iter_loader = iter(val_loader)
                loader_output = next(iter_loader)

            if len(loader_output) == 3:
                qry_img, q_label, image_name = loader_output
            if len(loader_output) == 2:
                qry_img, image_name = loader_output
                q_label = None

            qry_img = qry_img.to(device, non_blocking=True)
            features_q = model.module.extract_features(qry_img).detach().unsqueeze(1)
            if q_label is not None:
                q_label = q_label.to(device, non_blocking=True)
                gt_q = q_label.unsqueeze(1)

        # =========== Initialize the classifier and run the method ===============
        base_weight = rl_classifier.base_weight.detach().clone()
        base_weight = base_weight.unsqueeze(0)
        base_bias = rl_classifier.base_bias.detach().clone()

        e_weight = rl_classifier.aux_e_weight.detach().clone()
        e_bias = rl_classifier.aux_e_bias.detach().clone()
        m_weight = rl_classifier.aux_m_weight.detach().clone()
        m_bias = rl_classifier.aux_m_bias.detach().clone()
        d_weight = rl_classifier.aux_d_weight.detach().clone()
        d_bias = rl_classifier.aux_d_bias.detach().clone()

        classifier = Classifier(args, base_weight, base_bias, e_weight, e_bias, m_weight, m_bias, d_weight, d_bias, n_tasks=features_q.size(0))
        # classifier.init_prototypes(features_s, gt_s)
        # print(classifier.base_bias.shape)

        novel_weight = rl_classifier.novel_weight.detach().clone()
        novel_bias = rl_classifier.novel_bias.detach().clone()
        
        #classifier.init_novel_cls(novel_weight, novel_bias)
        classifier.init_novel(novel_weight, novel_bias)

        classifier.optimize(features_s, features_q, gt_s)

        runtime += time.time() - t0

        # =========== Perform inference and compute metrics ===============
        logits = classifier.get_logits(features_q).detach()
        probas = classifier.get_probas(logits)

        if q_label is not None:
            if args.save_pred_maps is True:  # Save predictions in '.png' file format
                # ensure_dir('results/targets')
                ensure_dir('results/preds')
                n_task, shots, num_classes, h, w = probas.size()
                H, W = gt_q.size()[-2:]
                if (h, w) != (H, W):
                    probas = F.interpolate(probas.view(n_task * shots, num_classes, h, w),
                                           size=(H, W), mode='bilinear', align_corners=True).view(n_task, shots,
                                                                                                  num_classes, H, W)
                pred = probas.argmax(2)  # [n_query, shot, H, W]
                pred = np.array(pred.squeeze().cpu(), np.uint8)
                # target = np.array(gt_q.squeeze().cpu(), np.uint8)
                fname = ''.join(image_name)
                # cv2.imwrite(os.path.join('results/targets', fname + '.png'), target)
                cv2.imwrite(os.path.join('results/preds', fname + '.png'), pred)

            intersection, union, target = fast_intersection_and_union(probas, gt_q)  # [batch_size_val, 1, num_classes]
            intersection, union, target = intersection.squeeze(1).cpu(), union.squeeze(1).cpu(), target.squeeze(1).cpu()
            cls_intersection += intersection.sum(0)
            cls_union += union.sum(0)
            cls_target += target.sum(0)
        else:
            if args.save_pred_maps is True:  # Save predictions in '.png' file format
                ensure_dir('results/preds')
                # n_task, shots, num_classes, h, w = probas.size()
                # if (h, w) != (args.image_size, args.image_size):
                #    probas = F.interpolate(probas.view(n_task * shots, num_classes, h, w),
                #                           size=(args.image_size, args.image_size),
                #                           mode='bilinear', align_corners=True).view(n_task, shots, num_classes,
                #                                                                     args.image_size, args.image_size)

                merger = Merger(type=merge_mode, n=len(tta_transforms))
                for trans in tta_transforms:
                    augmented_img = trans.augment_image(qry_img)
                    augmented_feat = model.module.extract_features(augmented_img).detach().unsqueeze(1)
                    augmented_logits = classifier.get_logits(augmented_feat).detach()
                    augmented_probas = classifier.get_probas(augmented_logits)

                    n_task, shots, num_classes, h, w = augmented_probas.size()

                    if (h, w) != (args.image_size, args.image_size):
                        augmented_probas = F.interpolate(augmented_probas.view(n_task * shots, num_classes, h, w),
                                                         size=(args.image_size, args.image_size),
                                                         mode='bilinear', align_corners=True).view(n_task, shots,
                                                                                                   num_classes,
                                                                                                   args.image_size,
                                                                                                   args.image_size)

                    deaugmented_output = trans.deaugment_mask(augmented_probas.squeeze(0))
                    merger.append(deaugmented_output.unsqueeze(0))

                final_pred = merger.result
                final_pred = adjust_background_to_highest_class(args, final_pred) 

                pred = final_pred.argmax(2)  # [n_query, shot, H, W]
                pred = np.array(pred.squeeze().cpu(), np.uint8)
                pred = open_close_per_class(pred)
                pred = remove_small_objects_per_class(pred, 200)
                # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred, connectivity=8)
                # for i in range(1, num_labels):
                #    if stats[i, cv2.CC_STAT_AREA] < 200:
                #        pred[labels == i] = 0

                #print(np.unique(pred).tolist())
                fname = ''.join(image_name)
                cv2.imwrite(os.path.join('results/preds', fname + '.png'), pred)

    if q_label is None:
        return

    base_count, novel_count, sum_base_IoU, sum_novel_IoU = 4 * [0]
    results = []
    results.append('\nClass IoU Results')
    results.append('---------------------------------------')

    if args.novel_classes is not None:  # Update novel classnames
        update_novel_classes(base_novel_classes, args.novel_classes)

    for i, class_ in enumerate(val_loader.dataset.all_classes):
        if class_ == 0:
            continue

        IoU = cls_intersection[i] / (cls_union[i] + 1e-10)
        classname = base_novel_classes[class_].capitalize()
        if classname == '':
            classname = 'Novel class'
        results.append(f'%d %-25s \t %.2f' % (i, classname, IoU * 100))

        if class_ in val_loader.dataset.base_class_list:
            sum_base_IoU += IoU
            base_count += 1
        elif class_ in val_loader.dataset.novel_class_list:
            sum_novel_IoU += IoU
            novel_count += 1

    base_mIoU, novel_mIoU = sum_base_IoU / base_count, sum_novel_IoU / novel_count
    agg_mIoU = (base_mIoU + novel_mIoU) / 2
    wght_base_mIoU, wght_novel_mIoU = base_mIoU * 0.4, novel_mIoU * 0.6
    wght_sum_mIoU = wght_base_mIoU + wght_novel_mIoU

    results.append('---------------------------------------')
    results.append(f'\n%-30s \t %.2f' % ('Base mIoU', base_mIoU * 100))
    results.append(f'%-30s \t %.2f' % ('Novel mIoU', novel_mIoU * 100))
    results.append(f'%-30s \t %.2f' % ('Average of Base-and-Novel mIoU', agg_mIoU * 100))
    results.append(f'\n%-30s \t %.2f' % ('Weighted Base mIoU', wght_base_mIoU * 100))
    results.append(f'%-30s \t %.2f' % ('Weighted Novel mIoU', wght_novel_mIoU * 100))
    results.append(f'%-30s \t %.2f' % ('Weighted-sum of Base-and-Novel mIoU', wght_sum_mIoU * 100))
    results.append(
        f'The weighted scores are calculated using `0.4:0.6 => base:novel`, which are derived\nfrom the results presented in the SOA GFSS paper adopted as baseline.')
    iou_results = "\n".join(results)
    print(iou_results)

    if args.save_ious is True:  # Save class IoUs
        ensure_dir('results')
        with open(os.path.join('results', 'base_novel_ious.txt'), 'w') as f:
            f.write(iou_results)

    print('\n===> Runtime --- {:.1f}\n'.format(runtime))

    resume_random_state(random_state)
    return agg_mIoU


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.gpus)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    world_size = len(args.gpus)
    distributed = world_size > 1
    assert not distributed, 'Testing should not be done in a distributed way'
    args.distributed = distributed
    args.port = find_free_port()
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    mp.spawn(main_worker,
             args=(world_size, args),
             nprocs=world_size,
             join=True)
