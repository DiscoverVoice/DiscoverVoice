# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
from utils.model_utils import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast, GradScaler
from utils.paths import paths



def train(audio_model, train_loader, test_loader, exp_dir=paths.Results, metrics='acc',
          loss_fn_name='CE', n_epochs=50, lr=0.001, warmup=True, n_print_steps=100,
          scheduler_patience=5, scheduler_factor=0.1, wa=False, wa_start=20, wa_end=30, save_model=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                         time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.AdamW(trainables, lr, weight_decay=5e-4)

    if loss_fn_name == 'BCE':
        loss_fn = nn.BCEWithLogitsLoss()
    elif loss_fn_name == 'CE':
        loss_fn = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor,
                                                           patience=scheduler_patience)
    print(
        f'Now training, main metrics: {metrics}, loss function: {loss_fn_name}, learning rate scheduler: ReduceLROnPlateau')
    print(
        f'The learning rate scheduler reduces learning rate with factor {scheduler_factor} after {scheduler_patience} epochs of no improvement')

    epoch += 1
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([n_epochs, 10])
    audio_model.train()
    while epoch < n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print(f'warm-up learning rate is {optimizer.param_groups[0]["lr"]:.6f}')

            with autocast():
                audio_output = audio_model(audio_input)
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
                else:
                    loss = loss_fn(audio_output, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time) / audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time) / audio_input.shape[0])

            print_step = global_step % n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (n_print_steps / 10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                      'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                      'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                      'Train Loss {loss_meter.avg:.4f}\t'.format(
                    epoch, i, len(train_loader), per_sample_time=per_sample_time,
                    per_sample_data_time=per_sample_data_time,
                    per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, loss_fn, exp_dir, epoch)

        cum_stats = validate_ensemble(exp_dir, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if metrics == 'mAP':
            result[epoch - 1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg,
                                    valid_loss,
                                    cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch - 1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg,
                                    valid_loss,
                                    cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if metrics == 'acc':
                best_epoch = epoch

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state.dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state.dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state.dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step(valid_loss)

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param.groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) + '.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time - begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    if wa:
        stats = validate_wa(audio_model, test_loader, exp_dir, wa_start, wa_end, save_model)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions']) / 2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls']) / 2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        wa_result = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC)]
        print('---------------Training Finished---------------')
        print('weighted averaged model results')
        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        np.savetxt(exp_dir + '/wa_result.csv', wa_result)


def validate(audio_model, val_loader, loss_fn, exp_dir, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to("cpu").detach()

            A_predictions.append(predictions)
            A_targets.append(labels.to("cpu").detach())

            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                loss = loss_fn(audio_output, labels)
            A_loss.append(loss.to("cpu").detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        if not os.path.exists(exp_dir + "/predictions"):
            os.mkdir(exp_dir + "/predictions")
            np.savetxt(exp_dir + "/predictions/target.csv", target, delimiter=",")
        np.savetxt(
            exp_dir + "/predictions/predictions_" + str(epoch) + ".csv",
            audio_output,
            delimiter=",",
        )

    return stats, loss


def validate_ensemble(exp_dir, epoch):
    target = np.loadtxt(exp_dir + "/predictions/target.csv", delimiter=",")
    if epoch == 1:
        cum_predictions = np.loadtxt(
            exp_dir + "/predictions/predictions_1.csv", delimiter=","
        )
    else:
        cum_predictions = np.loadtxt(
            exp_dir + "/predictions/cum_predictions.csv", delimiter=","
        ) * (epoch - 1)
        predictions = np.loadtxt(
            exp_dir + "/predictions/predictions_" + str(epoch) + ".csv", delimiter=","
        )
        cum_predictions = cum_predictions + predictions
        os.remove(exp_dir + "/predictions/predictions_" + str(epoch - 1) + ".csv")

    cum_predictions = cum_predictions / epoch
    np.savetxt(
        exp_dir + "/predictions/cum_predictions.csv", cum_predictions, delimiter=","
    )

    stats = calculate_stats(cum_predictions, target)
    return stats


def validate_wa(audio_model, val_loader, exp_dir, start_epoch, end_epoch, save_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sdA = torch.load(
        exp_dir + "/models/audio_model." + str(start_epoch) + ".pth",
        map_location=device,
    )

    model_cnt = 1
    for epoch in range(start_epoch + 1, end_epoch + 1):
        sdB = torch.load(
            exp_dir + "/models/audio_model." + str(epoch) + ".pth", map_location=device
        )
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        if not save_model:
            os.remove(exp_dir + "/models/audio_model." + str(epoch) + ".pth")

    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + "/models/audio_model_wa.pth")

    stats, loss = validate(audio_model, val_loader, "wa")
    return stats


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_stats(output, target):
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_recall_curve
    output = output.numpy()
    target = target.numpy()

    acc = accuracy_score(target, output.argmax(axis=1))
    AP = average_precision_score(target, output, average='macro')
    auc = roc_auc_score(target, output, average='macro', multi_class='ovo')
    precisions, recalls, _ = precision_recall_curve(target.ravel(), output.ravel())

    stats = [{'acc': acc, 'AP': AP, 'auc': auc, 'precisions': precisions, 'recalls': recalls}]
    return stats


def d_prime(auc):
    from scipy.stats import norm
    return norm.ppf(auc) * (2 ** 0.5)
