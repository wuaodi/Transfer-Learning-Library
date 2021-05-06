import torch
import os
import math
import data_loader
import models
from config import CFG
import utils
import numpy as np
from mixmatrix import mixmatrix

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log = []

def test(model, target_test_loader):
    # 如果文件存在，则删除里面内容，如果不存在，则创建，存结果
    f_pred = open('predvalue.txt', mode='w')
    f_true = open('truevalue.txt', mode='w')
    f_pred.close()
    f_true.close()

    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
            # 统计每一个测试图片真值和标签
            predvalue = pred.cpu().numpy()
            truevalue = target.cpu().numpy()
            for i in range(len(predvalue)):
                f_pred = open('predvalue.txt', mode='a+')
                f_true = open('truevalue.txt', mode='a+')
                f_pred.write(str(predvalue[i]) + "\n")
                f_true.write(str(truevalue[i]) + "\n")
                f_pred.close()
                f_true.close()

        mixmatrix()  # 输出精度混淆矩阵

    print('{} --> {}: max correct: {}, accuracy{: .2f}%\n'.format(
        source_name, target_name, correct, 100. * correct / len_target_dataset))


def train(source_loader, target_train_loader, target_test_loader, model, optimizer, CFG):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    for e in range(CFG['epoch']):
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target = iter(
            source_loader), iter(target_train_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for i in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target, _ = iter_target.next()
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target = data_target.to(DEVICE)

            optimizer.zero_grad()
            label_source_pred, transfer_loss = model(data_source, data_target)
            clf_loss = criterion(label_source_pred, label_source)
            loss = clf_loss + CFG['lambda'] * transfer_loss
            # loss = clf_loss + 0 * transfer_loss    # source_only
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
            if i % CFG['log_interval'] == 0:
                print('Train Epoch: [{}/{} ({:02d}%)], cls_Loss: {:.6f}, transfer_loss: {:.6f}, total_Loss: {:.6f}'.format(
                    e + 1,
                    CFG['epoch'],
                    int(100. * i / n_batch), train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg))
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg])
        np_log = np.array(log, dtype=float)
        np.savetxt('train_log.csv', np_log, delimiter=',', fmt='%.6f')
        # Test
        test(model, target_test_loader)
        torch.save(model, 'latest_all.pth')    # save weights and net arch

def load_data(src, tar, tar_test, root_dir):
    folder_src = root_dir + src + '/images/'
    folder_tar = root_dir + tar + '/images/'
    folder_tar_test = root_dir + tar_test + '/images/'
    source_loader = data_loader.load_data(
        folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(
        folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    target_test_loader = data_loader.load_data(
        folder_tar_test, CFG['batch_size'], False, CFG['kwargs'])
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)

    source_name = "VNI"
    target_name = "IALT"
    tar_test_name = "IALT_test"

    print('Src: %s, Tar: %s' % (source_name, target_name))

    source_loader, target_train_loader, target_test_loader = load_data(
        source_name, target_name, tar_test_name, CFG['data_path'])

    model = models.Transfer_Net(
        CFG['n_class'], transfer_loss='mmd', base_net='resnet50').to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * CFG['lr']},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * CFG['lr']},
    ], lr=CFG['lr'], momentum=CFG['momentum'], weight_decay=CFG['l2_decay'])

    train(source_loader, target_train_loader,
          target_test_loader, model, optimizer, CFG)
    print(target_test_loader.dataset.class_to_idx)  # 输出数字标签对应关系
