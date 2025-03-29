import os
import sys

import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
import time
import matplotlib.pyplot as plt
from model import VisionTransformer, VisionTransformer_wo_token, VisionTransformerGCN, LabelSmoothSoftmaxCEV1
from sklearn.metrics import classification_report, cohen_kappa_score


from loadData import data_reader
from loadData.split_data import HyperX, sample_gt


patch_size = 27
pad_width = patch_size // 2
num_components = 32          # india pines pavia 32 cunadong 35
split_type = ['number', 'ratio', 'disjoint'][0]
train_num = 50
val_num = 0
train_ratio = 0.1 
val_ratio = 0

max_epoch = 100
batch_size = 128
learning_rate = 5e-4   # cunadong 0.0005 
lb_smooth = 0.1
continued_epoch = 20
log_interval = 1
weight_decay = 0.001

dataset_name = "PaviaU"

plot_loss_curve = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

path_weight = "/home/leo/Oil_Spill_Detection/vitdgcn/weights/"
path_result = "/home/leo/Oil_Spill_Detection/vitdgcn/result/"
path_data = "/home/leo/DatasetSMD/"


data, data_gt = data_reader.load_data(dataset_name, path_data=path_data)
# type(data[0][0][0]),type(data_gt[0][0])
height, width, bands = data.shape
data = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))       # (height, width, Band+pad_width)
data = data[:, :, pad_width:data.shape[2]-pad_width]                                 # (height, width, Band)
gt = np.pad(data_gt, pad_width=pad_width, mode="constant", constant_values=(0))
# print(data.shape, gt.shape)


class_num= np.max(gt)
# print(np.max(gt))

data, pca = data_reader.apply_PCA(data, num_components=num_components)
train_gt, test_gt = sample_gt(gt, train_num=train_num, mode=split_type)
# train_gt, test_gt = sample_gt(gt, train_ratio=train_ratio , mode="ratio")
# train_gt, test_gt = sample_gt(gt, train_ratio=0.1, mode="disjoint")
# data_reader.data_info(train_gt, test_gt, start=0)
print(train_gt.shape, test_gt.shape)
# data_reader.data_info(train_gt, test_gt)

train_dataset = HyperX(data, train_gt, patch_size=patch_size, flip_augmentation=True, 
                        radiation_augmentation=True, mixture_augmentation=False, remove_zero_labels=True)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,)

test_dataset = HyperX(data, test_gt, patch_size=patch_size, remove_zero_labels=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,)
next(iter(train_loader))[0].shape, next(iter(test_loader))[1].shape


# net = VisionTransformer(img_size=patch_size, patch_size=4, in_c=num_components, num_classes=class_num,
#                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0).to(device)         # 88,89
# net = VisionTransformer_wo_token(img_size=patch_size, patch_size=4, in_c=num_components, num_classes=class_num,
#                 embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0).to(device)         # 89,90
net = VisionTransformerGCN(img_size=patch_size, patch_size=4, in_c=num_components, num_classes=class_num,
                embed_dim=64, depth=4, num_heads=4, mlp_ratio=4.0).to(device)           # 93,95
# criterion = nn.CrossEntropyLoss()
criterion = LabelSmoothSoftmaxCEV1(lb_smooth=lb_smooth)
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  # , weight_decay=weight_decay


# train and test
def train(net, num_epochs, criterion, optimizer):
  best_loss = 9999
  train_losses = []
  net.train()

  for epoch in range(1, num_epochs+1):
    correct = 0
    for data, target in train_loader:
      target = target -1
      data = data.to(device)
      # print(data.shape)
      target = target.to(device)

      optimizer.zero_grad()
      output = net(data)
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()

      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
    train_losses.append(loss.cpu().detach().item())
    
    if epoch % log_interval == 0:
      print('Train Epoch: {}\tLoss: {:.6f} \tAccuracy: {:.6f}'.format(epoch,  loss.item(),  correct / len(train_loader.dataset)))
    
    if loss.item() < best_loss:
      best_loss = loss.item()
      torch.save({"epoch":epoch,
                  "model": net.state_dict(),
                  "optimizer": optimizer.state_dict()},
                  os.path.join(path_weight, dataset_name + '_weights.pth'))
  return train_losses


def test(net):
  net.eval()
  test_losses = []
  test_preds = []
  test_loss = 0
  correct = 0
  
  with torch.no_grad():
    for data, target in test_loader:
      target = target - 1
      data = data.to(device)
      target = target.to(device)
      
      output = net(data)
      test_loss += criterion(output, target).item()
      test_pred = output.data.max(1, keepdim=True)[1]
      correct += test_pred.eq(target.data.view_as(test_pred)).sum()

      test_label = torch.argmax(output, dim=1)
      test_preds.append(test_label.cpu().numpy().tolist())
  test_losses.append(test_loss)
  
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
  return test_losses, test_preds



tic1 = time.time()
train_losses = train(net, max_epoch, criterion, optimizer)
toc1 = time.time()


path = os.path.join(path_weight, dataset_name + '_weights.pth')
if path != '':
    checkpoint = torch.load(path)
    net.load_state_dict(checkpoint['model'], strict=False)
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(path), "start", epoch_start)
else:
    raise NotImplementedError("Not weights")

tic2 = time.time()
test_losses, test_preds = test(net)
toc2 = time.time()

# plot
# if plot_loss_curve:
    # fig = plt.figure()
    # plt.plot(range(max_epoch), train_losses, color='blue')

    # # test_counter 是先定义好的，test——losses 训练一轮记录一次
    # # plt.scatter(test_counter, test_losses, color='red')
    # plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    # plt.xlabel('number of training examples seen')
    # plt.ylabel('negative log likelihood loss')
    # plt.show()


training_time = toc1 - tic1
testing_time = toc2 - tic2


# classification report
test_label = test_gt[pad_width : test_gt.shape[0]-pad_width, pad_width : test_gt.shape[1]-pad_width]
test_label = np.reshape(test_gt, [-1])
test_label = test_label[test_label>0] - 1

y_pred_test = [j for i in test_preds for j in i]


classification = classification_report(test_label, y_pred_test, digits=4)
kappa = cohen_kappa_score(test_label, y_pred_test)
print(classification, kappa)


run_date = time.strftime('%Y.%m.%d.%H.%M',time.localtime(time.time()))
print(run_date)

# save results
f = open(path_result + dataset_name + '.txt', 'a+')
str_results = '\n ======================' \
            + '\nrun data = ' + run_date \
            + "\nlearning rate = " + str(learning_rate) \
            + "\nlabel_smoothing = " + str(lb_smooth) \
            + "\nepochs = " + str(max_epoch) \
            + "\nsamples_type = " + str(split_type) \
            + "\ntrain ratio = " + str(train_ratio) \
            + "\nval ratio = " + str(val_ratio) \
            + "\ntrain num = " + str(train_num) \
            + "\nval num = " + str(val_num) \
            + "\nbatch_size = " + str(batch_size) \
            + "\npatch_size = " + str(patch_size) \
            + "\nnum_components = " + str(num_components) \
            + '\ntrain time = ' + str(training_time) \
            + '\ntest time = ' + str(testing_time) \
            + '\n' + classification \
            + "kappa = " + str(kappa) \
            + '\n'
            
f.write(str_results)
f.close()