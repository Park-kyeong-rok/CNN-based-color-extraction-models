import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from DATASET import train_dataset, test_dataset, data_split, val_dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchvision import models


# 랜덤 시드 고정
torch.manual_seed(41)
torch.cuda.manual_seed_all(41)
np.random.seed(41)

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_number', type=int, default=1)
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--model', type=str, default='resnet')
parser.add_argument('--data', type=str, default='bgr')
parser.add_argument('--file_name', type=str, default='hsv__result')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--SEED', type=int, default=0)
parser.add_argument('--CPU', type = int, default = 0)
args = parser.parse_args()
if args.CPU == 1:
    device = torch.device('cpu')
else:
    device = torch.device('cuda:' + str(args.cuda_number))

# load the dataset
train_x = np.load('data/' + args.data + 'train_x.npy')
test_x = np.load('data/' + args.data + 'test_x.npy')
train_y = np.load('data/train_y.npy')
test_y = np.load('data/test_y.npy')
train_x, val_x, test_x, train_y, test_y, val_y = data_split(train_x, test_x, train_y, test_y, args.SEED)

train_y = torch.LongTensor(train_y)
test_y = torch.LongTensor(test_y)
val_y = torch.LongTensor(val_y)

# each data's mean, std
b_mean, g_mean, r_mean, h_mean, s_mean, v_mean = 0.42613672995985785, 0.4206323304396943, 0.4209770064850829, 0.3475583787663586, 0.13983006898417755, 0.45113514880737915
b_std, g_std, r_std, h_std, s_std, v_std = 0.21486078047886728, 0.21338082775957695, 0.2142748399667734, 0.17522139676542173, 0.148305626909194, 0.21543327051939412
if args.model == 'resnet':
    model = models.resnet50(pretrained=bool(args.pretrain))
    num_class = 10
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)
    model.conv1 = nn.Conv2d(len(args.data), 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)


if args.data == 'bgr':
    mean_list = [b_mean, g_mean, r_mean]
    std_list = [b_std, g_std, r_std]
elif args.data == 'hs':
    mean_list = [h_mean, s_mean]
    std_list = [h_std, s_std]
elif args.data == 'bgrv':
    mean_list = [b_mean, g_mean, r_mean, v_mean]
    std_list = [b_std, g_std, r_std, v_std]
elif args.data == 'h':
    mean_list = [h_mean]
    std_list = [h_std]
elif args.data == 'bgrh':
    mean_list = [b_mean, g_mean, r_mean, h_mean]
    std_list = [b_std, g_std, r_std, h_std]
# 트랜스포머
train_transformer = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean_list, std_list)])
val_transformer = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean_list, std_list)])
test_transformer = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean_list, std_list)])

train_dataset = train_dataset(train_x, train_y, train_transformer)
val_dataset = val_dataset(val_x, val_y, val_transformer)
test_dataset = test_dataset(test_x, test_y, test_transformer)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[i*3 for i in range(1,80)],
                                                 gamma=0.5)

start_time = time.time()
train_batch_length = len(train_dataloader)
val_batch_length = len(val_dataloader)
test_batch_length = len(test_dataloader)

train_loss_dict = {}
val_loss_dict = {}
acc_dict = {}
for epoch in range(1, args.epochs + 1):
    train_loss = 0
    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        cost = criterion(prediction, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        train_loss += cost
    train_loss /= train_batch_length
    train_loss_dict[epoch] = train_loss.cpu().item()
    with torch.no_grad():
        test_acc = 0
        val_loss = 0
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            acc = torch.argmax(prediction, 1) == y
            acc = acc.float().mean()
            val_loss += criterion(prediction, y)

            optimizer.zero_grad()
        for x, y in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            acc = torch.argmax(prediction, 1) == y
            acc = acc.float().mean()
            test_acc += acc
            optimizer.zero_grad()
        val_loss /= val_batch_length
        test_acc /= test_batch_length
        val_loss_dict[epoch] = val_loss.cpu().item()
        acc_dict[epoch] = test_acc.cpu().item()

    print('epoch: %d, train_loss: %f, val_loos: %f, test_acc = %f' % (epoch, train_loss, val_loss, test_acc))
learning_time = time.time() - start_time
print('train_time: %f' % (learning_time))
result_x = np.array([11])
result_y = np.array([11])
test_acc = 0
with torch.no_grad():
    result_acc = 0
    for x, y in test_dataloader:
        x = x.to(device)
        y = y.to(device)
        x = model(x)
        x = torch.argmax(x, 1)
        out = x == y
        # x, y 값 저장해놓기
        x = x.cpu()
        y = y.cpu()
        x = x.numpy()
        y = y.numpy()
        result_x = np.concatenate((result_x, x))
        result_y = np.concatenate((result_y, y))
        test_acc = out.float().mean()
        result_acc += test_acc
    result_acc /= test_batch_length
print('-------------')
print('result acc:', result_acc)
np.save('result/'+args.data+'/train_loss'  + str(args.epochs) + args.data + args.file_name,
        np.array(list(train_loss_dict.values())))
np.save('result/'+args.data+'/val_loss'  + str(args.epochs) + args.data + args.file_name,
        np.array(list(val_loss_dict.values())))
np.save('result/'+args.data+ '/test_acc'  + str(args.epochs) + args.data + args.file_name,
        np.array(list(acc_dict.values())))
np.save('result/'+args.data+ '/result_x'  + str(args.epochs) + args.data + args.file_name, result_x)
np.save('result/'+args.data+ '/result_y'  + str(args.epochs) + args.data + args.file_name, result_y)
a = open('result/'+args.data+ '/Args_of_'  + str(args.epochs) + args.data + args.file_name + '.txt', 'w')
a.write(str(args))
a.write('learning_time:' + str(learning_time))
a.write('result acc:' + str(result_acc))