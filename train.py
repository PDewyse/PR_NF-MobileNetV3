# Training script based on https://github.com/EstherBear/small-net-cifar100
import torch.nn as nn
import torch
import os
import sys
import argparse
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import datetime
import torch.cuda

import random
import wandb
from signal_propagation_plot.pytorch import plot_spp, get_average_channel_squared_mean_by_depth, get_average_channel_variance_by_depth
import numpy as np
from sklearn.metrics import roc_auc_score

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
CHECK_POINT_PATH = "./checkpoint"
MILESTONES = [60, 100, 120]#[60, 120, 160]#[15, 20, 30]

def training():
    net.train()
    length = len(trainloader)
    total_sample = len(trainloader.dataset)
    total_loss = 0
    correct = 0
    train_compute_time = 0
    for batch, (x, y) in enumerate(trainloader):
        x = x.cuda()
        y = y.cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        output = net(x)
        loss = loss_function(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end.record()

        # Waits for everything to finish running
        torch.cuda.synchronize()
        train_compute_time += start.elapsed_time(end)  # milliseconds

        _, predict = torch.max(output, 1)
        total_loss += loss.item()
        correct += (predict == y).sum()

        # fbatch.write("Epoch:{}\t batch:{}\t TrainedSample:{}\t TotalSample:{}\t Loss:{:.3f}\n".format(
        #         epoch+1, batch+1, batch*args.b + len(y), total_sample, loss.item()
        #     ))
        # fbatch.flush()
        
        if batch % 50 == 0 and epoch % 1 == 0:
            # generate signal propagation plots
            metrics = [get_average_channel_squared_mean_by_depth, get_average_channel_variance_by_depth]
            plot_spp(net, x, metrics, save_dir=spp_path, plot_name=f"epoch_{epoch}_batch_{batch}")

            print("| Epoch: {}/{}\t| Batch: {}/{} \t| Train Loss: {:.4f}\t|".format(
                epoch+1, args.e, batch+1,length, loss.item()
            ))

    # fepoch.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
    #     epoch + 1, total_loss/length, optimizer.param_groups[0]['lr'], float(correct)/ total_sample
    # ))
    # fepoch.flush()
            
    print("| Epoch {} Summary:\n| Average Train Loss: {:.4f}\n| Average training compute time (ms): {}".format(
        epoch+1, total_loss/length, train_compute_time/length
        ))
    return total_loss/length, train_compute_time/length


def evaluating():
    net.eval()
    length = len(testloader)
    total_sample = len(testloader.dataset)
    total_loss = 0
    correct = 0
    val_pred_time = 0
    all_targets = []
    all_outputs = []
    for batch, (x, y) in enumerate(testloader):
        x = x.cuda()
        y = y.cuda()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        output = net(x)
        end.record()
        
        # Waits for everything to finish running
        torch.cuda.synchronize()
        val_pred_time += start.elapsed_time(end)  # milliseconds

        _, predict = torch.max(output, 1)
        loss = loss_function(output, y)
        total_loss += loss.item()
        correct += (predict == y).sum()

        # for AUC metric
        all_targets.extend(y.cpu().numpy())
        all_outputs.extend(torch.softmax(output, dim=1).detach().cpu().numpy())

        # if batch % 10 == 0:
        #     print("| Epoch: {}/{}\t| Batch: {}/{} \t| Val Loss: {:.4f}\t|".format(
        #         epoch+1, args.e, batch+1,length, loss.item()
        #     ))

    acc = float(correct) / total_sample

    all_targets = np.array(all_targets)
    all_outputs = np.array(all_outputs)
    auc_scores = []
    for class_idx in range(num_classes):
        auc_score = roc_auc_score((all_targets == class_idx).astype(int), all_outputs[:, class_idx])
        auc_scores.append(auc_score)

    average_auc = np.mean(auc_scores)
    # feval.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
    #     epoch + 1, total_loss / length, optimizer.param_groups[0]['lr'], acc
    # ))
    # feval.flush()

    print("| Epoch {} Summary:\n| Average Val Loss: {:.4f}\n| Val Acc: {:.4f}\n| Average AUC: {:.4f}".format(
        epoch+1, total_loss/length, acc, average_auc
        ))
    return acc, total_loss/length, val_pred_time/length, average_auc

def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", default='NFMobilenet_v3_small', help='net type')
    parser.add_argument("-b", default=128, type=int, help='batch size')
    parser.add_argument("-lr", default=0.01, help='initial learning rate', type=float)
    parser.add_argument("-e", default=100, help='EPOCH', type=int)
    parser.add_argument("-optim", default="SGD", help='optimizer')
    parser.add_argument("-s", default=None, help='seed', type=int)
    parser.add_argument("-log", default=1, help='logging with wandb', type=int)
    args = parser.parse_args()
    
    # data processing
    print("=================================================================")
    print("Data processing...")
    # set seed
    seed = args.s
    set_seed(seed)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR100_TRAIN_MEAN, std=CIFAR100_TRAIN_STD)
    ])

    traindata = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testdata = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(traindata, batch_size=args.b, shuffle=True) #, num_workers=2 # changed num_workers to 0
    testloader = DataLoader(testdata, batch_size=args.b, shuffle=False) #, num_workers=2
    
    # define net
    print("=================================================================")
    print("Defining network and parameters...")
    if args.net == 'NFmobilenet':
        from models.NFmobilenet import NFmobilenet
        net = NFmobilenet(1, 100).cuda()
    elif args.net == 'NFmobilenetv2':
        from models.NFmobilenetv2 import NFmobilenetv2
        net = NFmobilenetv2(1, 100).cuda()
    elif args.net == 'NFMobilenet_v3_large':
        from models.NFmobilenetv3 import NFMobilenet_v3_large
        net = NFMobilenet_v3_large(1).cuda()
    elif args.net == 'NFMobilenet_v3_small':
        from models.NFmobilenetv3 import NFMobilenet_v3_small
        net = NFMobilenet_v3_small(1).cuda()
    elif args.net == 'NFefficientnetb0':
        from models.NFefficientnet import NFefficientnet
        print("loading net")
        net = NFefficientnet(1, 1, 100, bn_momentum=0.9).cuda()
        print("loading finish")
    else:
        print('We don\'t support this net.')
        sys.exit()

    # define loss, optimizer, lr_scheduler and checkpoint path
    loss_function = nn.CrossEntropyLoss()
    if args.optim == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    else:
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, last_epoch=-1)#, gamma = 0.1)
    time = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_').replace('.', '_' )
    checkpoint_path = os.path.join(CHECK_POINT_PATH, args.net, time)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    spp_path = os.path.join(checkpoint_path, 'spp_plots')
    # spp_path = "./Temp" # for testing spp layout
    if not os.path.exists(spp_path):
        os.makedirs(spp_path)
    
    # Training information
    # initialize wandb
    if args.log:
        wandb.init(project=f"Training {args.net} spp", name=args.net)
    print("=================================================================")
    print(f"Training information for {args.net}:")
    print("model:", args.net)
    print("hardware:", "GPU" if torch.cuda.is_available() else "CPU (CUDA was not found)")
    print("iterations:", args.e)
    print("batch size:", args.b)
    print("Learning rate:", args.lr)
    print("Optimizer:", args.optim)
    print("Scheduler:", f"MultiStepLR with milestones at {MILESTONES}.")
    print("Saving checkpoints to:", checkpoint_path)
    print("=================================================================")
    print("Starting training now...")
    print("=================================================================")
    # train and eval
    best_acc = 0
    total_train_time = 0
    # Get the number of classes (for cifar100, it is 100)
    num_classes = len(traindata.classes)
    diff = "calculating..."
    with open(os.path.join(checkpoint_path, 'EpochLog.txt'), 'w') as fepoch:
        with open(os.path.join(checkpoint_path, 'BatchLog.txt'), 'w') as fbatch:
            with open(os.path.join(checkpoint_path, 'EvalLog.txt'), 'w') as feval:
                with open(os.path.join(checkpoint_path, 'Best.txt'), 'w') as fbest:
                    for epoch in range(args.e):
                        epoch_start = datetime.datetime.now()
                        print(f"Training Epoch {epoch+1}/{args.e} (ETA = {diff:.4}s)")
                        train_loss, train_compute_time = training()

                        print(f"Evaluating Epoch {epoch+1}/{args.e}")
                        accuracy, val_loss, val_pred_time, average_auc = evaluating()
                        
                        scheduler.step()

                        print("--> Saving regular")
                        torch.save(net.state_dict(), os.path.join(checkpoint_path, 'regularParam.pth'))

                        if accuracy > best_acc:
                            print("--> Saving best")
                            torch.save(net.state_dict(), os.path.join(checkpoint_path, 'bestParam.pth'))

                            # fbest.write("Epoch:{}\t Loss:{:.3f}\t lr:{:.5f}\t acc:{:.3%}\n".format(
                            #     epoch + 1, averageloss, optimizer.param_groups[0]['lr'], accuracy
                            # ))
                            # fbest.flush()
                            best_acc = accuracy
                        
                        # total training time for an epoch
                        diff = (datetime.datetime.now() - epoch_start).total_seconds()
                        total_train_time += diff
                        print("=================================================================")
                        print(f"Estimated Total training time: {(diff*args.e/3600):.4} hrs.")
                        # log to wandb
                        if args.log:
                            wandb.log({"epoch": epoch,
                                       "learning_rate": optimizer.param_groups[0]['lr'],
                                       "accuracy (%)": accuracy*num_classes,
                                       "average_auc": average_auc,
                                       "train_loss": train_loss,
                                       "val_loss": val_loss,
                                       "train_compute_time (ms)": train_compute_time, # time spent for forward pass, backward pass, loss, and optimizer
                                       "val_pred_time (ms)": val_pred_time, # time spent for forward pass only
                                       "Training time": total_train_time
                                        })
    print("=================================================================")
    print(f"Total training time: {total_train_time/3600:.4}hrs.")
    print(f"Average training time per epoch: {total_train_time / args.e:.4}s.")
    print(f"Best accuracy: {best_acc:.4f}")
    print("=================================================================")
    print(f"Training information for {args.net}:")
    print("model:", args.net)
    print("hardware:", "GPU" if torch.cuda.is_available() else "CPU (CUDA was not found)")
    print("iterations:", args.e)
    print("batch size:", args.b)
    print("Learning rate:", args.lr)
    print("Optimizer:", args.optim)
    print("Scheduler:", f"MultiStepLR with milestones at {MILESTONES}.")
    print("=================================================================")
    if args.log:
        wandb.finish()