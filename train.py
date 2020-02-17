"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import datetime
from torch.utils.data import DataLoader
from src.utils import *
from src.dataset import MyDataset
from src.han_model import *
from src.stock_han_model import *
from tensorboardX import SummaryWriter
import argparse
import shutil
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    # training params
    parser.add_argument("--model_type", type=str, default="muil_han")    # model_type : ori_han; sent_ori_han; muil_han; sent_muil_han;muil_stock_han;sent_muil_stock_han
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_epoches", type=int, default=100)
    parser.add_argument("--lambda", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--es_min_delta", type=float, default=0.0005,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("--es_patience", type=int, default=20,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    # model params
    parser.add_argument("--add_stock", type=bool, default=False)
    parser.add_argument("--days_hidden_size", type=int, default=128)
    parser.add_argument("--news_hidden_size", type=int, default=64)
    parser.add_argument("--sent_hidden_size", type=int, default=32)
    parser.add_argument("--stock_hidden_size", type=int, default=128)
    parser.add_argument("--head_num", type=int, default=16)
    parser.add_argument("--days_num", type=int, default=days_num)
    # data params
    parser.add_argument("--train_set", type=str, default= data_root + "train_data_{}_{}_{}.csv".format(dataset, vectors_dim, days_num))
    parser.add_argument("--test_set", type=str, default= data_root + "test_data_{}_{}_{}.csv".format(dataset, vectors_dim, days_num))
    parser.add_argument("--test_interval", type=int, default=2, help="Number of epoches between testing phases")
    parser.add_argument("--word2vec_path", type=str, default=dict_path)
    parser.add_argument("--log_path", type=str, default="/home/ingrid/Projects/PythonProjects/stock_predict/tensorboard/")
    parser.add_argument("--saved_path", type=str, default="/home/ingrid/Projects/PythonProjects/stock_predict/trained_models/")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
        print("cuda...")
    else:
        torch.manual_seed(123)
    # training setting
    output_file = open(opt.saved_path + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))
    training_params = {"batch_size": opt.batch_size,
                       "num_workers": opt.num_workers,
                       "shuffle": True,
                       "pin_memory": True,
                       "drop_last": True}
    test_params = {"batch_size": opt.batch_size,
                   "num_workers": opt.num_workers,
                   "shuffle": False,
                   "pin_memory": True,
                   "drop_last": False}
    # training dataset info
    # max_news_length, max_sent_length, max_word_length = get_max_lengths(opt.train_set)
    # stock_length = 9

    data_init_time = datetime.datetime.now()
    training_set = MyDataset(data_path=opt.train_set)
    training_generator = DataLoader(training_set, **training_params)
    test_set = MyDataset(data_path=opt.test_set)
    test_generator = DataLoader(test_set, **test_params)
    data_end_time = datetime.datetime.now()
    print("the data loading time is: {}s...".format((data_end_time - data_init_time).seconds))

    # model init
    model_init_time = datetime.datetime.now()
    if opt.model_type == "ori_han":
        model = Ori_HAN(days_num=opt.days_num,
                        days_hidden_size=opt.days_hidden_size,
                        news_hidden_size=opt.news_hidden_size,
                        num_classes=training_set.num_classes,
                        pretrained_word2vec_path=opt.word2vec_path,
                        dropout=opt.dropout)
    elif opt.model_type == "sent_ori_han":
        model = Sent_Ori_HAN(days_num=opt.days_num,
                             days_hidden_size=opt.days_hidden_size,
                             news_hidden_size=opt.news_hidden_size,
                             sent_hidden_size=opt.sent_hidden_size,
                             num_classes=training_set.num_classes,
                             pretrained_word2vec_path=opt.word2vec_path,
                             dropout=opt.dropout)
    elif opt.model_type == "muil_han":
        model = Muil_HAN(head_num=opt.head_num,
                         days_num=opt.days_num,
                         days_hidden_size=opt.days_hidden_size,
                         news_hidden_size=opt.news_hidden_size,
                         num_classes=training_set.num_classes,
                         pretrained_word2vec_path=opt.word2vec_path,
                         dropout=opt.dropout)
    elif opt.model_type == "sent_muil_han":
        model = Sent_Muil_HAN(head_num=opt.head_num,
                              days_num=opt.days_num,
                              days_hidden_size=opt.days_hidden_size,
                              news_hidden_size=opt.news_hidden_size,
                              sent_hidden_size=opt.sent_hidden_size,
                              num_classes=training_set.num_classes,
                              pretrained_word2vec_path=opt.word2vec_path,
                              dropout=opt.dropout)
    elif opt.model_type == "muil_stock_han":
        model = Muil_Stock_HAN(head_num=opt.head_num,
                               days_num=opt.days_num,
                               days_hidden_size=opt.days_hidden_size,
                               news_hidden_size=opt.news_hidden_size,
                               stock_hidden_size=opt.stock_hidden_size,
                               stock_length=stock_length,
                               num_classes=training_set.num_classes,
                               pretrained_word2vec_path=opt.word2vec_path,
                               dropout=opt.dropout)
    elif opt.model_type == "sent_muil_stock_han":
        model = Sent_Muil_Stock_HAN(head_num=opt.head_num,
                                    days_num=opt.days_num,
                                    days_hidden_size=opt.days_hidden_size,
                                    news_hidden_size=opt.news_hidden_size,
                                    sent_hidden_size=opt.sent_hidden_size,
                                    stock_hidden_size=opt.stock_hidden_size,
                                    stock_length=stock_length,
                                    num_classes=training_set.num_classes,
                                    pretrained_word2vec_path=opt.word2vec_path,
                                    dropout=opt.dropout)
    model_end_time = datetime.datetime.now()
    print("the model init time is: {}s...".format((model_end_time - model_init_time).seconds))

    # other setting
    if os.path.isdir(opt.log_path + opt.model_type):
        shutil.rmtree(opt.log_path + opt.model_type) # 递归删除文件夹下的所有子文件夹
    os.makedirs(opt.log_path + opt.model_type)
    writer = SummaryWriter(opt.log_path + opt.model_type)

    # 模型训练相关信息初始化
    if torch.cuda.is_available():
        model.cuda()
        print("model use cuda...")


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    # 训练模型
    print("start to train model...")
    for epoch in range(opt.num_epoches):
        dataloader_init_time = datetime.datetime.now()
        for iter, (days_news, days_stock, label) in enumerate(training_generator):
            dataloader_end_time = datetime.datetime.now()
            print("the dataloader loading time is: {}s...".format((dataloader_end_time - dataloader_init_time).seconds))
            if torch.cuda.is_available():
                days_news = days_news.cuda()
                days_stock = days_stock.cuda()
                label = label.cuda()
                print("data use cuda...")
            training_init_time = datetime.datetime.now()
            optimizer.zero_grad()
            if opt.model_type in ["ori_han", "sent_ori_han", "muil_han", "sent_muil_han"]:
                predictions = model(days_news)
            elif opt.model_type in ["muil_stock_han", "sent_muil_stock_han"]:
                predictions = model(days_news, days_stock)
            loss = criterion(predictions, torch.tensor(label))
            loss.backward()
            optimizer.step()
            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(), list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
            training_end_time = datetime.datetime.now()
            print("the training time is: {}s...".format((training_end_time - training_init_time).seconds))
            dataloader_init_time = datetime.datetime.now()
        if epoch % opt.test_interval == 0:
            model.eval()
            loss_ls = []
            te_label_ls = []
            te_pred_ls = []
            for te_days_news, te_days_stock, te_label in test_generator:
                num_sample = len(te_label)
                if torch.cuda.is_available():
                    te_days_news = te_days_news.cuda()
                    te_days_stock = te_days_stock.cuda()
                    te_label = te_label.cuda()
                with torch.no_grad():
                    if opt.model_type in ["ori_han", "sent_ori_han", "muil_han", "sent_muil_han"]:
                        te_predictions = model(te_days_news)
                    elif opt.model_type in ["muil_stock_han", "sent_muil_stock_han"]:
                        te_predictions = model(te_days_news, te_days_stock)
                te_loss = criterion(te_predictions, torch.tensor(te_label))
                loss_ls.append(te_loss * num_sample)
                te_label_ls.extend(te_label.clone().cpu())
                te_pred_ls.append(te_predictions.clone().cpu())
            te_loss = sum(loss_ls) / test_set.__len__()
            te_pred = torch.cat(te_pred_ls, 0)
            te_label = np.array(te_label_ls)
            test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
            output_file.write(
                "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                    epoch + 1,
                    opt.num_epoches,
                    te_loss,
                    test_metrics["accuracy"],
                    test_metrics["confusion_matrix"]))
            print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epoches,
                optimizer.param_groups[0]['lr'],
                te_loss, test_metrics["accuracy"]))
            writer.add_scalar('Test/Loss', te_loss, epoch)
            writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
            model.train()
            if te_loss + opt.es_min_delta < best_loss:
                best_loss = te_loss
                best_epoch = epoch
                torch.save(model, opt.saved_path + opt.model_type + "_model")
                print("Success to save model....")

            # Early stopping
            if epoch - best_epoch > opt.es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, te_loss))
                break


if __name__ == "__main__":
    opt = get_args()
    train(opt)
