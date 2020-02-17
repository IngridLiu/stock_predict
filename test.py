"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils import get_evaluation, get_max_lengths
from src.dataset import MyDataset
from src.utils import *
import argparse
import shutil
import csv
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Hierarchical Attention Networks for Document Classification""")
    parser.add_argument("--model_type", type=str, default="sent_ori_han")  # model_type : ori_han; sent_ori_han; muil_han; sent_muil_han;muil_stock_han;sent_muil_stock_han
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--train_set", type=str, default= data_root + "train_data_{}_{}_{}.csv".format(dataset, vectors_dim, days_num))
    parser.add_argument("--test_set", type=str, default= data_root + "test_data_{}_{}_{}.csv".format(dataset, vectors_dim, days_num))
    parser.add_argument("--model_path", type=str, default="/home/ingrid/Projects/PythonProjects/stock_predict/trained_models/")
    parser.add_argument("--word2vec_path", type=str, default="/home/ingrid/Model/glove_ch/vectors_50.txt")
    parser.add_argument("--output", type=str, default="/home/ingrid/Projects/PythonProjects/stock_predict/predictions/")
    args = parser.parse_args()
    return args


def test(opt):
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "drop_last": False}
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)
    model_init_time = datetime.datetime.now()
    if torch.cuda.is_available():
        model = torch.load(opt.model_path + opt.model_type + "_model")
    else:
        model = torch.load(opt.model_path + opt.model_type + "_model", map_location=lambda storage, loc: storage)
    model_end_time = datetime.datetime.now()
    print("the model loading time is: {}s...".format((model_end_time - model_init_time).seconds))

    data_init_time = datetime.datetime.now()
    test_set = MyDataset(data_path=opt.test_set)
    test_generator = DataLoader(test_set, **test_params)
    data_end_time = datetime.datetime.now()
    print("the data loading time is: {}s...".format((data_end_time - data_init_time).seconds))

    if torch.cuda.is_available():
        model.cuda()
        print("model use cuda...")


    model.eval()
    te_label_ls = []
    te_pred_ls = []
    num_iter = len(test_generator)
    # 测试模型
    print("start to test model...")
    dataloader_init_time = datetime.datetime.now()
    for iter, (te_days_news, te_days_stock, te_label) in enumerate(test_generator):
        dataloader_end_time = datetime.datetime.now()
        print("the dataloader loading time of iter :{}/{} is {}s...".format(iter, num_iter, (dataloader_end_time - dataloader_init_time).seconds))
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
            te_predictions = F.softmax(te_predictions)
        te_label_ls.extend(te_label.clone().cpu())
        te_pred_ls.append(te_predictions.clone().cpu())
        dataloader_init_time = datetime.datetime.now()
    te_pred = torch.cat(te_pred_ls, 0).numpy()
    te_label = np.array(te_label_ls)

    fieldnames = ['True label', 'Predicted label', 'News Content', 'Stock Content']
    with open(opt.output + opt.model_type + "pred.csv", 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        for i, j, x, y in zip(te_label, te_pred, test_set.total_newses, test_set.total_stocks):
            writer.writerow(
                {'True label': i + 1, 'Predicted label': np.argmax(j) + 1, 'News Content': x, 'Stock Content': y})

    test_metrics = get_evaluation(te_label, te_pred,
                                  list_metrics=["accuracy", "loss", "confusion_matrix"])
    print("Prediction:\nLoss: {} Accuracy: {} \nConfusion matrix: \n{}".format(test_metrics["loss"],
                                                                               test_metrics["accuracy"],
                                                                               test_metrics["confusion_matrix"]))


if __name__ == "__main__":
    opt = get_args()
    test(opt)
