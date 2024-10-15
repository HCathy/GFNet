import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import spectral
import scipy.io as sio
from matplotlib import pyplot as plt

# from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
# 训练和验证
from final_model import GFNet
from function import normalize, get_data, GET_A2, metrics, show_results, train_and_test_data, train_epoch, valid_epoch, \
    output_metric, applyPCA

# 设置参数
parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'PaviaU', 'Salinas'], default='Indian', help='dataset to use')
parser.add_argument("--num_run", type=int, default=1)
parser.add_argument('--epoches', type=int, default=10, help='epoch number')
parser.add_argument('--patches', type=int, default=9, help='number of patches')
parser.add_argument('--n_gcn', type=int, default=15, help='number of related pix')
parser.add_argument('--pca_band', type=int, default=50, help='pca_components')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=10, help='number of evaluation')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# 设置随机种子，保证结果可复现
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

# 准备数据
input,gt_label, num_classes, total_pos_train, total_pos_test, total_pos_true, y_train, y_test, y_true = get_data(args.dataset)
input = applyPCA(input, numComponents=args.pca_band)
input_normalize = normalize(input)
height, width, band = input_normalize.shape
print("height={0}, width={1}, band={2}".format(height, width, band))


# 获取训练和测试数据
x_train_band, x_test_band, x_true_band, corner_train, corner_test, corner_true, indexs_train, indexs_test, indexs_true = train_and_test_data(
    input_normalize, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, w=height, h=width,
    n_gcn=args.n_gcn)

input2 = torch.from_numpy(input_normalize).type(torch.FloatTensor)
A_train = GET_A2(x_train_band, input2, corner=corner_train, patches=args.patches, l=3, sigma=10)
x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
Label_train = Data.TensorDataset(A_train, x_train, y_train)

A_test = GET_A2(x_test_band, input2, corner=corner_test, patches=args.patches, l=3, sigma=10)
x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)
Label_test = Data.TensorDataset(A_test, x_test, y_test)

x_true = torch.from_numpy(x_true_band).type(torch.FloatTensor)
y_true = torch.from_numpy(y_true).type(torch.LongTensor)
Label_true = Data.TensorDataset(x_true, y_true)

label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=False)
label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=False)
label_true_loader = Data.DataLoader(Label_true, batch_size=100, shuffle=False)

results = []


def save_results(file_path, total_pos_train, total_pos_test,total_gt, y_train, y_test, pre_v, height, width, num_classes, best_OA2):
    # 准备保存的数据
    train_labels = np.zeros((height, width))  # 保存训练集坐标
    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_gt2.jpg"), gt_label.astype(int),
    #                   colors=spectral.spy_colors)  # 总预测图
    predict_map = total_gt
    predict_error_map = np.zeros((height, width))  # 保存判断错误的点
    predict_error_gt = np.zeros((height, width))  # 保存判断错误的点的真实标签

    # # 绘制训练集所在的位置
    # for index in range(len(total_pos_train)):
    #     pos2d = total_pos_train[index]
    #     train_labels[pos2d[0], pos2d[1]] = y_train[pos2d[0], pos2d[1]]
    for index in range(len(total_pos_train)):
        pos2d = total_pos_train[index]
        train_labels[pos2d[0], pos2d[1]] = y_train[index].item()  # 使用单个索引访问

    # 总预测
    for index in range(len(total_pos_test)):
        pos2d = total_pos_test[index]
        if (pre_v[index] + 1) != y_test[index].item():
            predict_error_map[pos2d[0], pos2d[1]] = pre_v[index] + 1
            predict_error_gt[pos2d[0], pos2d[1]] = y_test[index].item()+1
        predict_map[pos2d[0], pos2d[1]] = pre_v[index] + 1

    sio.savemat(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_pos2d.mat"),
                {'pos2d_train': total_pos_train, "pos2d_test": total_pos_test, "class_train_num": [sum(y_train == i) for i in range(num_classes)]})

    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_trainsets_gt.jpg"), train_labels.astype(int), colors=spectral.spy_colors)  # 训练集位置图
    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_predict.jpg"), predict_map.astype(int), colors=spectral.spy_colors)  # 总预测图
    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_predict_error.jpg"), predict_error_map.astype(int), colors=spectral.spy_colors)  # 总预测失败图
    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_predict_error_gt.jpg"), predict_error_gt.astype(int), colors=spectral.spy_colors)  # 预测失败的地方的正确标签图

    # plt.figure(figsize=(8, 8))
    # plt.imshow(train_labels.astype(int), cmap='nipy_spectral')
    # # plt.colorbar(label='Class ID')  # 显示颜色条以标识类别
    # plt.axis('off')
    # temp_image_path = '../可视化/res_recorder/In/gt'
    # plt.savefig(temp_image_path, dpi=1800, bbox_inches='tight', pad_inches=0)

    '''
    viridis: 默认的颜色映射，适用于大多数情况，具有良好的视觉效果和可读性。
    plasma: 热烈的颜色映射，适合用于强调数据的强度。
    inferno: 深色到亮色的渐变，适合展示高动态范围的数据。
    magma: 类似于 inferno，但颜色渐变更柔和，适合展示较暗的区域。
    cividis: 针对色盲友好的颜色映射，适合所有观众。
    gray: 灰度映射，适合表示单通道图像或进行黑白打印。
    jet: 彩虹色映射，虽然在数据可视化中不推荐使用，但在某些情况下仍然被使用。
    hot: 从黑色到红色，再到黄色的渐变，适合展示温度分布。
    cool: 从青色到洋红色的渐变，适合用于较低对比度的数据。
    nipy_spectral: 从蓝色到红色的渐变，适合科学可视化。
    '''
    # 绘图
    plt.figure(figsize=(8, 8))
    plt.imshow(predict_map.astype(int), cmap='nipy_spectral')
    # plt.colorbar(label='Class ID')  # 显示颜色条以标识类别
    plt.axis('off')
    temp_image_path = '../final/res_recorder/Salinas'
    plt.savefig(temp_image_path, dpi=1800, bbox_inches='tight', pad_inches=0)
    plt.show()


for run in range(args.num_run):
    print(f"Run {run + 1}/{args.num_run}")
    # 初始化模型
    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0
    gcn_net = GFNet(height, width, band, num_classes, dim=64)
    gcn_net = gcn_net.cuda()
    # 损失函数
    criterion = nn.CrossEntropyLoss().cuda()
    # 优化器
    optimizer = torch.optim.Adam(gcn_net.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma)
    print("Start training")
    tic = time.time()
    train_time = 0
    for epoch in range(args.epoches):
        scheduler.step()
        gcn_net.train()
        train_start = time.time()
        train_acc, train_obj, tar_t, pre_t = train_epoch(gcn_net, label_train_loader, criterion, optimizer,
                                                         indexs_train)
        train_end = time.time()
        train_sum = train_end - train_start
        train_time += train_sum
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}".format(epoch + 1, train_obj, train_acc))

        if (epoch % args.test_freq == 0) or (epoch == args.epoches - 1) and epoch >= args.epoches * 0.6:
            gcn_net.eval()
            # tar_v, pre_v = valid_epoch(gcn_net, label_test_loader, criterion, indexs_test)
            tar_v, pre_v = valid_epoch(gcn_net, label_test_loader, criterion, indexs_test)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            if OA2 >= best_OA2 and AA_mean2 >= best_AA_mean2 and Kappa2 >= best_Kappa2:
                best_OA2 = OA2
                best_AA_mean2 = AA_mean2
                best_Kappa2 = Kappa2
                run_results = metrics(best_OA2, best_AA_mean2, best_Kappa2, AA2)

    # 保存结果
    file_path = "../final/res_recorder"
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = os.path.join(file_path, args.dataset)
    if not os.path.isdir(file_path):
        os.mkdir(file_path)
    file_path = os.path.join(file_path, f'run_{run + 1}')
    if not os.path.isdir(file_path):
        os.mkdir(file_path)

    # spectral.save_rgb(os.path.join(file_path, f"ps{args.patches}_oa{best_OA2:.2f}_gt1.jpg"), gt_label.astype(int), colors=spectral.spy_colors)  # 总预测图

    plt.figure(figsize=(8, 8))
    plt.imshow(gt_label.astype(int), cmap='nipy_spectral')
    # plt.colorbar(label='Class ID')  # 显示颜色条以标识类别
    plt.axis('off')
    temp_image_path = '../final/res_recorder/sa_gt_label'
    plt.savefig(temp_image_path, dpi=1800, bbox_inches='tight', pad_inches=0)
    plt.show()

    save_results(file_path, total_pos_train, total_pos_test,gt_label, y_train, y_test, pre_v, height, width, num_classes,
                     best_OA2)

    show_results(run_results, aggregated=False, dataset_name=args.dataset)
    results.append(run_results)
    print('训练时间：', train_time)
    toc = time.time()
    total_time = toc - tic
    print('训练+测试总时间：', total_time)

if args.num_run > 1:
    show_results(results, aggregated=True, dataset_name=args.dataset)


