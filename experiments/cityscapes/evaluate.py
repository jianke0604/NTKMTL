import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error, delta_fn_arr
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)

def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    # binary mark to mask out undefined pixel space
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        # semantic loss: depth-wise cross entropy
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        # depth loss: l1 norm
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)

    return loss

def evaluate_model(model, test_loader, device):
    model.eval()
    conf_mat = ConfMatrix(model.segnet.class_nb)
    avg_cost = np.zeros(12, dtype=np.float32)
    test_batch = len(test_loader)
    cost = np.zeros(12, dtype=np.float32)
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth = next(test_dataset)
            test_data, test_label = test_data.to(device), test_label.long().to(
                device
            )
            test_depth = test_depth.to(device)

            test_pred = model(test_data)
            test_loss = torch.stack(
                (
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                )
            )

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[6] = test_loss[0].item()
            cost[9] = test_loss[1].item()
            cost[10], cost[11] = depth_error(test_pred[1], test_depth)
            avg_cost[6:] += cost[6:] / test_batch

            # compute mIoU and acc
        avg_cost[7:9] = conf_mat.get_metrics()

    return avg_cost

def main(args):
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    # 加载检查点
    if os.path.isfile(args.ckpt):
        logging.info(f"Loading checkpoint from {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt))
    else:
        logging.error(f"Checkpoint not found at {args.ckpt}")
        return

    # 加载测试数据集
    test_set = Cityscapes(root=args.data_path, train=False)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)

    # 评估模型
    avg_cost = evaluate_model(model, test_loader, device)

    # 输出评估结果
    logging.info(
        f"MEAN_IOU: {avg_cost[7]:.4f} | PIXEL_ACC: {avg_cost[8]:.4f} | ABS_ERR: {avg_cost[10]:.4f} | REL_ERR: {avg_cost[11]:.4f}"
    )
    delta_m = delta_fn(np.array([avg_cost[7], avg_cost[8], avg_cost[10], avg_cost[11]]))
    print(f"average delta_m: {delta_m}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(), "dataset"), help="Path to the NYUv2 dataset.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model", type=str, default="mtan", choices=["segnet", "mtan"], help="Model type to use.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader.")

    args = parser.parse_args()

    main(args)
# ./save/exp_d0.8_t20_tr_cont/fairgrad_180.pth