import logging
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser

from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import ConfMatrix, delta_fn, depth_error, normal_error
import numpy as np

# 设置日志
logging.basicConfig(level=logging.INFO)

def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
    elif task_type == "depth":
        loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
    elif task_type == "normal":
        loss = 1 - torch.sum((x_pred * x_output) * binary_mask) / torch.nonzero(
            binary_mask, as_tuple=False
        ).size(0)
    return loss

def evaluate_model(model, test_loader, device):
    model.eval()
    conf_mat = ConfMatrix(model.segnet.class_nb)
    avg_cost = np.zeros(12, dtype=np.float32)
    test_batch = len(test_loader)
    cost = np.zeros(24, dtype=np.float32)
    avg_cost = np.zeros(24, dtype=np.float32)
    with torch.no_grad():
        test_dataset = iter(test_loader)
        for k in range(test_batch):
            test_data, test_label, test_depth, test_normal = next(test_dataset)
            test_data, test_label = test_data.to(device), test_label.long().to(
                device
            )
            test_depth, test_normal = test_depth.to(device), test_normal.to(device)

            test_pred = model(test_data)
            test_loss = torch.stack(
                (
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                    calc_loss(test_pred[2], test_normal, "normal"),
                )
            )

            conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

            cost[12] = test_loss[0].item()
            cost[15] = test_loss[1].item()
            cost[16], cost[17] = depth_error(test_pred[1], test_depth)
            cost[18] = test_loss[2].item()
            cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                test_pred[2], test_normal
            )
            avg_cost[12:] += cost[12:] / test_batch

            avg_cost[13:15] = conf_mat.get_metrics()

    return avg_cost

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    if os.path.isfile(args.ckpt):
        logging.info(f"Loading checkpoint from {args.ckpt}")
        model.load_state_dict(torch.load(args.ckpt))
    else:
        logging.error(f"Checkpoint not found at {args.ckpt}")
        return


    nyuv2_test_set = NYUv2(root=args.data_path, train=False)
    test_loader = DataLoader(dataset=nyuv2_test_set, batch_size=args.batch_size, shuffle=False)


    avg_cost = evaluate_model(model, test_loader, device)

    logging.info(
        f"Evaluation Results:  MEAN_IOU: {avg_cost[13]:.4f}, "
        f"PIX_ACC: {avg_cost[14]:.4f},  ABS_ERR: {avg_cost[16]:.4f}, "
        f"REL_ERR: {avg_cost[17]:.4f}, MEAN: {avg_cost[19]:.4f}, "
        f"MED: {avg_cost[20]:.4f}, <11.25: {avg_cost[21]:.4f}, <22.5: {avg_cost[22]:.4f}, <30: {avg_cost[23]:.4f}"
    )
    delta_m = delta_fn(np.array([avg_cost[13], avg_cost[14], avg_cost[16], avg_cost[17], avg_cost[19], avg_cost[20], avg_cost[21], avg_cost[22], avg_cost[23]]))
    print(f"average delta_m: {delta_m}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(), "dataset"), help="Path to the NYUv2 dataset.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--model", type=str, default="mtan", choices=["segnet", "mtan"], help="Model type to use.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for DataLoader.")

    args = parser.parse_args()

    main(args)
# ./save/exp_cont_new1/fairgrad_10.pth