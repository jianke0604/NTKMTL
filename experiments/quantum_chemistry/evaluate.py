from argparse import ArgumentParser
import os
import logging
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import numpy as np

from experiments.quantum_chemistry.models import Net
from experiments.quantum_chemistry.utils import (
    Complete,
    MyTransform,
    delta_fn,
    multiply_indx,
    delta_fn_arr,
    qm9_target_dict
)
from experiments.quantum_chemistry.utils import target_idx as targets
from experiments.utils import (
    common_parser,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)

set_logger()

@torch.no_grad()
def evaluate(model, loader, std, scale_target, device):
    model.eval()
    data_size = 0.0
    task_losses = 0.0
    for data in loader:
        data = data.to(device)
        out = model(data)
        if scale_target:
            task_losses += F.l1_loss(
                out * std.to(device), data.y * std.to(device), reduction="none"
            ).sum(0)  # MAE
        else:
            task_losses += F.l1_loss(out, data.y, reduction="none").sum(0)  # MAE
        data_size += len(data.y)

    avg_task_losses = task_losses / data_size

    # Report meV instead of eV.
    avg_task_losses = avg_task_losses.detach().cpu().numpy()
    avg_task_losses[multiply_indx] *= 1000

    delta_m = delta_fn(avg_task_losses)
    
    return dict(
        avg_loss=avg_task_losses.mean(),
        avg_task_losses=avg_task_losses,
        delta_m=delta_m,
    )


def main(
    data_path: str,
    batch_size: int,
    device: torch.device,
    ckpt_path: str,
    targets: list,
    scale_target: bool = True,
):
    dim = 64
    model = Net(n_tasks=len(targets), num_features=11, dim=dim).to(device)
    
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    
    logging.info(f"Loading model weights from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    
    transform = T.Compose([MyTransform(targets), Complete(), T.Distance(norm=False)])
    dataset = QM9(data_path, transform=transform).shuffle()

    # Use the test dataset for evaluation.
    test_dataset = dataset[:10000]
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    std = None
    if scale_target:
        mean = test_dataset.data.y[:, targets].mean(dim=0, keepdim=True)
        std = test_dataset.data.y[:, targets].std(dim=0, keepdim=True)
        dataset.data.y[:, targets] = (dataset.data.y[:, targets] - mean) / std

    test_loss_dict = evaluate(model, test_loader, std=std, scale_target=scale_target, device=device)

    # Print evaluation results
    # print("Test Loss (Average):", test_loss_dict["avg_loss"])
    np.set_printoptions(precision=4)
    print("Test Loss (Per Task):", test_loss_dict["avg_task_losses"])
    print("Delta m:", test_loss_dict["delta_m"])



if __name__ == "__main__":
    parser = ArgumentParser("QM9 Evaluation", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        batch_size=120,
    )
    parser.add_argument("--scale-y", default=True, type=str2bool)
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint file path to load model weights from.")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Get device for evaluation (GPU or CPU)
    device = get_device(gpus=args.gpu)

    # Run evaluation
    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        ckpt_path=args.ckpt,
        targets=targets,
        scale_target=args.scale_y,
    )
