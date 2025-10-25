import os
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from experiments.celeba.data import CelebaDataset
from experiments.celeba.models import Network
from experiments.utils import get_device, set_seed
from experiments.celeba.delta_m import delta_fn
import logging

class CelebaMetrics():
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.tp = 0.0 
        self.fp = 0.0 
        self.fn = 0.0 
        
    def incr(self, y_preds, ys):
        y_preds  = torch.stack(y_preds).detach() # (40, batch, 1)
        ys       = torch.stack(ys).detach()      # (40, batch, 1)
        y_preds  = y_preds.gt(0.5).float()
        self.tp += (y_preds * ys).sum([1, 2]) # (40,)
        self.fp += (y_preds * (1 - ys)).sum([1, 2])
        self.fn += ((1 - y_preds) * ys).sum([1, 2])
                
    def result(self):
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall    = self.tp / (self.tp + self.fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.cpu().numpy()

def evaluate(model, data_loader, device):
    model.eval()
    metrics = CelebaMetrics()
    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_ = model(x)
            metrics.incr(y_, y)
    return metrics.result()

def main(ckpt, data_path, batch_size, device):
    # Load model
    model = Network().to(device)
    model.load_state_dict(torch.load(ckpt))
    logging.info(f"Model loaded from {ckpt}")
    # Load test dataset
    test_set = CelebaDataset(data_dir=data_path, split='test')

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Evaluate
    test_f1 = evaluate(model, test_loader, device)

    print(f"Test F1 Score: {test_f1.mean():.4f}")
    delta_m = delta_fn(test_f1)
    print(f"Average Delta_m: {delta_m}")
    
if __name__ == "__main__":
    parser = ArgumentParser("Evaluate Celeba Model")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--data_path", type=str, default=os.path.join(os.getcwd(), "dataset"), help="Path to the dataset")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for evaluation")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device IDs (comma-separated)")
    args = parser.parse_args()

    # Set seed
    set_seed(42)  # You can set your own seed here
    device = get_device(gpus=args.gpu)
    main(ckpt=args.ckpt,
         data_path=args.data_path,
         batch_size=args.batch_size,
         device=device)
