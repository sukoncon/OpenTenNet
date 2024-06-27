import torch
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--use_int8", type=int, default=1, help="0: False, 1: True")
args = parser.parse_args()

nnodes = 1
ntask = 15
tasks_per_node = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pt_source = "results/complex32_int8/ntask{}".format(ntask - 1) if args.use_int8 == 1 else "results/complex32_complex32/ntask{}".format(ntask - 1)
suzhongling_pt_source = "results/complex64/ntask{}".format(ntask - 1)

if not os.path.exists(f"{pt_source}/concate.pt"):
    pt = torch.load("{}/rank0.pt".format(pt_source))
    res = pt.to(device)
    for i in range(1, 32):
        pt = torch.load("{}/rank{}.pt".format(pt_source, i)).to(device)
        res = torch.cat((res, pt), dim = 0)

    torch.save(res.cpu(), f"{pt_source}/concate.pt")

if not os.path.exists(f"{suzhongling_pt_source}/concate.pt"):
    pt = torch.load("{}/rank0.pt".format(suzhongling_pt_source))
    res = pt.to(device)
    for i in range(1, 128):
        pt = torch.load("{}/rank{}.pt".format(suzhongling_pt_source, i)).to(device)
        res = torch.cat((res, pt), dim = 0)

    torch.save(res.cpu(), f"{suzhongling_pt_source}/concate.pt")

###############################################################################
############# fidelity ########################################################

result = torch.load("{}/concate.pt".format(pt_source))
gt = torch.load("{}/concate.pt".format(suzhongling_pt_source))
fidelity = (
    (gt.conj() @ result.reshape(-1)).abs() /
    (gt.abs().square().sum().sqrt() * result.abs().square().sum().sqrt())
).square().item()
print(f"fidelity: {round(fidelity, 8)}")