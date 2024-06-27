import torch
import Quant
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def FHtoInt4(in_tensor, groupsize):
    n = int(torch.log2(torch.tensor(in_tensor.numel())).item()) 
    k = int(math.log2(groupsize))
    scales = torch.zeros(2**(n-k), dtype=in_tensor.dtype, device=in_tensor.device)
    zeros = torch.zeros(2**(n-k), dtype=in_tensor.dtype, device=in_tensor.device)
    uint8_out = torch.zeros(2**(n-1), dtype=torch.uint8, device=in_tensor.device)
    Quant.FHtoInt4(in_tensor, n, scales, zeros, uint8_out, groupsize) 
    return scales, zeros, uint8_out, n 

def Int4ToFH(scales, zeros, uint8_out, n, groupsize, in_tensor):
    out_tensor = torch.zeros(in_tensor.shape, dtype=in_tensor.dtype, device=in_tensor.device)
    # Quant.Int4ToFH(scales, zeros, uint8_out, n, out_tensor, groupsize)
    return out_tensor

seed = 42
torch.manual_seed(seed)
n = 20
in_tensor_origin = 1*torch.randn(2**n, dtype=torch.float, device="cuda:0")
in_tensor = in_tensor_origin.clone()
groupsize = 128

scales, zeros, uint8_out, n = FHtoInt4(in_tensor, groupsize)

out_tensor = Int4ToFH(scales, zeros, uint8_out, n, groupsize, in_tensor)

quant_mem = scales.numel()*scales.element_size() + zeros.numel()*zeros.element_size() + uint8_out.numel()*uint8_out.element_size()
origin_mem = in_tensor.numel()*in_tensor.element_size()

print(f"Memory used for Int4 data: {quant_mem/2**30} G")
print(f"Memory used for float data: {origin_mem/2**30} G")
print(f"Compression rate for Int4 quantization: {round(quant_mem/origin_mem*100, 2)}%")