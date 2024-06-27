import torch
import Quant
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

n = 20
in_tensor_origin = 0.1**5*torch.randn(2**n, dtype=torch.complex32, device="cuda:0")
in_tensor = in_tensor_origin.clone()
quant_tensor = torch.zeros(2**n, dtype=torch.uint8, device="cuda:0")
out_tensor = torch.zeros(2**n, dtype=in_tensor_origin.dtype, device="cuda:0")

# calculate power and scale
pow_idx = 3.

max0 = max(torch.view_as_real(in_tensor).max().abs(), torch.view_as_real(in_tensor).min().abs())
max0.pow_(1./pow_idx)
scale_ = 127./max0.item()


Quant.FHtoInt8(torch.view_as_real(in_tensor).view(-1), quant_tensor.view(-1), pow_idx, scale_)
Quant.Int8ToFH(quant_tensor.view(-1), torch.view_as_real(out_tensor).view(-1), pow_idx, scale_)  

quant_mem = quant_tensor.numel()*quant_tensor.element_size()
origin_mem = in_tensor.numel()*in_tensor.element_size()

print(f"Memory used for Int8 data: {quant_mem/2**30} G")
print(f"Memory used for float data: {origin_mem/2**30} G")
print(f"Compression rate for Int8 quantization: {round(quant_mem/origin_mem*100, 2)}%")