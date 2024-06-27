import torch
import Quant

import math

n = 20
in_tensor_origin = 0.1*torch.randn(2**n, dtype=torch.complex64, device="cuda:0")
in_tensor = in_tensor_origin.clone()
quant_tensor = torch.zeros(2**(n+1), dtype=torch.half, device="cuda:0")
out_tensor = torch.zeros(2**n, dtype=in_tensor_origin.dtype, device="cuda:0")

# calculate power and scale
pow_idx = 0

max0 = max(torch.view_as_real(in_tensor).max().abs(), torch.view_as_real(in_tensor).min().abs())
# max0.pow_(1./pow_idx)
scale_ = 65000./max0.item()


Quant.Float2Half(torch.view_as_real(in_tensor).view(-1), quant_tensor.view(-1), pow_idx, scale_)
Quant.Half2Float(quant_tensor.view(-1), torch.view_as_real(out_tensor).view(-1), pow_idx, scale_)  

quant_mem = quant_tensor.numel()*quant_tensor.element_size()
origin_mem = in_tensor.numel()*in_tensor.element_size() 

print(f"Memory used for half data: {quant_mem/2**30} G")
print(f"Memory used for complex float data: {origin_mem/2**30} G")
print(f"Compression rate for half quantization: {round(quant_mem/origin_mem*100, 2)}%")