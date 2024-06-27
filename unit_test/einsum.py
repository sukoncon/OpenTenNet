import time
import torch 
from cutensor.torch import EinsumGeneral, EinsumGeneralV2, getOutputShape, defaultAlgo, split_real_imag
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

use_cHalf_ = True
use_cFloat_ = True
eq = "GYEJePRhNfDcoX,gnQCVBiKjaSWdTAOUHLfDcoX->CVBGiKgnQjYaESWJdTAeOUHLPRhN"
kwargs = {"dtype_":"complex32Toririhalf"}

# Prepare data 
split0 = eq.split(",")[0]; split_1_2 = eq.split(",")[1]
split1 = split_1_2.split("->")[0]; split2 = split_1_2.split("->")[1]; 
x_len = len(split0); y_len = len(split1);  ans_len = len(split2)

dtype = torch.complex64
in1 = torch.randn([2]*x_len, dtype=torch.complex64, device = torch.device("cuda", 0))
in2 = torch.randn([2]*y_len, dtype=torch.complex64, device = torch.device("cuda", 0))

if use_cFloat_:
    dtype = torch.complex64
    kwargs["dtype_"] = "complex64"
    out_cFloat = torch.empty(getOutputShape(eq, in1, in2, **kwargs), dtype = dtype, device = in1.device)
    EinsumGeneralV2(out_cFloat, eq, in1, in2, **kwargs)
    allocF_end = torch.cuda.memory_allocated(device=in1.device)
    print(f"allocated memory for FLOAT is  {round((allocF_end)/1024/1024/1024, 2)} GB", flush=True)

if use_cHalf_:
    dtype = torch.complex32
    allocH_beg = torch.cuda.memory_allocated(device=in1.device)
    # initialize data 
    kwargs["dtype_"] = "complex32Toririhalf"
    kwargs["buffer_tensors"] = torch.zeros(in2.numel()*2, dtype = dtype, device = in2.device)
    # import pdb; pdb.set_trace()
    in1_cHalf = torch.zeros(in1.shape, dtype = dtype, device = torch.device("cuda", 0))
    in2_cHalf = torch.zeros(in2.shape, dtype = dtype, device = torch.device("cuda", 0))
    out_cHalf = torch.zeros(getOutputShape(eq, in1_cHalf, in2_cHalf, **kwargs), dtype = dtype, device = in1_cHalf.device)
    # fill data
    in1_cHalf.copy_(in1.to(dtype))
    in2_cHalf.copy_(in2.to(dtype))
    
    # apply cutensor einsum 
    EinsumGeneralV2(out_cHalf, eq, in1_cHalf, in2_cHalf, **kwargs)
    allocH_end = torch.cuda.memory_allocated(device=in1.device)
    print(f"allocated memory for HALF is  {round((allocH_end-allocH_beg)/1024/1024/1024, 2)} GB", flush=True)

    if use_cHalf_:
        dtype = torch.half
        repeat = 10
        for i in range(repeat):
            torch.cuda.synchronize(); time_begin = time.time()
            EinsumGeneralV2(out_cHalf, eq, in1_cHalf, in2_cHalf, **kwargs)
            torch.cuda.synchronize(); time_end = time.time()
        print(f"cutensor (HALF) used time {round((time_end-time_begin), 4)*1000}ms", flush = True)
    if use_cFloat_:
        dtype = torch.complex64
        repeat = 10
        
        for i in range(repeat):
            torch.cuda.synchronize(); time_begin = time.time()
            EinsumGeneralV2(out_cFloat, eq, in1, in2)
            torch.cuda.synchronize(); time_end = time.time()
        print(f"cutensor (FLOAT) used time {round((time_end-time_begin), 4)*1000}ms", flush = True)