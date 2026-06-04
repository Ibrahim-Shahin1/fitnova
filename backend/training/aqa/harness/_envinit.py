# F8 / D13: must execute before any 'import torch' so CUBLAS_WORKSPACE_CONFIG is set before CUDA context creation; otherwise torch.use_deterministic_algorithms(True) raises.
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
