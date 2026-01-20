import torch

def _to_device_and_compile(model):
    if torch.backends.mps.is_available():
        device = torch.device("mps") # 使用Apple Silicon GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    if device.type == "mps":
        # Apple Silicon（M1/M2/M3）使用特殊后端
        model = torch.compile(model, backend="aot_eager")
    else:
        model = torch.compile(model)

    return model, device


# -----------------------------------------------------------------
# from utils import _to_device_and_compile

# # 创建一个模型
# model = TransformerLM(...)

# # 调用这个函数
# model, device = _to_device_and_compile(model)

# # 现在 model 已经在 GPU 上，并且编译过了
# # device 告诉我们用的是什么设备（如 "mps" 或 "cuda"）