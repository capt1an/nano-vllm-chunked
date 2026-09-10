import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from huggingface_hub import snapshot_download

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    if not os.path.exists(path):
        path = snapshot_download(repo_id=path, local_files_only=True)
        print(f"[nano-vLLM] 成功从本地 hfcache 提取路径: {path}")

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    is_weight_local = getattr(model, "is_weight_local", None)
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # EP ranks do not instantiate remote experts. Check ownership
                # before materializing the checkpoint tensor on CPU.
                if is_weight_local is not None and not is_weight_local(weight_name):
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

    for name, module in model.named_modules():
        validate = getattr(module, "validate_loaded", None)
        if validate is not None:
            try:
                validate()
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
