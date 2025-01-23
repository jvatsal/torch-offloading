# Implement simple offloading (model parameters and gradients) for BERT
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertModel, AutoModel, AutoTokenizer

forward_record_objects = {}
backward_record_objects = {}

def forward_pre_hook(module, inputs, idx):
    module.to("cuda")
  
    if isinstance(inputs, tuple):
        inputs = tuple(i.to("cuda") if isinstance(i, torch.Tensor) else i for i in inputs)
    elif isinstance(inputs, dict):
        inputs = {key: val.to("cuda") if isinstance(val, torch.Tensor) else val for key, val in inputs.items()}
    elif isinstance(inputs, torch.Tensor):
        inputs = inputs.to("cuda")

    rf_obj = record_function(f"Layer_{idx}_Forward")
    rf_obj.__enter__()
    forward_record_objects[idx] = rf_obj
    
    return inputs

def forward_post_hook(module, inputs, outputs, idx):
    rf_obj = forward_record_objects.pop(idx, None)
    if rf_obj is not None:
        rf_obj.__exit__(None, None, None)
        
    module.to("cpu")
  
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.to("cpu")
    elif isinstance(outputs, tuple):
        outputs = tuple(out.to("cpu") if isinstance(out, torch.Tensor) else out for out in outputs)
    elif isinstance(outputs, dict):
        outputs = {key: val.to("cpu") if isinstance(val, torch.Tensor) else val for key, val in outputs.items()}
    return outputs

def backward_pre_hook(module, grad_output, idx):
    module.to("cuda")

    rf_obj = record_function(f"Layer_{idx}_Backward")
    rf_obj.__enter__()
    backward_record_objects[idx] = rf_obj

    return grad_output

def backward_post_hook(module, grad_input, grad_output, idx):
    module.to("cpu")
  
    rf_obj = backward_record_objects.pop(idx, None)
    if rf_obj is not None:
        rf_obj.__exit__(None, None, None)

    return grad_input

def make_forward_pre_hook(idx):
    def hook(module, layer_input):
        return forward_pre_hook(module, layer_input, idx)
    return hook

def make_forward_post_hook(idx):
    def hook(module, layer_input, layer_output):
        return forward_post_hook(module, layer_input, layer_output, idx)
    return hook

def make_backward_pre_hook(idx):
    def hook(module, grad_output):
        return backward_pre_hook(module, grad_output, idx)
    return hook

def make_backward_post_hook(idx):
    def hook(module, grad_input, grad_output):
        return backward_post_hook(module, grad_input, grad_output, idx)
    return hook

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

for i, layer_module in enumerate(model.encoder.layer):
    layer_module.register_forward_pre_hook(make_forward_pre_hook(i))
    layer_module.register_forward_hook(make_forward_post_hook(i))
    layer_module.register_full_backward_pre_hook(make_backward_pre_hook(i))
    layer_module.register_full_backward_hook(make_backward_post_hook(i))

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
for key, val in inputs.items():
    if val.dtype in (torch.float32, torch.float64):
        inputs[key] = val.requires_grad_()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    outputs = model(**inputs)
    loss = outputs.last_hidden_state.mean()
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
