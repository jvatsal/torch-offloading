import os
import copy
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import pytest
from transformers import AutoModel, AutoTokenizer
from torch.testing import assert_close
from typing import Dict, Tuple
import contextlib
import gc
from torch import nn
from huggingface_hub import login
login(token=os.environ["HUGGINGFACE_TOKEN"])

from activation_offloading import OffloadActivations

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False

def get_model() -> Tuple[AutoModel, Dict[str, torch.Tensor]]:
  """Load pretrained model and tokenize sample input. Returns model & inputs"""
  tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
  model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to(dtype=torch.float32, device="cuda")
  # model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to(dtype=torch.bfloat16, device="cuda")
  model.train()
  inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
  return model, inputs

def run_model(
    model: AutoModel, 
    inputs: Dict[str, torch.Tensor], 
    use_offloading: bool = False
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
  """
  Runs a forward/backward pass on the model with the provided inputs.
  If use_offloading is True, the model is run under the activation offloading context.
  Returns the loss and a dict mapping parameter names to their gradients.
  """
  model.zero_grad()

  with OffloadActivations() if use_offloading else contextlib.nullcontext():
    model.gradient_checkpointing_enable({"use_reentrant": False})

    outputs = model(**inputs)
    loss = outputs.last_hidden_state.mean()
    loss.backward()

  grad = {}
  for name, param in model.named_parameters():
    if param.grad is not None:
      grad[name] = param.grad.clone()

  return loss, grad

def test_activation_offloading_accuracy() -> None:
  """Runs the model with and without activation offloading and compares the loss and gradients."""

  model_no_offloading, inputs_no_offloading = get_model()
  model_with_offloading = copy.deepcopy(model_no_offloading)
  inputs_with_offloading = copy.deepcopy(inputs_no_offloading)

  torch.manual_seed(2024)
  loss_no_offloading, grad_no_offloading = run_model(model_no_offloading, inputs_no_offloading, use_offloading=False)
  torch.manual_seed(2024)
  loss_with_offloading, grad_with_offloading = run_model(model_with_offloading, inputs_with_offloading, use_offloading=True)

  atol = rtol = 1e-5

  # Compare loss values
  print("---------")
  print("Loss Test")
  print("---------")
  assert_close(loss_no_offloading, loss_with_offloading, atol=atol, rtol=rtol)
  print("passed!")

  # Verify that both runs had gradients for same set of parameters
  assert set(grad_no_offloading.keys()) == set(grad_with_offloading.keys())

  # Compare gradients for each model parameter
  print("---------")
  print("Gradient Test")
  print("---------")
  for name in grad_no_offloading:
    assert_close(grad_no_offloading[name], grad_with_offloading[name], atol=atol, rtol=rtol)

  del model_no_offloading, model_with_offloading, inputs_no_offloading, inputs_with_offloading
  del loss_no_offloading, loss_with_offloading, grad_no_offloading, grad_with_offloading

  torch.cuda.empty_cache()
  gc.collect()
  print(torch.cuda.memory_summary(device=None, abbreviated=False))
  print("passed!")
