import torch
import pytest
from transformers import AutoModel, AutoTokenizer
from torch.testing import assert_close

from activation_offloading import OffloadActivations


def get_model():
  """Load pretrained model and tokenize sample input. Returns model & inputs"""
  tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
  model = AutoModel.from_pretrained("bert-base-uncased").cuda()
  model.train()
  inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
  return model, inputs

def run_model(model, inputs, use_offloading=False):
  """
  Runs a forward/backward pass on the model with the provided inputs.
  If use_offloading is True, the model is run under the activation offloading context.
  Returns the loss and a dict mapping parameter names to their gradients.
  """
  # zero out existing gradients
  model.zero_grad()

  if use_offloading:
    with OffloadActivations(min_offload_size=1024, use_pin_memory=True):
      outputs = model(**inputs)
      loss = outputs.last_hidden_state.mean()
      loss.backward()
  else:
    outputs = model(**inputs)
    loss = outputs.last_hidden_state.mean()
    loss.backward()

  grad = {}
  for name, param in model.named_parameters():
    if param.grad is not None:
      grad[name] = param.grad

  return loss, grad

def test_activation_offloading_accuracy():
  """Runs the model with and without activation offloading and compares the loss and gradients."""

  model_no_offloading, inputs_no_offloading = get_model()
  model_with_offloading, inputs_with_offloading = get_model()

  loss_no_offloading, grad_no_offloading = run_model(model_no_offloading, inputs_no_offloading, use_offloading=False)
  loss_with_offloading, grad_with_offloading = run_model(model_with_offloading, inputs_with_offloading, use_offloading=True)

  # Compare loss values
  assert_close(loss_no_offloading, loss_with_offloading, atol=1e-5, rtol=1e-5)

  # Check that both runs had gradients for same set of parameters
  assert set(grad_no_offloading.keys()) == set(grad_with_offloading.keys())

  # Compare gradients for each model parameter
  for name in grad_no_offloading:
    assert_close(grad_no_offloading[name], grad_with_offloading[name], atol=1e-5, rtol=1e-5)