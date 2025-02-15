import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModel, AutoTokenizer
from torch.autograd.graph import saved_tensors_hooks

from dataclasses import dataclass

@dataclass
class OffloadedActivation:
  tensor: torch.Tensor
  is_offloaded: bool

class OffloadActivations(saved_tensors_hooks):
  def __init__(self, min_offload_size: int =1024, use_pin_memory: bool = True):
    self.min_offload_size = min_offload_size
    self.use_pin_memory = use_pin_memory
    self.offloaded_activations: dict[int, OffloadedActivation] = {}
    self.id = 0
    super().__init__(self.pack_tensor, self.unpack_tensor)

  def pack_tensor(self, activation: torch.Tensor) -> int:
    self.id += 1
    currId = self.id
    size = activation.nelement() * activation.element_size()
    print(f"pack_tensor: id={currId}, size={size} bytes, device={activation.device}")

    if size >= self.min_offload_size and activation.is_cuda:
      cpu_tensor = activation.cpu()  # This is synchronous
      self.offloaded_activations[currId] = OffloadedActivation(cpu_tensor, True)
    else:
      self.offloaded_activations[currId] = OffloadedActivation(activation, False)

    return currId

  def unpack_tensor(self, activation_id: int) -> torch.Tensor:
    if activation_id not in self.offloaded_activations:
      raise RuntimeError("Lost Tensor")
    offloaded_act = self.offloaded_activations.pop(activation_id)
    if offloaded_act.is_offloaded:
      gpu_tensor = offloaded_act.tensor.cuda()
      print(f"unpack_tensor: id={activation_id} restored from CPU to GPU")
      return gpu_tensor
    else:
      print(f"unpack_tensor: id={activation_id} no offloading needed")
      return offloaded_act.tensor

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased").cuda()

inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")

context_manager = OffloadActivations(min_offload_size=1024, use_pin_memory=True)

with context_manager:
  with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True, record_shapes=True) as prof:
        outputs = model(**inputs)
        loss = outputs.last_hidden_state.mean()
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace.json")
