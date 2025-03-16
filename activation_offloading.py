import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoModel, AutoTokenizer
from torch.autograd.graph import saved_tensors_hooks

from dataclasses import dataclass

@dataclass
class OffloadedActivation:
  tensor: torch.Tensor
  is_offloaded: bool
  event: torch.cuda.Event = None

class OffloadActivations(saved_tensors_hooks):
  def __init__(self, min_offload_size: int=1024, use_pin_memory: bool = True):
    self.min_offload_size = min_offload_size
    self.use_pin_memory = use_pin_memory
    self.offloaded_activations: dict[int, OffloadedActivation] = {}
    self.id = 0
    self.comm_stream = torch.cuda.Stream()
    super().__init__(self.pack_tensor, self.unpack_tensor)

  def pack_tensor(self, activation: torch.Tensor) -> int:
    self.id += 1
    currId = self.id
    size = activation.nelement() * activation.element_size()

    if (
      size >= self.min_offload_size 
      and activation.is_cuda 
      and not isinstance(activation, (torch.nn.Parameter, torch.nn.Buffer))
    ):
      self.comm_stream.wait_stream(torch.cuda.default_stream())
      
      with torch.cuda.stream(self.comm_stream):
        cpu_tensor = torch.empty_like(activation, pin_memory=self.use_pin_memory, device="cpu")
        cpu_tensor.copy_(activation, non_blocking=True)
        event = self.comm_stream.record_event()
        self.offloaded_activations[currId] = OffloadedActivation(cpu_tensor, True, event=event)
    else:
      self.offloaded_activations[currId] = OffloadedActivation(activation, False)
    return currId

  def unpack_tensor(self, activation_id: int) -> torch.Tensor:
    self.comm_stream.wait_stream(torch.cuda.default_stream())
    
    if activation_id not in self.offloaded_activations:
      raise RuntimeError("Lost Tensor")
    offloaded_act = self.offloaded_activations.pop(activation_id)
    if offloaded_act.is_offloaded:
      with torch.cuda.stream(self.comm_stream), torch.cuda.nvtx.range(f"offloading {activation_id}"):
        self.comm_stream.wait_event(offloaded_act.event)
        gpu_tensor = offloaded_act.tensor.to("cuda", non_blocking=True)

      torch.cuda.default_stream().wait_stream(self.comm_stream)
      return gpu_tensor
    else:
      return offloaded_act.tensor