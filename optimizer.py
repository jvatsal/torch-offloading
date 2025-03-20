import torch
from torch.optim.optimizer import Optimizer
from typing import Type
from contextlib import contextmanager

"""
TODO: Add gradient accumulation
Step 1:
    * When initializing the optimizer, it copies params (above minimal_size) to CPU
    * These CPU copies are stored and used for optimizer calcs
Step 2:
    * During backward pass, backward_hook is triggered as gradients are computed on GPU
    * For each offloaded param, its gradient is copied from GPU -> corresponding CPU tensor
Step 3:
    * In the step() method, the CPU-based optimizer uses CPU gradients to update CPU copies of params.
    * After the update, the updated CPU params are copied back to GPU
"""
class CPUOffloadOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        minimal_size: int = 4096,
        **kwargs,
    ) -> None:
        
        if optimizer_class is torch.optim.AdamW and "fused" not in kwargs:
            kwargs.update(fused=True)

        params = list(params)
        if len(params) == 0: raise ValueError("empty parameter list")
        if not isinstance(params[0], dict):
            params = [{"params": params}]

        self.minimal_size = minimal_size
        self.device = "cuda"
        self.stream = torch.cuda.Stream()

        # GPU parameters -> CPU copies
        self.gpu2cpuCopies = {}

        # CPU params -> optimizer
        self.optim_dict = {}

        # Maintain order of which param we should do optim step on first
        # Param -> event
        self.queue = {}

        # GPU params
        self.gpu_params = []
        self.gpu_optimizer = None

        # This flag controls whether the backward hook will offload gradients.
        self._offload_enabled = True

        def backward_hook(gpu_param):
            """Intercepted the moment after a gradient is computed on a parameter"""
            if not self._offload_enabled:
                return
            if gpu_param.grad is None: 
                return
            
            cpu_param = self.gpu2cpuCopies[gpu_param]
            
            # make sure backward for this param finishes
            self.stream.wait_stream(torch.cuda.current_stream())

            # copy gradients from GPU -> CPU
            with torch.cuda.stream(self.stream):
                cpu_param.grad.copy_(gpu_param.grad, non_blocking=True)

            if gpu_param in self.queue:
                del self.queue[gpu_param] 

            self.queue[gpu_param] = self.stream.record_event()
            
            # deallocate gpu gradients once GPU -> CPU transfer completes
            # event is recorded on the stream, which will be later syncrhonized in step()
            # Goal: make sure copy has finished
            gpu_param.grad.record_stream(self.stream)
            gpu_param.grad = None

        # Param shift to CPU
        for param_group in params:
            param = param_group.pop("params")
            retained_params = []

            for gpu_param in param:
                if not gpu_param.requires_grad:
                    continue
                if gpu_param.numel() < self.minimal_size:
                    retained_params.append(gpu_param)
                    continue

                # CPU copies of params & grads
                cpu_param = torch.empty_like(gpu_param, device="cpu", pin_memory=True)
                cpu_param.grad = torch.empty_like(cpu_param, pin_memory=True)
                cpu_param.copy_(gpu_param.detach(), non_blocking=True)

                self.gpu2cpuCopies[gpu_param] = cpu_param 

                # Called after backward pass computes and accumulates the grad for this parameter
                gpu_param.register_post_accumulate_grad_hook(backward_hook)
                self.optim_dict[gpu_param] = optimizer_class(
                    [{"params": cpu_param, **param_group}], **kwargs
                )
            
            # Create GPU optimizer group
            if len(retained_params) > 0:
                self.gpu_params.append({"params": retained_params, **param_group})
        
        # Use single optimizer instance for group 
        if len(self.gpu_params) > 0:
            self.gpu_optimizer = optimizer_class(self.gpu_params, **kwargs)

    @contextmanager
    def no_offload(self):
        """Disables offloading for gradient accumulation"""
        old_flag = self._offload_enabled
        self._offload_enabled = False
        try:
            yield
        finally:
            self._offload_enabled = old_flag


    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # do the optimizer step on the GPU for the small parameters
        if self.gpu_optimizer is not None: 
            self.gpu_optimizer.step()
        
        for gpu_param, gpu_grad_event in self.queue.items():
            # wait until GPU -> CPU gradient transfer is completed
            gpu_grad_event.synchronize()
            self.optim_dict[gpu_param].step()

            # submit job to self.stream
            # this guarentees that we only move param CPU->GPU once all backwards finish
            # self.stream will wait for current_stream when moving gradient GPU->CPU
            cpu_param = self.gpu2cpuCopies[gpu_param]
            with torch.cuda.stream(self.stream):
                gpu_param.copy_(cpu_param, non_blocking=True)
            
        # Make sure all parameters are transferred from CPU->GPU before next forward pass
        self.stream.synchronize()
        self.queue.clear()
        return loss


    def zero_grad(self, set_to_none=True):
        """Clears gradients for all parameters"""
        for gpu_param in self.gpu2cpuCopies.keys():
            gpu_param.grad = None

        if self.gpu_optimizer:
            self.gpu_optimizer.zero_grad(set_to_none=set_to_none)
    
    @property
    def param_groups(self):
        """
        Whenever someone accesses optimizer.param_groups (during state saving/loading, LR scheduler, etc.),
        this function returns a combined list of parameter groups from CPU and GPU
        """
        return sum(
            (opt.param_groups for opt in self.optim_dict.values()),
            start=(self.gpu_params),
        )
    
    def state_dict(self):
        """Returns optimizer's state dict"""
        state_dict = {
            "offloaded": [optimizer.state_dict() for optimizer in self.optim_dict.values()]
        }
        if self.gpu_optimizer:
            state_dict["on-device"] = self.gpu_optimizer.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        for opt, s in zip(self.optim_dict.values(), state_dict["offloaded"]):
            opt.load_state_dict(s)
        
        if self.gpu_optimizer:
            self.gpu_optimizer.load_state_dict(state_dict["on-device"])