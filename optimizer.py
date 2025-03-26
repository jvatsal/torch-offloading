import torch
from torch.optim.optimizer import Optimizer
from typing import Type
from contextlib import contextmanager

"""
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

        # A map to track when GPU->CPU gradient copies have completed.
        # Param -> event
        self.queue = {}

        # GPU params
        self.gpu_params = []
        self.gpu_optimizer = None

        # This flag controls whether the backward hook will offload gradients.
        self.used_gradient_accum = False
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
            print(f"[{gpu_param.shape}] gradient offloaded")
            
            # deallocate gpu gradients once GPU -> CPU transfer completes
            # event is recorded on the stream, which will be later syncrhonized in step()
            # Goal: make sure copy has finished
            gpu_param.grad.record_stream(self.stream)
            print("cleared gradient")
            gpu_param.grad = None

            # # Free the GPU param copy now that gradient is offloaded.
            gpu_param.data = torch.empty(0, device=gpu_param.device, dtype=cpu_param.dtype)

        # param shift to CPU
        for param_group in params:
            param = param_group.pop("params")
            retained_params = []

            for gpu_param in param:
                if not gpu_param.requires_grad:
                    retained_params.append(gpu_param)
                    continue
                if gpu_param.numel() < self.minimal_size:
                    retained_params.append(gpu_param)
                    continue

                # create persistent CPU master copy
                cpu_param = torch.empty_like(gpu_param, device="cpu", pin_memory=True)
                cpu_param.grad = torch.empty_like(cpu_param, pin_memory=True)
                cpu_param.copy_(gpu_param.detach(), non_blocking=True)

                self.gpu2cpuCopies[gpu_param] = cpu_param 

                # immediately free GPU copy
                gpu_param.data = torch.empty(0, device=gpu_param.device, dtype=cpu_param.dtype)

                # called after backward pass computes and accumulates the grad for this parameter
                gpu_param.register_post_accumulate_grad_hook(backward_hook)

                # make CPU optimizer for this parameter
                self.optim_dict[gpu_param] = optimizer_class(
                    [{"params": cpu_param, **param_group}], **kwargs
                )
            
            # store small params on gpu
            if len(retained_params) > 0:
                self.gpu_params.append({"params": retained_params, **param_group})
        
        # use single optimizer instance for small gpu params 
        if len(self.gpu_params) > 0:
            self.gpu_optimizer = optimizer_class(self.gpu_params, **kwargs)

    @contextmanager
    def no_offload(self):
        """Disables offloading for gradient accumulation"""
        self.used_gradient_accum = True
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
            
        # make sure all parameters are transferred from CPU->GPU before next forward pass
        # TODO: add a wait_stream here
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

    def load_params_to_gpu(self, module):
        """Before forward, load master CPU copy to GPU if needed"""
        for gpu_param in module.parameters():
            if gpu_param in self.gpu2cpuCopies:
                cpu_param = self.gpu2cpuCopies[gpu_param]
                if gpu_param.data.numel() == 0:
                    gpu_param.data = torch.empty_like(cpu_param, device=gpu_param.device)
                    gpu_param.data.copy_(cpu_param, non_blocking=True)

    def offload_params_to_cpu(self, module):
        """After forward, simply clear the GPU params to free memory"""
        for gpu_param in module.parameters():
            if gpu_param in self.gpu2cpuCopies:
                cpu_param = self.gpu2cpuCopies[gpu_param]
                gpu_param.data = torch.empty(0, device=gpu_param.device, dtype=cpu_param.dtype)


    def register_model(self, model):
        """Attach hooks to modules for parameter offloading"""
        for m in model.modules():
            if any(True for _ in m.parameters(recurse=False)):
                m.register_forward_pre_hook(self.forward_prehook)
                m.register_forward_hook(self.forward_hook)
                m.register_full_backward_pre_hook(self.backward_pre_hook)
                # m.register_full_backward_hook(self.backward_post_hook)

    def check_no_gpu_params(self, module):
        for gpu_param in module.parameters():
            if gpu_param in self.gpu2cpuCopies:
                assert gpu_param.data.numel() == 0 or torch.all(gpu_param.data == 0), f"GPU params not cleared"
    
    def check_exists_gpu_params(self, module):
        for gpu_param in module.parameters():
            if gpu_param in self.gpu2cpuCopies:
                assert gpu_param.data.numel() != 0 and not torch.all(gpu_param.data == 0), f"Missing GPU param"

    def forward_prehook(self, module, inputs):
        """Before forward, load parameters CPU->GPU"""
        if not self.used_gradient_accum: self.check_no_gpu_params(module)
        self.load_params_to_gpu(module)
        if not self.used_gradient_accum: self.check_exists_gpu_params(module)

    def forward_hook(self, module, inputs, output):
        """After forward, clear GPU params"""
        # def reload_on_grad(grad):
        #     self.load_params_to_gpu(module)
        #     return grad

        # output.register_hook(reload_on_grad)

        if not self.used_gradient_accum: self.check_exists_gpu_params(module)
        self.offload_params_to_cpu(module)
        if not self.used_gradient_accum: self.check_no_gpu_params(module)
    

    def backward_pre_hook(self, module, grad_input):
        """Before backward, reload params CPU->GPU"""
        print("pre_hook called")
        # for param in module.parameters():
        #     if param in self.gpu2cpuCopies:
        #         print(f"PREHOOK: Gradient for param before: {param.grad}")
                
        if not self.used_gradient_accum: self.check_no_gpu_params(module)
        self.load_params_to_gpu(module)
        if not self.used_gradient_accum: self.check_exists_gpu_params(module)

        # for param in module.parameters():
        #     if param in self.gpu2cpuCopies:
        #         print(f"PREHOOK: Gradient for param after: {param.grad}")

    # Deprecate? 
    # It's more intuitive to clear the parameter right after it's done being needed (backward_hook)
    def backward_post_hook(self, module, grad_input, grad_output):
        """After backward, clear GPU params"""
        print("post_hook called")
        # for param in module.parameters():
        #     if param in self.gpu2cpuCopies:
        #         print(f"POSTHOOK: Gradient for param before: {param.grad}")
        if not self.used_gradient_accum: self.check_exists_gpu_params(module)
        self.offload_params_to_cpu(module)
        if not self.used_gradient_accum: self.check_no_gpu_params(module)
        # for param in module.parameters():
        #     if param in self.gpu2cpuCopies:
        #         print(f"POSTHOOK: Gradient for param after: {param.grad}")
        

    @torch.no_grad()
    def sync_cpu_to_gpu(self):
        """Copies updated CPU master copies back to GPU parameters."""
        for gpu_param, cpu_param in self.gpu2cpuCopies.items():
            if gpu_param.data.numel() == 0:
                gpu_param.data = torch.empty_like(cpu_param, device=gpu_param.device)
            gpu_param.data.copy_(cpu_param, non_blocking=True)
