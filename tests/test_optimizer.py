import os
import copy
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import tempfile
import torch
import pytest
from transformers import AutoModel, AutoTokenizer
from torch.testing import assert_close
from typing import Dict, Tuple
from huggingface_hub import login
from optimizer import CPUOffloadOptimizer
import gc

login(token=os.environ["HUGGINGFACE_TOKEN"])

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def get_model() -> Tuple[AutoModel, Dict[str, torch.Tensor]]:
    """Load pretrained LLaMA model and tokenize sample input."""
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to(dtype=torch.float32, device="cuda")
    # model = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B").to(dtype=torch.bfloat16, device="cuda")
    model.train()
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    return model, inputs

@pytest.mark.parametrize("grad_accum", [1, 2])
def test_optim_cpu_offload_correctness(grad_accum):
    """Tests optimizer correctness by ensuring parameter updates match."""
    torch.cuda.empty_cache()
    gc.collect()
    model1, inputs1 = get_model()  
    model2 = copy.deepcopy(model1)
    inputs2 = {key: val.clone() for key, val in inputs1.items()}

    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
    optimizer2 = CPUOffloadOptimizer(model2.parameters(), torch.optim.AdamW, lr=1e-3)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer1, 100)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer2, 100)

    rng = torch.Generator(device="cuda")
    rng.manual_seed(42)

    for _ in range(2):
        for _ in range(grad_accum):
            outputs = model1(**inputs1)
            outputs.last_hidden_state.mean().backward()

        optimizer1.step()
        optimizer1.zero_grad()
        scheduler1.step()

    rng.manual_seed(42)

    # for _ in range(2):
    #     for _ in range(grad_accum):
    #         outputs = model2(**inputs2)
    #         outputs.last_hidden_state.mean().backward()

    #     optimizer2.step()
    #     optimizer2.zero_grad()
    #     scheduler2.step()

    for _ in range(2):
        for i in range(grad_accum):
            if grad_accum > 1 and i < grad_accum - 1:
                with optimizer2.no_offload():
                    outputs = model2(**inputs2)
                    outputs.last_hidden_state.mean().backward()
            else:
                outputs = model2(**inputs2)
                outputs.last_hidden_state.mean().backward()
        optimizer2.step()
        optimizer2.zero_grad()
        scheduler2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2, atol=1e-3, rtol=1e-3)
    
    torch.cuda.empty_cache()
    gc.collect()
    print("passed!")