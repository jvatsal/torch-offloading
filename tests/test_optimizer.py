import os
import copy
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
import torch
import pytest
from transformers import AutoModel, AutoTokenizer
from torch.testing import assert_close
from typing import Dict, Tuple
from huggingface_hub import login
from optimizer import CPUOffloadOptimizer
import gc
from torch import nn

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
    # inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    inputs = {}
    inputs["input_ids"] = torch.randint(0, 2000, (1, 512), dtype=torch.long, device="cuda")
    inputs["attention_mask"] = torch.ones(1, 512, dtype=torch.long, device="cuda")
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
    # optimizer2.register_model(model2)

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
        torch.testing.assert_close(p1, p2, atol=1e-5, rtol=1e-5)

    torch.cuda.empty_cache()
    gc.collect()
    print("passed!")


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_ao(grad_accum):
    torch.cuda.empty_cache()
    gc.collect()

    # First two layers have terrible arithmetic density
    # Forcefully having long transfers with quick computation, increasing chances that synchronization will lead to test failures
    # 3rd layer checks if it works with non_trainable parameters, shouldn't influence timings
    model1 = nn.Sequential(
        nn.Linear(32, 131072),
        nn.ReLU(),
        nn.Linear(131072, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 64, bias=True),
        nn.ReLU(),
        nn.Linear(64, 128, bias=True),
    )
    model1.to("cuda")
    # make sure it can work in the presence of non-trainable params
    model1[2].requires_grad_(True)
    model2 = copy.deepcopy(model1)

    optim1 = torch.optim.AdamW(model1.parameters())
    optim2 = CPUOffloadOptimizer(
        model2.parameters(),
        torch.optim.AdamW,
    )
    optim2.register_model(model2)

    scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, 100)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optim2, 100)

    rng = torch.Generator(device="cuda")
    rng.manual_seed(42)

    epochs = 1

    # make sure to run both models separately; otherwise, model1 gives additional
    # time for operations in model2 to complete, marking potential race conditions.
    for _ in range(epochs):
        with torch.cuda.nvtx.range("No Offloading"):
            for _ in range(grad_accum):
                x = torch.randn(4, 32, device="cuda", generator=rng)
                model1(x).sum().backward()

            optim1.step()
            optim1.zero_grad()
            scheduler1.step()

    # reset the rng
    # rng.manual_seed(42)
    # for _ in range(2):
    #     for _ in range(grad_accum):
    #         x = torch.randn(4, 32, device="cuda", generator=rng)
    #         model2(x).sum().backward()

    #     optim2.step()
    #     optim2.zero_grad()
    #     scheduler2.step()

    rng.manual_seed(42)
    for _ in range(epochs):
        with torch.cuda.nvtx.range("With Offloading"):
            for i in range(grad_accum):
                x = torch.randn(4, 32, device="cuda", generator=rng)
                if grad_accum > 1 and i < grad_accum - 1:
                    with optim2.no_offload():
                        loss = model2(x).sum()
                        loss.backward()
                else:
                    loss = model2(x).sum()
                    loss.backward()
            optim2.step()
            optim2.zero_grad()
            scheduler2.step()

    optim2.sync_cpu_to_gpu()
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        torch.testing.assert_close(p1, p2)

    torch.cuda.empty_cache()
    gc.collect()
    print("passed!")