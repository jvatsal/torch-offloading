# Offloading Model States in Pytorch

This repository demonstrates implementations of offloading layers and their tensors between GPU and CPU during training in Pytorch. Currently, it includes:
- simple_offloading.py:
    - Forward and backward hooks for offloading model parameters and gradients.
    - Integration with PyTorch Profiler to track memory and runtime details.
## Installation
```pip install -r requirements.txt```
## Usage
Run the example with BERT:
```python simple_offloading.py```
The profile will be saved as `trace.json` and can be viewed in a trace visualizer.
## Current Work
- Implementing smarter, more granular offloading of all model states: model parameters, activations, gradients, and optimizer states
- Overlapping computation/communication using CUDA streams
