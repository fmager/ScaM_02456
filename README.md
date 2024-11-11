# ScaM - Scalar Metrics for Latent Spaces

**ScaM** is a Python package designed to quantify properties of a latent space during the training process of a given model.

## Features

- **Model Agnostic**: Works with any PyTorch model.
- **Layer-Specific Metrics**: Compute metrics for specified layers of the model.
- **Static Methods**: Metric functions are implemented as static methods for flexible usage.

## Installation

To install the package, use:

```bash
pip install scam
```

## Getting started

```python
import torch
from ScaM import scam

# Assuming you have a model and input data
model = ...  # Your model

layer_names = ['encoder', 'decoder']  # Replace with your layer names
cust_select_fn = [lambda x: x.mean(dim=1), lambda x: x[:, 0, :]] # A custom select for each layer, e.g. global pooling, token selection.

# define a model tracker
tracker.model_tracker(model, layer_names, cust_select_fn)

# define a metric, e.g. Signal to Noise ratio, using euclidean distances
metric = metrics.SNR(dist_fn='euclidean')

x = ...  # Your input data
label_dict = ...  # Your label dictionary with some meta information e.g. {'A': [0, 2, 1, 2, 2, 4, ... ], 'B': ['cat1, 'cat2', 'cat1', ...]}

# Forward pass
_ = model(x)

# Compute the metrics
results = tracker.compute_metrics(label_dict) # returns a dictionary: {'SNR': {'encoder': {'A': tensor(27.4428)}, 'decoder': {'A': tensor(21.2904)}}}

```

