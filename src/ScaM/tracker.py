from collections.abc import Mapping
from .metrics import Metric

class model_tracker:
    def __init__(self, model, layer_names, cust_select_fn=None):
        """
        Initialize with a model and a list of layer names.
        """
        self.model = model
        self.layer_names = layer_names
        self.cust_select_fn = [lambda x: x for _ in layer_names] if cust_select_fn is None else cust_select_fn
        assert len(self.layer_names) == len(self.cust_select_fn) # must have a token selection function for each layer
        self.hooks = []
        self.activations = {}
        self.metrics = {}

        # activate hooks by default
        self.activate_hooks()

    def _hook_fn(self, layer_name, cust_select_fn):
        def hook(module, input, output):
            self.activations[layer_name] = cust_select_fn(output.detach()) # could also be the input?
        return hook
    
    def activate_hooks(self):
        """
        Register hooks to capture activations.
        """
        for layer_name, sel_fn in zip(self.layer_names, self.cust_select_fn):
            layer = dict([*self.model.named_modules()])[layer_name]
            hook = layer.register_forward_hook(self._hook_fn(layer_name, sel_fn))
            self.hooks.append(hook)

    def deactivate_hooks(self):
        """
        Remove all registered hooks.
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def add_metric(self, metric: Metric, *args, **kwargs):
        """
        Add a metric function to be computed.
        """
        self.metrics[metric.__name__] = metric

    def compute_metrics(self, label_dict: Mapping):
        """
        Compute all registered metrics.
        """
        
        out = {}
        for metric_name, metric_instance in self.metrics.items():
            out[metric_name] = metric_instance(self.activations, label_dict)
        return out