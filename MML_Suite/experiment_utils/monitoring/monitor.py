from pathlib import Path

import torch.nn as nn
from config.monitor_config import MonitorConfig
from experiment_utils.printing import get_console

from .storage import MonitorStorage

console = get_console()


class ExperimentMonitor:
    """Monitor for tracking internal model dynamics during training."""

    def __init__(
        self,
        config: MonitorConfig,
        model: nn.Module,
        log_dir: Path,
    ):
        self.config = config
        self.model = model
        self.log_dir = Path(log_dir)

        # Initialize storage
        self.storage = MonitorStorage(
            path=self.log_dir,
            config=self.config,
        )

        # State tracking
        self.step_count = 0
        self.epoch = 0
        self.hooks = {}

        self._attach_hooks()

    def _should_track(self, interval: int) -> bool:
        """Determine if we should track metrics this step."""
        return self.step_count % interval == 0

    def _attach_hooks(self):
        """Attach monitoring hooks to model."""

        def _get_activation(name: str):
            def hook(module, input, output):
                if self._should_track(self.config.activation_interval):
                    # For LSTM layers, output is (output, (hidden_state, cell_state))
                    if isinstance(module, nn.LSTM):
                        output_seq, (hidden, cell) = output
                        self.storage.add_to_buffer(
                            "activations",
                            f"epoch_{self.epoch}/step_{self.step_count}/{name}/output",
                            output_seq,
                            {"epoch": self.epoch, "step": self.step_count},
                        )
                        self.storage.add_to_buffer(
                            "activations",
                            f"epoch_{self.epoch}/step_{self.step_count}/{name}/hidden_state",
                            hidden,
                            {"epoch": self.epoch, "step": self.step_count},
                        )
                        self.storage.add_to_buffer(
                            "activations",
                            f"epoch_{self.epoch}/step_{self.step_count}/{name}/cell_state",
                            cell,
                            {"epoch": self.epoch, "step": self.step_count},
                        )
                    else:
                        # Regular layers
                        self.storage.add_to_buffer(
                            "activations",
                            f"epoch_{self.epoch}/step_{self.step_count}/{name}",
                            output,
                            {"epoch": self.epoch, "step": self.step_count},
                        )

            return hook

        def _get_gradient(name: str):
            def hook(grad):
                if self._should_track(self.config.gradient_interval):
                    self.storage.add_to_buffer(
                        "gradients",
                        f"epoch_{self.epoch}/step_{self.step_count}/{name}",
                        grad,
                        {"epoch": self.epoch, "step": self.step_count},
                    )

            return hook

        # Attach hooks based on config
        for name, module in self.model.named_modules():
            # Skip excluded layers
            if any(excl in name for excl in self.config.exclude_layers):
                continue

            # Skip if not in included layers (if specified)
            if self.config.include_layers and not any(incl in name for incl in self.config.include_layers):
                continue

            # Activation tracking
            if self.config.enable_activation_tracking:
                self.hooks[f"{name}_activation"] = module.register_forward_hook(_get_activation(name))

            # Gradient tracking
            if self.config.enable_gradient_tracking:
                # For regular layers with weights
                if hasattr(module, "weight") and module.weight is not None:
                    self.hooks[f"{name}_gradient_weight"] = module.weight.register_hook(_get_gradient(f"{name}.weight"))
                    if hasattr(module, "bias") and module.bias is not None:
                        self.hooks[f"{name}_gradient_bias"] = module.bias.register_hook(_get_gradient(f"{name}.bias"))

                # For LSTM layers
                if isinstance(module, nn.LSTM):
                    # Track all LSTM parameters (ih, hh, and biases)
                    for i, param in enumerate(module.parameters()):
                        param_name = f"{name}.param_{i}"
                        self.hooks[f"{name}_gradient_param_{i}"] = param.register_hook(_get_gradient(param_name))

    def _track_weights(self):
        """Track weight distributions."""
        console.print("Tracking weights...", end=" ")

        for name, module in self.model.named_modules():
            # For regular layers
            if hasattr(module, "weight") and module.weight is not None:
                self.storage.add_to_buffer(
                    "weights", f"epoch_{self.epoch}/{name}.weight", module.weight.data, {"epoch": self.epoch}
                )
                if hasattr(module, "bias") and module.bias is not None:
                    self.storage.add_to_buffer(
                        "weights", f"epoch_{self.epoch}/{name}.bias", module.bias.data, {"epoch": self.epoch}
                    )

            # For LSTM layers
            if isinstance(module, nn.LSTM):
                # Store weight_ih_l[k], weight_hh_l[k], bias_ih_l[k], bias_hh_l[k]
                # for each layer k
                for layer in range(module.num_layers):
                    for weight_type in ["weight_ih_l", "weight_hh_l", "bias_ih_l", "bias_hh_l"]:
                        param_name = f"{weight_type}{layer}"
                        if hasattr(module, param_name):
                            param = getattr(module, param_name)
                            self.storage.add_to_buffer(
                                "weights", f"epoch_{self.epoch}/{name}.{param_name}", param.data, {"epoch": self.epoch}
                            )

            console.print(".", end="")

    def start_epoch(self, epoch: int):
        """Start monitoring new epoch."""
        self.epoch = epoch

    def step(self):
        """Update step counter and handle buffered data."""
        self.step_count += 1

        # Check if we need to force a buffer flush
        if self.step_count % self.config.flush_interval == 0:
            self.storage.flush_all()

    def end_epoch(self):
        """Finish monitoring current epoch."""
        if self.config.enable_weight_tracking:
            self._track_weights()
        self.storage.flush_all()

    def close(self):
        """Clean up monitor resources."""
        # Remove hooks
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

        # Close storage
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __str__(self) -> str:
        """Return string representation of the ExperimentMonitor."""
        tracking_modes = []
        if self.config.enable_activation_tracking:
            tracking_modes.append("activations")
        if self.config.enable_gradient_tracking:
            tracking_modes.append("gradients")
        if self.config.enable_weight_tracking:
            tracking_modes.append("weights")

        return (
            f"ExperimentMonitor("
            f"log_dir='{self.log_dir}', "
            f"tracking={', '.join(tracking_modes)}, "
            f"current_epoch={self.epoch}, "
            f"step_count={self.step_count}, "
            f"active_hooks={len(self.hooks)})"
        )
