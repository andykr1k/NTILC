"""
Validity-aware loss function for tool call generation.

Implements multi-term loss that enforces:
- Correct tool selection
- Required arguments present
- No illegal arguments
- Correct types
- Value constraints respected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from ablation.tool_schemas import TOOL_SCHEMAS


class ValidityAwareLoss(nn.Module):
    """
    Multi-term loss function that enforces argument validity.

    Loss = λ_tool * L_tool + λ_structure * L_structure + λ_args * L_args + 
           λ_values * L_values + λ_constraints * L_constraints
    """

    def __init__(
        self,
        lambda_tool: float = 1.0,
        lambda_structure: float = 1.0,
        lambda_args: float = 1.0,
        lambda_values: float = 1.0,
        lambda_constraints: float = 0.5,
        lambda_reconstruction: float = 1.0
    ):
        """
        Args:
            lambda_tool: Weight for tool selection loss
            lambda_structure: Weight for argument structure (presence/absence) loss
            lambda_args: Weight for argument name loss
            lambda_values: Weight for value type loss
            lambda_constraints: Weight for constraint loss
            lambda_reconstruction: Weight for standard reconstruction loss
        """
        super().__init__()
        self.lambda_tool = lambda_tool
        self.lambda_structure = lambda_structure
        self.lambda_args = lambda_args
        self.lambda_values = lambda_values
        self.lambda_constraints = lambda_constraints
        self.lambda_reconstruction = lambda_reconstruction

        # Build tool and argument vocabularies
        self._build_vocabularies()

        # Cross-entropy loss for classification tasks
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _build_vocabularies(self):
        """Build vocabularies for tools and arguments."""
        self.tool_names = sorted(TOOL_SCHEMAS.keys())
        self.tool_to_idx = {tool: idx for idx,
                            tool in enumerate(self.tool_names)}
        self.idx_to_tool = {idx: tool for tool,
                            idx in self.tool_to_idx.items()}

        # Build argument vocabularies per tool
        self.tool_args = {}
        self.tool_required_args = {}
        self.tool_arg_types = {}
        self.tool_arg_constraints = {}

        for tool_name, schema in TOOL_SCHEMAS.items():
            args = sorted(schema["parameters"].keys())
            self.tool_args[tool_name] = args
            self.tool_arg_to_idx = {arg: idx for idx, arg in enumerate(args)}

            # Required args
            required = [
                arg for arg, info in schema["parameters"].items()
                if info.get("required", False)
            ]
            self.tool_required_args[tool_name] = set(required)

            # Argument types
            arg_types = {}
            for arg, info in schema["parameters"].items():
                arg_types[arg] = info.get("type", "str")
            self.tool_arg_types[tool_name] = arg_types

            # Constraints
            constraints = {}
            for arg, info in schema["parameters"].items():
                arg_constraints = {}
                if "options" in info:
                    arg_constraints["enum"] = info["options"]
                if "default" in info:
                    arg_constraints["default"] = info["default"]
                constraints[arg] = arg_constraints
            self.tool_arg_constraints[tool_name] = constraints

    def parse_tool_call(self, tool_call_str: str) -> Optional[Dict[str, Any]]:
        """Parse a tool call JSON string into structured format."""
        try:
            data = json.loads(tool_call_str)
            return data
        except (json.JSONDecodeError, TypeError):
            return None

    def extract_tool(self, tool_call: Dict[str, Any]) -> Optional[str]:
        """Extract tool name from parsed tool call."""
        return tool_call.get("tool") if tool_call else None

    def extract_arguments(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """Extract arguments from parsed tool call."""
        return tool_call.get("arguments", {}) if tool_call else {}

    def compute_tool_loss(
        self,
        predicted_tool_logits: torch.Tensor,
        target_tool: str
    ) -> torch.Tensor:
        """
        Compute tool selection loss.

        Args:
            predicted_tool_logits: (batch_size, num_tools) logits
            target_tool: Target tool name string

        Returns:
            loss: Scalar loss tensor
        """
        if target_tool not in self.tool_to_idx:
            return torch.tensor(0.0, device=predicted_tool_logits.device)

        target_idx = self.tool_to_idx[target_tool]
        target_tensor = torch.tensor(
            target_idx,
            device=predicted_tool_logits.device,
            dtype=torch.long
        )

        # If batch_size > 1, expand target
        if predicted_tool_logits.shape[0] > 1:
            target_tensor = target_tensor.unsqueeze(
                0).expand(predicted_tool_logits.shape[0])

        return self.ce_loss(predicted_tool_logits, target_tensor)

    def compute_structure_loss(
        self,
        predicted_presence_logits: torch.Tensor,
        target_tool: str,
        target_args: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute argument structure loss (presence/absence).

        Args:
            predicted_presence_logits: (batch_size, num_args) logits for each arg
            target_tool: Target tool name
            target_args: Target arguments dict

        Returns:
            loss: Scalar loss tensor
        """
        if target_tool not in self.tool_args:
            return torch.tensor(0.0, device=predicted_presence_logits.device)

        valid_args = self.tool_args[target_tool]
        required_args = self.tool_required_args[target_tool]

        # Build target vector: 1 if arg should be present, 0 otherwise
        target_vector = []
        for arg in valid_args:
            if arg in target_args or arg in required_args:
                target_vector.append(1.0)
            else:
                target_vector.append(0.0)

        target_tensor = torch.tensor(
            target_vector,
            device=predicted_presence_logits.device,
            dtype=torch.float32
        )

        # Expand if batch_size > 1
        if predicted_presence_logits.shape[0] > 1:
            target_tensor = target_tensor.unsqueeze(
                0).expand_as(predicted_presence_logits)

        return self.bce_loss(predicted_presence_logits, target_tensor)

    def compute_type_loss(
        self,
        predicted_values: Dict[str, torch.Tensor],
        target_tool: str,
        target_args: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute value type loss.

        Args:
            predicted_values: Dict mapping arg names to predicted value tensors
            target_tool: Target tool name
            target_args: Target arguments dict

        Returns:
            loss: Scalar loss tensor
        """
        if target_tool not in self.tool_arg_types:
            return torch.tensor(0.0, device=next(iter(predicted_values.values())).device)

        arg_types = self.tool_arg_types[target_tool]
        type_losses = []

        for arg_name, target_value in target_args.items():
            if arg_name not in arg_types:
                continue

            expected_type = arg_types[arg_name]

            if arg_name not in predicted_values:
                # Missing argument - penalize
                type_losses.append(torch.tensor(1.0, device=next(
                    iter(predicted_values.values())).device))
                continue

            pred_value = predicted_values[arg_name]

            # Check type match
            if expected_type == "int":
                # Predicted should be integer-like
                if not isinstance(target_value, int):
                    type_losses.append(torch.tensor(
                        1.0, device=pred_value.device))
                else:
                    # L1 loss for integer values
                    target_tensor = torch.tensor(
                        float(target_value),
                        device=pred_value.device,
                        dtype=pred_value.dtype
                    )
                    type_losses.append(F.l1_loss(pred_value, target_tensor))

            elif expected_type == "float":
                if not isinstance(target_value, (int, float)):
                    type_losses.append(torch.tensor(
                        1.0, device=pred_value.device))
                else:
                    target_tensor = torch.tensor(
                        float(target_value),
                        device=pred_value.device,
                        dtype=pred_value.dtype
                    )
                    type_losses.append(F.l1_loss(pred_value, target_tensor))

            elif expected_type == "bool":
                if not isinstance(target_value, bool):
                    type_losses.append(torch.tensor(
                        1.0, device=pred_value.device))
                else:
                    target_tensor = torch.tensor(
                        float(target_value),
                        device=pred_value.device,
                        dtype=pred_value.dtype
                    )
                    type_losses.append(F.binary_cross_entropy_with_logits(
                        pred_value, target_tensor))

            elif expected_type == "enum":
                # Enum type - check if value is in options
                constraints = self.tool_arg_constraints[target_tool].get(
                    arg_name, {})
                options = constraints.get("enum", [])
                if target_value not in options:
                    type_losses.append(torch.tensor(
                        1.0, device=pred_value.device))
                else:
                    # Categorical loss
                    option_idx = options.index(target_value)
                    target_tensor = torch.tensor(
                        option_idx,
                        device=pred_value.device,
                        dtype=torch.long
                    )
                    if pred_value.dim() == 0:
                        pred_value = pred_value.unsqueeze(0)
                    if pred_value.shape[0] != len(options):
                        # Reshape if needed
                        pred_value = pred_value[:len(options)]
                    type_losses.append(self.ce_loss(
                        pred_value.unsqueeze(0), target_tensor.unsqueeze(0)))

            else:  # str or other
                # For strings, use string similarity or reconstruction loss
                # This is handled by reconstruction loss, so minimal penalty here
                type_losses.append(torch.tensor(0.1, device=pred_value.device))

        if not type_losses:
            return torch.tensor(0.0, device=next(iter(predicted_values.values())).device)

        return torch.stack(type_losses).mean()

    def compute_constraint_loss(
        self,
        predicted_values: Dict[str, torch.Tensor],
        target_tool: str,
        target_args: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Compute constraint loss (range, enum, regex, etc.).

        Args:
            predicted_values: Dict mapping arg names to predicted value tensors
            target_tool: Target tool name
            target_args: Target arguments dict

        Returns:
            loss: Scalar loss tensor
        """
        if target_tool not in self.tool_arg_constraints:
            return torch.tensor(0.0, device=next(iter(predicted_values.values())).device)

        constraints = self.tool_arg_constraints[target_tool]
        constraint_losses = []

        for arg_name, target_value in target_args.items():
            if arg_name not in constraints:
                continue

            arg_constraints = constraints[arg_name]

            if arg_name not in predicted_values:
                continue

            pred_value = predicted_values[arg_name]

            # Enum constraint
            if "enum" in arg_constraints:
                options = arg_constraints["enum"]
                if target_value not in options:
                    constraint_losses.append(
                        torch.tensor(1.0, device=pred_value.device))
                else:
                    # Soft constraint: encourage prediction to be close to valid options
                    # This is handled by type loss, so minimal additional penalty
                    constraint_losses.append(
                        torch.tensor(0.0, device=pred_value.device))

            # Range constraints (for numeric types)
            if isinstance(target_value, (int, float)):
                # Check if there are implicit range constraints
                # For now, just ensure value is reasonable
                if isinstance(target_value, int):
                    # Penalize if predicted is way outside reasonable range
                    pred_scalar = pred_value.item() if pred_value.numel() == 1 else pred_value.mean().item()
                    if abs(pred_scalar - target_value) > 1000:
                        constraint_losses.append(
                            torch.tensor(0.5, device=pred_value.device))
                    else:
                        constraint_losses.append(
                            torch.tensor(0.0, device=pred_value.device))

        if not constraint_losses:
            return torch.tensor(0.0, device=next(iter(predicted_values.values())).device)

        return torch.stack(constraint_losses).mean()

    def forward(
        self,
        reconstruction_logits: torch.Tensor,
        target_tool_calls: List[str],
        predicted_tool_logits: Optional[torch.Tensor] = None,
        predicted_presence_logits: Optional[torch.Tensor] = None,
        predicted_values: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total validity-aware loss.

        Args:
            reconstruction_logits: (batch_size, seq_len, vocab_size) from decoder
            target_tool_calls: List of target tool call JSON strings
            predicted_tool_logits: (batch_size, num_tools) optional tool prediction logits
            predicted_presence_logits: (batch_size, num_args) optional argument presence logits
            predicted_values: Optional dict of predicted argument values

        Returns:
            Dictionary with individual loss terms and total loss
        """
        device = reconstruction_logits.device

        # Parse target tool calls
        parsed_targets = [self.parse_tool_call(tc) for tc in target_tool_calls]

        # Note: Reconstruction loss is computed separately in training loop
        # using standard cross-entropy with target_ids. This validity loss
        # focuses on structured validity terms (tool, structure, args, values, constraints).
        # The reconstruction_loss term here is a placeholder and should be 0
        # when validity loss is used alongside standard reconstruction loss.
        reconstruction_loss = torch.tensor(0.0, device=device)

        # Tool loss
        tool_loss = torch.tensor(0.0, device=device)
        if predicted_tool_logits is not None:
            for i, parsed in enumerate(parsed_targets):
                if parsed:
                    tool = self.extract_tool(parsed)
                    if tool:
                        # For batch processing, we'd need to handle this differently
                        # For now, compute per-sample and average
                        sample_tool_loss = self.compute_tool_loss(
                            predicted_tool_logits[i:i+1],
                            tool
                        )
                        tool_loss = tool_loss + sample_tool_loss
            if len(parsed_targets) > 0:
                tool_loss = tool_loss / len(parsed_targets)

        # Structure loss
        structure_loss = torch.tensor(0.0, device=device)
        if predicted_presence_logits is not None:
            for i, parsed in enumerate(parsed_targets):
                if parsed:
                    tool = self.extract_tool(parsed)
                    args = self.extract_arguments(parsed)
                    if tool:
                        sample_structure_loss = self.compute_structure_loss(
                            predicted_presence_logits[i:i+1],
                            tool,
                            args
                        )
                        structure_loss = structure_loss + sample_structure_loss
            if len(parsed_targets) > 0:
                structure_loss = structure_loss / len(parsed_targets)

        # Type loss
        type_loss = torch.tensor(0.0, device=device)
        if predicted_values is not None:
            for i, parsed in enumerate(parsed_targets):
                if parsed:
                    tool = self.extract_tool(parsed)
                    args = self.extract_arguments(parsed)
                    if tool:
                        # Extract predicted values for this sample
                        sample_values = {
                            k: v[i] if v.dim() > 0 else v
                            for k, v in predicted_values.items()
                        }
                        sample_type_loss = self.compute_type_loss(
                            sample_values,
                            tool,
                            args
                        )
                        type_loss = type_loss + sample_type_loss
            if len(parsed_targets) > 0:
                type_loss = type_loss / len(parsed_targets)

        # Constraint loss
        constraint_loss = torch.tensor(0.0, device=device)
        if predicted_values is not None:
            for i, parsed in enumerate(parsed_targets):
                if parsed:
                    tool = self.extract_tool(parsed)
                    args = self.extract_arguments(parsed)
                    if tool:
                        sample_values = {
                            k: v[i] if v.dim() > 0 else v
                            for k, v in predicted_values.items()
                        }
                        sample_constraint_loss = self.compute_constraint_loss(
                            sample_values,
                            tool,
                            args
                        )
                        constraint_loss = constraint_loss + sample_constraint_loss
            if len(parsed_targets) > 0:
                constraint_loss = constraint_loss / len(parsed_targets)

        # Total loss
        # Note: reconstruction_loss is 0 here because it's computed separately
        # in the training loop using standard cross-entropy
        total_loss = (
            self.lambda_reconstruction * reconstruction_loss +
            self.lambda_tool * tool_loss +
            self.lambda_structure * structure_loss +
            self.lambda_values * type_loss +
            self.lambda_constraints * constraint_loss
        )

        return {
            "total_loss": total_loss,
            # Always 0.0 (computed separately)
            "reconstruction_loss": reconstruction_loss,
            "tool_loss": tool_loss,
            "structure_loss": structure_loss,
            "type_loss": type_loss,
            "constraint_loss": constraint_loss
        }
