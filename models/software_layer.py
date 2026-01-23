"""
Software Layer for NTILC: Maps cluster IDs to tools.

This decouples the model from execution. All tool execution logic,
versioning, safety rules, and permissions are handled here.
The model only retrieves the correct cluster ID.
"""

import json
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict

from ablation.tool_schemas import TOOL_SCHEMAS


class ClusterToToolMapper:
    """
    Maps cluster IDs to one or more tools.
    
    Maintains:
    - Cluster → Tool mappings
    - Versioning
    - Safety rules
    - Permissions
    - Tool indices and arguments
    """
    
    def __init__(
        self,
        cluster_to_tools: Optional[Dict[int, List[str]]] = None,
        tool_versions: Optional[Dict[str, str]] = None,
        safety_rules: Optional[Dict[str, List[str]]] = None
    ):
        """
        Args:
            cluster_to_tools: Mapping from cluster ID to list of tool names
            tool_versions: Mapping from tool name to version string
            safety_rules: Mapping from tool name to list of safety constraints
        """
        self.cluster_to_tools = cluster_to_tools or {}
        self.tool_versions = tool_versions or {}
        self.safety_rules = safety_rules or {}
        
        # Build reverse mapping: tool -> clusters
        self.tool_to_clusters = defaultdict(list)
        for cluster_id, tools in self.cluster_to_tools.items():
            for tool in tools:
                self.tool_to_clusters[tool].append(cluster_id)
    
    def add_cluster_mapping(self, cluster_id: int, tools: List[str]):
        """Add or update cluster to tool mapping."""
        self.cluster_to_tools[cluster_id] = tools
        
        # Update reverse mapping
        for tool in tools:
            if cluster_id not in self.tool_to_clusters[tool]:
                self.tool_to_clusters[tool].append(cluster_id)
    
    def get_tools_for_cluster(self, cluster_id: int) -> List[str]:
        """Get tools associated with a cluster ID."""
        return self.cluster_to_tools.get(cluster_id, [])
    
    def get_clusters_for_tool(self, tool_name: str) -> List[int]:
        """Get clusters associated with a tool."""
        return self.tool_to_clusters.get(tool_name, [])
    
    def resolve_cluster(
        self,
        cluster_id: int,
        similarity_score: float = 1.0,
        threshold: float = 0.5
    ) -> Optional[Dict[str, Any]]:
        """
        Resolve cluster ID to tool with safety checks.
        
        Args:
            cluster_id: Cluster ID from retrieval
            similarity_score: Confidence score
            threshold: Minimum similarity threshold
        
        Returns:
            Dictionary with tool information or None if below threshold
        """
        if similarity_score < threshold:
            return None
        
        if cluster_id not in self.cluster_to_tools:
            return None
        
        tools = self.cluster_to_tools[cluster_id]
        
        # For now, return first tool (can be extended for multi-tool plans)
        if len(tools) == 0:
            return None
        
        tool_name = tools[0]
        
        # Get tool schema
        tool_schema = TOOL_SCHEMAS.get(tool_name)
        if not tool_schema:
            return None
        
        return {
            "tool": tool_name,
            "version": self.tool_versions.get(tool_name, "1.0.0"),
            "schema": tool_schema,
            "safety_rules": self.safety_rules.get(tool_name, []),
            "confidence": similarity_score,
            "cluster_id": cluster_id
        }
    
    def resolve_multiple_clusters(
        self,
        cluster_ids: List[int],
        similarities: List[float],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Resolve multiple cluster IDs (for multi-tool scenarios).
        
        Args:
            cluster_ids: List of cluster IDs
            similarities: List of similarity scores
            threshold: Minimum similarity threshold
        
        Returns:
            List of resolved tool dictionaries
        """
        results = []
        for cluster_id, similarity in zip(cluster_ids, similarities):
            resolved = self.resolve_cluster(cluster_id, similarity, threshold)
            if resolved:
                results.append(resolved)
        
        return results
    
    def validate_tool_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        user_permissions: Optional[Set[str]] = None
    ) -> tuple[bool, List[str]]:
        """
        Validate tool execution with safety rules and permissions.
        
        Args:
            tool_name: Name of tool to execute
            arguments: Tool arguments
            user_permissions: Set of user permissions
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check tool exists
        if tool_name not in TOOL_SCHEMAS:
            errors.append(f"Unknown tool: {tool_name}")
            return False, errors
        
        # Check permissions
        if user_permissions is not None:
            # For now, all tools are allowed (can add permission checks)
            pass
        
        # Check safety rules
        safety_rules = self.safety_rules.get(tool_name, [])
        for rule in safety_rules:
            # Apply safety rule (can be extended)
            pass
        
        # Validate arguments against schema
        schema = TOOL_SCHEMAS[tool_name]
        required_args = [
            name for name, info in schema["parameters"].items()
            if info.get("required", False)
        ]
        
        for arg in required_args:
            if arg not in arguments:
                errors.append(f"Missing required argument: {arg}")
        
        # Check argument types
        for arg_name, arg_value in arguments.items():
            if arg_name not in schema["parameters"]:
                errors.append(f"Unknown argument: {arg_name}")
                continue
            
            param_info = schema["parameters"][arg_name]
            expected_type = param_info.get("type", "str")
            
            # Type checking
            if expected_type == "int" and not isinstance(arg_value, int):
                errors.append(f"Argument {arg_name} should be int")
            elif expected_type == "float" and not isinstance(arg_value, (int, float)):
                errors.append(f"Argument {arg_name} should be float")
            elif expected_type == "bool" and not isinstance(arg_value, bool):
                errors.append(f"Argument {arg_name} should be bool")
            elif expected_type == "enum":
                options = param_info.get("options", [])
                if options and arg_value not in options:
                    errors.append(f"Argument {arg_name} must be one of {options}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def initialize_from_tools(self, tools: List[str] = None):
        """
        Initialize cluster mappings from tool list.
        
        For now, creates one cluster per tool (1:1 mapping).
        Can be extended to learn more sophisticated mappings.
        """
        if tools is None:
            tools = list(TOOL_SCHEMAS.keys())
        
        # Simple 1:1 mapping: cluster_id = tool_index
        for idx, tool_name in enumerate(tools):
            self.add_cluster_mapping(idx, [tool_name])
    
    def save(self, filepath: str):
        """Save mappings to file."""
        data = {
            "cluster_to_tools": self.cluster_to_tools,
            "tool_versions": self.tool_versions,
            "safety_rules": self.safety_rules
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Load mappings from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.cluster_to_tools = {int(k): v for k, v in data.get("cluster_to_tools", {}).items()}
        self.tool_versions = data.get("tool_versions", {})
        self.safety_rules = data.get("safety_rules", {})
        
        # Rebuild reverse mapping
        self.tool_to_clusters = defaultdict(list)
        for cluster_id, tools in self.cluster_to_tools.items():
            for tool in tools:
                self.tool_to_clusters[tool].append(cluster_id)
