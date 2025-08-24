# Import mappings from each node module
from .nodes.tipo_wildcard import NODE_CLASS_MAPPINGS as wildcard_mappings, NODE_DISPLAY_NAME_MAPPINGS as wildcard_display_mappings
from .nodes.tipo_natural_language import NODE_CLASS_MAPPINGS as nl_mappings, NODE_DISPLAY_NAME_MAPPINGS as nl_display_mappings
from .nodes.tipo_simple import NODE_CLASS_MAPPINGS as simple_mappings, NODE_DISPLAY_NAME_MAPPINGS as simple_display_mappings

# Combine all mappings into one
NODE_CLASS_MAPPINGS = {
    **wildcard_mappings,
    **nl_mappings,
    **simple_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **wildcard_display_mappings,
    **nl_display_mappings,
    **simple_display_mappings,
}

# Expose the combined mappings
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("------------------------------------------")
print("TIPO Custom Nodes (Refactored) loaded.")
print("------------------------------------------")
