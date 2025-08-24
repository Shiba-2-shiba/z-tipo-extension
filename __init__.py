
from .nodes.tipo_preprocessor import NODE_CLASS_MAPPINGS as preprocessor_mappings, NODE_DISPLAY_NAME_MAPPINGS as preprocessor_display_mappings
from .nodes.tipo_wildcard import NODE_CLASS_MAPPINGS as wildcard_mappings, NODE_DISPLAY_NAME_MAPPINGS as wildcard_display_mappings
from .nodes.tipo_natural_language import NODE_CLASS_MAPPINGS as nl_mappings, NODE_DISPLAY_NAME_MAPPINGS as nl_display_mappings
from .nodes.tipo_simple import NODE_CLASS_MAPPINGS as simple_mappings, NODE_DISPLAY_NAME_MAPPINGS as simple_display_mappings

NODE_CLASS_MAPPINGS = {
    **preprocessor_mappings,
    **wildcard_mappings,
    **nl_mappings,
    **simple_mappings,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **preprocessor_display_mappings,
    **wildcard_display_mappings,
    **nl_display_mappings,
    **simple_display_mappings,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("------------------------------------------")
print("TIPO Custom Nodes (Refactored) loaded.")
print("------------------------------------------")
