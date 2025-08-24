from typing import Any
from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import (
    parse_tipo_request,
    tipo_single_request,
    tipo_runner,
    apply_tipo_prompt,
    parse_tipo_result,
    OPERATION_LIST,
)
from kgen.formatter import seperate_tags, apply_format

from . import util

class TIPOOperation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {"default": "", "multiline": True}),
                "nl_prompt": ("STRING", {"default": "", "multiline": True}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
                "operation": (sorted(OPERATION_LIST), {"default": sorted(OPERATION_LIST)[0]}),
                "width": ("INT", {"default": 1024, "max": 16384}),
                "height": ("INT", {"default": 1024, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
                "top_k": ("INT", {"default": 80}),
                "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "nl_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "seed": ("INT", {"default": 1234}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            },
        }
    
    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("full_output", "addon_output")
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(self, tipo_model: str, tags: str, nl_prompt: str, width: int, height: int, seed: int, tag_length: str, nl_length: str, ban_tags: str, operation: str, temperature: float, top_p: float, min_p: float, top_k: int, device: str):
        util.load_tipo_model(tipo_model, device)
        
        aspect_ratio = width / height
        prompt_parse_strength = util.parse_prompt_attention(tags)
        nl_prompt_parse_strength = util.parse_prompt_attention(nl_prompt)
        
        processed_nl = "".join(part for part, strength in nl_prompt_parse_strength)
        strength_map_nl = [item for item in nl_prompt_parse_strength if item[1] != 1.0]
        
        tipo.BAN_TAGS = [t.strip() for t in ban_tags.split(",") if t.strip()]
        
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [t.strip() for t in part.strip().split(",") if t.strip()]
            all_tags.extend(part_tags)
            if strength != 1.0:
                for tag in part_tags:
                    strength_map[tag] = strength
                    
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, final_nl_prompt = tipo_single_request(
            org_tag_map, processed_nl, 
            tag_length_target=tag_length.replace(" ", "_"), 
            nl_length_target=nl_length.replace(" ", "_"), 
            operation=operation,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        
        tag_map, _ = tipo_runner(
            meta, operations, general, final_nl_prompt, 
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        
        addon = {"tags": [], "nl": ""}
        for cate, content in tag_map.items():
            if cate in ["generated", "extended"]:
                if addon["nl"] == "": addon["nl"] = content
                continue
            if cate in org_tag_map and isinstance(content, list):
                for tag in content:
                    if tag not in org_tag_map[cate]:
                        addon["tags"].append(tag)

        addon = util.apply_strength(addon, strength_map, strength_map_nl)
        addon["user_tags"] = tags
        addon["user_nl"] = nl_prompt
        
        final_tag_map = util.apply_strength(tag_map, strength_map, strength_map_nl)
        
        return (final_tag_map, addon)


class TIPOFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "full_output": ("LIST", {}),
                "addon_output": ("LIST", {}),
                "format": ("STRING", {
                    "default": """<|special|>, \n<|characters|>, <|copyrights|>, \n<|artist|>, \n\n<|general|>,\n\n<|extended|>.\n\n<|quality|>, <|meta|>, <|rating|>""",
                    "multiline": True,
                }),
            },
        }
        
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "user_prompt", "unformatted_prompt", "unformatted_user_prompt")
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(self, full_output: dict, addon_output: dict, format: str):
        tags = addon_output.get("user_tags", "")
        nl_prompt = addon_output.get("user_nl", "")
        
        prompt_parse_strength = util.parse_prompt_attention(tags)
        all_tags = [t for part, strength in prompt_parse_strength for t in part.strip().split(",") if t.strip()]
        org_tag_map = seperate_tags(all_tags)
        
        # Reconstruct user prompt for comparison
        formatted_prompt_by_user = apply_format(org_tag_map, format)
        unformatted_prompt_by_user = f"{tags}\n{nl_prompt}".strip()
        
        # Construct TIPO's full prompt
        formatted_prompt_by_tipo = apply_format(full_output, format)
        addon_tags_str = ", ".join(addon_output.get("tags", []))
        addon_nl_str = addon_output.get("nl", "")
        unformatted_prompt_by_tipo = f"{tags}, {addon_tags_str}\n{addon_nl_str}".strip(", \n")
        
        return (formatted_prompt_by_tipo, formatted_prompt_by_user, unformatted_prompt_by_tipo, unformatted_prompt_by_user)

# Add mappings for these nodes
NODE_CLASS_MAPPINGS = {
    "TIPOOperation": TIPOOperation,
    "TIPOFormat": TIPOFormat,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPOOperation": "TIPO Single Operation",
    "TIPOFormat": "TIPO Format",
}
