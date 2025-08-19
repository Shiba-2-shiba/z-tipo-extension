import os
import re
from pathlib import Path
from typing import Any

import torch
import folder_paths

from ..tipo_installer import install_tipo_kgen, install_llama_cpp

install_llama_cpp()
install_tipo_kgen()

import kgen.models as models
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
from kgen.logging import logger


models.model_dir = Path(folder_paths.models_dir) / "kgen"
os.makedirs(models.model_dir, exist_ok=True)
logger.info(f"Using model dir: {models.model_dir}")

model_list = tipo.models.tipo_model_list
MODEL_NAME_LIST = [
    f"{model_name} | {file}".strip("_")
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


attn_syntax = (
    r"\\\(|"
    r"\\\)|"
    r"\\\[|"
    r"\\]|"
    r"\\\\|"
    r"\\|"
    r"\(|"
    r"\[|"
    r":\s*([+-]?[.\d]+)\s*\)|"
    r"\)|"
    r"]|"
    r"[^\\()\[\]:]+|"
    r":"
)
re_attention = re.compile(
    attn_syntax,
    re.X,
)

re_break = re.compile(r"\s*\bBREAK\b\s*", re.S)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    """
    res = []
    round_brackets = []
    square_brackets = []
    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)
        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and round_brackets:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and round_brackets:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and square_brackets:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            parts = re.split(re_break, text)
            for i, part in enumerate(parts):
                if i > 0:
                    res.append(["BREAK", -1])
                res.append([part, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)
    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1
    return res


def apply_strength(tag_map, strength_map, strength_map_nl):
    for cate in tag_map.keys():
        new_list = []
        if isinstance(tag_map[cate], str):
            if all(part in tag_map[cate] for part, strength in strength_map_nl):
                org_prompt = tag_map[cate]
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, org_prompt = org_prompt.split(part, 1)
                    new_prompt += before.replace("(", "\\(").replace(")", "\\)")
                    part = part.replace("(", "\\(").replace(")", "\\)")
                    new_prompt += f"({part}:{strength})"
                new_prompt += org_prompt
            tag_map[cate] = new_prompt
            continue
        for org_tag in tag_map[cate]:
            tag = org_tag.replace("(", "\\(").replace(")", "\\)")
            if org_tag in strength_map:
                new_list.append(f"({tag}:{strength_map[org_tag]})")
            else:
                new_list.append(tag)
        tag_map[cate] = new_list
    return tag_map


current_model = None
FUNCTION = "execute"
CATEGORY = "utils/promptgen"


class TIPO:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tags": ("STRING", {"default": "", "multiline": True}),
                "nl_prompt": ("STRING", {"default": "", "multiline": True}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "tipo_model": (MODEL_NAME_LIST, {"default": MODEL_NAME_LIST[0]}),
                "format": (
                    "STRING",
                    {
                        "default": """<|special|>, 
<|characters|>, <|copyrights|>, 
<|artist|>, 

<|wildcard_appearance|>, <|wildcard_clothing|>,
<|wildcard_pose_emotion|>, <|wildcard_background|>,
<|wildcard_camera_lighting|>, <|wildcard_art_style|>,

<|general|>,

<|quality|>, <|meta|>, <|rating|>""",
                        "multiline": True,
                    },
                ),
                "width": ("INT", {"default": 1024, "max": 16384}),
                "height": ("INT", {"default": 1024, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
                "top_k": ("INT", {"default": 80}),
                "tag_length": (
                    ["very_short", "short", "long", "very_long"],
                    {"default": "long"},
                ),
                "nl_length": (
                    ["very_short", "short", "long", "very_long"],
                    {"default": "long"},
                ),
                "seed": ("INT", {"default": 1234, "min": -1, "max": 0xffffffffffffffff}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            },
            "optional": {
                "appearance_tags": ("STRING", {"default": "", "multiline": True}),
                "clothing_tags": ("STRING", {"default": "", "multiline": True}),
                "background_tags": ("STRING", {"default": "", "multiline": True}),
                "pose_emotion_tags": ("STRING", {"default": "", "multiline": True}),
                "camera_lighting_tags": ("STRING", {"default": "", "multiline": True}),
                "art_style_tags": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "user_prompt", "unformatted_prompt", "unformatted_user_prompt")
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY

    def execute(
        self,
        tipo_model: str,
        tags: str,
        nl_prompt: str,
        width: int,
        height: int,
        seed: int,
        tag_length: str,
        nl_length: str,
        ban_tags: str,
        format: str,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        device: str,
        appearance_tags: str = "",
        clothing_tags: str = "",
        background_tags: str = "",
        pose_emotion_tags: str = "",
        camera_lighting_tags: str = "",
        art_style_tags: str = "",
    ):
        global current_model
        
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            models.load_model(target, gguf, device=device)
            current_model = (tipo_model, device)
            
        aspect_ratio = width / height
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        tipo.BAN_TAGS = black_list
        
        final_prompt_parts = {}
        all_original_tags_list = []
        all_addon_tags_list = []

        # === Part 1: Process main 'tags' and 'nl_prompt' ===
        if tags.strip() or nl_prompt.strip():
            prompt_parse_strength = parse_prompt_attention(tags)
            nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
            nl_prompt_processed = "".join(part for part, strength in nl_prompt_parse_strength)
            strength_map_nl = [item for item in nl_prompt_parse_strength if item[1] != 1.0]

            main_all_tags = []
            strength_map = {}
            for part, strength in prompt_parse_strength:
                part_tags = [t.strip() for t in part.strip().split(",") if t.strip()]
                main_all_tags.extend(part_tags)
                if strength != 1.0:
                    for tag in part_tags:
                        strength_map[tag] = strength
            all_original_tags_list.extend(main_all_tags)

            org_tag_map = seperate_tags(main_all_tags)
            meta, operations, general, _ = parse_tipo_request(
                org_tag_map, nl_prompt_processed,
                tag_length_target=tag_length.replace(" ", "_"),
                nl_length_target=nl_length.replace(" ", "_"),
                generate_extra_nl_prompt=("<|extended|>" in format or "<|generated|>" in format)
            )
            meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            tag_map_main, _ = tipo_runner(
                meta, operations, general, nl_prompt_processed,
                temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
            )
 
            if 'general' in tag_map_main and isinstance(tag_map_main['general'], list):
                tag_map_main['general'] = [tag for tag in tag_map_main['general'] if tag.strip() != "1.0"]

            for key, tag_list in tag_map_main.items():
                if isinstance(tag_list, list):
                    tag_map_main[key] = list(dict.fromkeys(tag_list))

            final_prompt_parts = apply_strength(tag_map_main, strength_map, strength_map_nl)

            main_original_tags_set = set(main_all_tags)
            for cate, tag_list in tag_map_main.items():
                if isinstance(tag_list, list):
                    for tag in tag_list:
                        if tag not in main_original_tags_set:
                            all_addon_tags_list.append(tag)

        # === Part 2: Process each wildcard category individually ===
        wildcard_categories = {
            "<|wildcard_appearance|>": appearance_tags.strip(),
            "<|wildcard_clothing|>": clothing_tags.strip(),
            "<|wildcard_background|>": background_tags.strip(),
            "<|wildcard_pose_emotion|>": pose_emotion_tags.strip(),
            "<|wildcard_camera_lighting|>": camera_lighting_tags.strip(),
            "<|wildcard_art_style|>": art_style_tags.strip(),
        }

        current_seed = seed + 1
        for placeholder, category_tags in wildcard_categories.items():
            placeholder_key = placeholder.strip("<|>")
            if not category_tags:
                final_prompt_parts[placeholder_key] = ""
                continue

            logger.info(f"TIPO is extending category: {placeholder}")
            cat_all_tags = [t.strip() for t in category_tags.split(',') if t.strip()]
            all_original_tags_list.extend(cat_all_tags)

            cat_org_tag_map = seperate_tags(cat_all_tags)
            
            cat_meta, cat_operations, cat_general, _ = parse_tipo_request(
                cat_org_tag_map, "", tag_length_target=tag_length.replace(" ", "_"),
                nl_length_target="very_short", generate_extra_nl_prompt=False,
            )
            cat_meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            cat_tag_map, _ = tipo_runner(
                cat_meta, cat_operations, cat_general, "",
                temperature=temperature, seed=current_seed, top_p=top_p, min_p=min_p, top_k=top_k,
            )

            original_tags_set = set(cat_all_tags)
            addon_tags = []
            for tag_list in cat_tag_map.values():
                 if isinstance(tag_list, list):
                    for tag in tag_list:
                        if tag not in original_tags_set:
                            addon_tags.append(tag)
            
            combined_tags_list = cat_all_tags + addon_tags
            unique_tags_list = list(dict.fromkeys(combined_tags_list))
            all_addon_tags_list.extend(addon_tags)

            final_prompt_parts[placeholder_key] = ", ".join(unique_tags_list)
            current_seed += 1

        # === Part 3: Final Assembly ===
        final_prompt = apply_format(final_prompt_parts, format)
        
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        unique_final_tags_list = list(dict.fromkeys(final_tags_list))
        final_prompt = ", ".join(unique_final_tags_list)

        # === Part 4: Reconstruct other return values ===
        all_original_tags_str = ", ".join(list(dict.fromkeys(all_original_tags_list)))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(all_addon_tags_list)))
        unformatted_prompt_by_tipo = (all_original_tags_str + ", " + unformatted_addon_tags).strip(", ")
        
        user_prompt_parts = seperate_tags(all_original_tags_list)
        formatted_prompt_by_user = apply_format(user_prompt_parts, format)
        
        for placeholder, original_tags in wildcard_categories.items():
            formatted_prompt_by_user = formatted_prompt_by_user.replace(placeholder, original_tags)
        
        user_tags_list = [tag.strip() for tag in formatted_prompt_by_user.split(',') if tag.strip()]
        formatted_prompt_by_user = ", ".join(list(dict.fromkeys(user_tags_list)))

        unformatted_prompt_by_user = all_original_tags_str + "\n" + nl_prompt
        
        return (final_prompt, formatted_prompt_by_user, unformatted_prompt_by_tipo, unformatted_prompt_by_user)

# --- Start of New Node Definition ---

class TIPONaturalLanguage:
    """
    A node dedicated to generating natural language (NL) descriptions from tags.
    It first creates an overall description, then uses it as context to generate
    more detailed descriptions for individual categories, ensuring consistency.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_model": (MODEL_NAME_LIST, {"default": MODEL_NAME_LIST[0]}),
                "tags": ("STRING", {"default": "", "multiline": True, "placeholder": "Overall tags (character, copyright, general...)"}),
                "ban_tags": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 1024, "max": 16384}),
                "height": ("INT", {"default": 1024, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.6, "step": 0.01, "min": 0.1, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
                "top_k": ("INT", {"default": 80}),
                "overall_nl_length": (
                    ["very_short", "short", "long", "very_long"],
                    {"default": "long"},
                ),
                "category_nl_length": (
                    ["very_short", "short", "long", "very_long"],
                    {"default": "short"},
                ),
                "seed": ("INT", {"default": 1234, "min": -1, "max": 0xffffffffffffffff}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            },
            "optional": {
                "appearance_tags": ("STRING", {"default": "", "multiline": True}),
                "clothing_tags": ("STRING", {"default": "", "multiline": True}),
                "background_tags": ("STRING", {"default": "", "multiline": True}),
                "pose_emotion_tags": ("STRING", {"default": "", "multiline": True}),
                "camera_lighting_tags": ("STRING", {"default": "", "multiline": True}),
                "art_style_tags": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "overall_nl",
        "appearance_nl",
        "clothing_nl",
        "background_nl",
        "pose_emotion_nl",
        "camera_lighting_nl",
        "art_style_nl",
    )
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(
        self,
        tipo_model: str,
        tags: str,
        ban_tags: str,
        width: int,
        height: int,
        temperature: float,
        top_p: float,
        min_p: float,
        top_k: int,
        overall_nl_length: str,
        category_nl_length: str,
        seed: int,
        device: str,
        appearance_tags: str = "",
        clothing_tags: str = "",
        background_tags: str = "",
        pose_emotion_tags: str = "",
        camera_lighting_tags: str = "",
        art_style_tags: str = "",
    ):
        global current_model

        # --- Model Loading ---
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            models.load_model(target, gguf, device=device)
            current_model = (tipo_model, device)
        
        # --- Common Parameters ---
        aspect_ratio = width / height
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        tipo.BAN_TAGS = black_list
        if seed == -1:
            seed = int.from_bytes(os.urandom(8), 'big')

        # --- Gather All Tags for Overall NL ---
        all_tags_list = [t.strip() for t in tags.split(',') if t.strip()]
        
        wildcard_categories = {
            "appearance": appearance_tags, "clothing": clothing_tags,
            "background": background_tags, "pose_emotion": pose_emotion_tags,
            "camera_lighting": camera_lighting_tags, "art_style": art_style_tags,
        }
        for category_tags_str in wildcard_categories.values():
            all_tags_list.extend([t.strip() for t in category_tags_str.split(',') if t.strip()])
        
        all_tags_list = list(dict.fromkeys(all_tags_list)) # Deduplicate

        # === 1. Generate Overall NL ===
        overall_nl = ""
        logger.info("Generating overall NL description...")
        if all_tags_list:
            org_tag_map = seperate_tags(all_tags_list)
            meta, operations, general, _ = tipo_single_request(
                org_tag_map, nl_prompt="",
                tag_length_target="short",
                nl_length_target=overall_nl_length.replace(" ", "_"),
                operation="tag_to_long"
            )
            meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            tag_map_overall, _ = tipo_runner(
                meta, operations, general, "",
                temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
            )
            overall_nl = tag_map_overall.get("generated", tag_map_overall.get("extended", ""))
            logger.info(f"Overall NL: {overall_nl[:100]}...")
        else:
            logger.info("No input tags for overall NL generation.")

        # === 2. Generate Category-specific NL using Overall NL as Context ===
        category_nls = {}
        current_seed = seed + 1
        for name, category_tags_str in wildcard_categories.items():
            output_key = name + "_nl"
            if not category_tags_str.strip():
                category_nls[output_key] = ""
                continue
            
            logger.info(f"Generating contextual NL for category: {name}")
            cat_tags = [t.strip() for t in category_tags_str.split(',') if t.strip()]
            cat_org_tag_map = seperate_tags(cat_tags)

            # MODIFIED: Create a more specific prompt to focus the model's output
            # This instructs the model to use the overall_nl as context but to only
            # describe the specific category, preventing content from other categories.
            category_name_for_prompt = name.replace('_', ' ')
            contextual_prompt = f"{overall_nl} In this scene, describe the {category_name_for_prompt}."

            # Use "short_to_tag_to_long" to refine/expand the overall NL with category-specific tags
            cat_meta, cat_ops, cat_general, cat_nl_prompt = tipo_single_request(
                cat_org_tag_map,
                nl_prompt=contextual_prompt, # Provide the new focused prompt
                nl_length_target=category_nl_length.replace(" ", "_"),
                operation="short_to_tag_to_long"
            )
            cat_meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            cat_tag_map, _ = tipo_runner(
                cat_meta, cat_ops, cat_general, cat_nl_prompt,
                temperature=temperature, seed=current_seed, top_p=top_p, min_p=min_p, top_k=top_k,
            )
            category_nls[output_key] = cat_tag_map.get("generated", cat_tag_map.get("extended", ""))
            logger.info(f"  -> {name} NL: {category_nls[output_key][:100]}...")
            current_seed += 1

        return (
            overall_nl,
            category_nls.get("appearance_nl", ""),
            category_nls.get("clothing_nl", ""),
            category_nls.get("background_nl", ""),
            category_nls.get("pose_emotion_nl", ""),
            category_nls.get("camera_lighting_nl", ""),
            category_nls.get("art_style_nl", ""),
        )

# --- End of New Node Definition ---


class TIPOOperation:
    # This class remains unchanged.
    INPUT_TYPES = lambda: { "required": { "tags": ("STRING", {"defaultInput": True, "multiline": True}), "nl_prompt": ("STRING", {"defaultInput": True, "multiline": True}), "ban_tags": ("STRING", {"defaultInput": True, "multiline": True}), "tipo_model": (MODEL_NAME_LIST, {"default": MODEL_NAME_LIST[0]}), "operation": (sorted(OPERATION_LIST), {"default": sorted(OPERATION_LIST)[0]}), "width": ("INT", {"default": 1024, "max": 16384}), "height": ("INT", {"default": 1024, "max": 16384}), "temperature": ("FLOAT", {"default": 0.5, "step": 0.01}), "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}), "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}), "top_k": ("INT", {"default": 80}), "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}), "nl_length": (["very_short", "short", "long", "very_long"], {"default": "long"}), "seed": ("INT", {"default": 1234}), "device": (["cpu", "cuda"], {"default": "cuda"}), }, }
    RETURN_TYPES = ("LIST", "LIST")
    RETURN_NAMES = ("full_output", "addon_output")
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    def execute(self, tipo_model: str, tags: str, nl_prompt: str, width: int, height: int, seed: int, tag_length: str, nl_length: str, ban_tags: str, operation: str, temperature: float, top_p: float, min_p: float, top_k: int, device: str):
        global current_model
        if (tipo_model, device) != current_model:
            if " | " in tipo_model:
                model_name, gguf_name = tipo_model.split(" | ")
                target_file = f"{model_name.split('/')[-1]}_{gguf_name}"
                if str(models.model_dir / target_file) not in models.list_gguf():
                    models.download_gguf(model_name, gguf_name)
                target = os.path.join(str(models.model_dir), target_file)
                gguf = True
            else:
                target = tipo_model
                gguf = False
            models.load_model(target, gguf, device=device)
            current_model = (tipo_model, device)
        aspect_ratio = width / height
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)
        nl_prompt_wihtout_extranet = nl_prompt
        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))
        black_list = [tag.strip() for tag in ban_tags.split(",") if tag.strip()]
        tipo.BAN_TAGS = black_list
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength
        tag_length = tag_length.replace(" ", "_")
        org_tag_map = seperate_tags(all_tags)
        meta, operations, general, nl_prompt = tipo_single_request(
            org_tag_map, nl_prompt, tag_length_target=tag_length, nl_length_target=nl_length, operation=operation,
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        tag_map, _ = tipo_runner(
            meta, operations, general, nl_prompt, temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        addon = {"tags": [], "nl": ""}
        for cate in tag_map.keys():
            if cate == "generated" and addon["nl"] == "":
                addon["nl"] = tag_map[cate]
                continue
            if cate == "extended":
                extended = tag_map[cate]
                addon["nl"] = extended
                continue
            if cate not in org_tag_map:
                continue
            for tag in tag_map[cate]:
                if tag in org_tag_map[cate]:
                    continue
                addon["tags"].append(tag)
        addon = apply_strength(addon, strength_map, strength_map_nl)
        addon["user_tags"] = prompt_without_extranet
        addon["user_nl"] = nl_prompt_wihtout_extranet
        tag_map = apply_strength(tag_map, strength_map, strength_map_nl)
        return (tag_map, addon)


class TIPOFormat:
    # This class remains unchanged.
    INPUT_TYPES = lambda: { "required": { "full_output": ("LIST", {"default": []}), "addon_output": ("LIST", {"default": []}), "format": ("STRING", { "default": """<|special|>, \n<|characters|>, <|copyrights|>, \n<|artist|>, \n\n<|general|>,\n\n<|extended|>.\n\n<|quality|>, <|meta|>, <|rating|>""", "multiline": True, }), }, }
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "user_prompt", "unformatted_prompt", "unformatted_user_prompt")
    FUNCTION = FUNCTION
    CATEGORY = CATEGORY
    def execute(self, full_output: list, addon_output: dict[str, Any], format: str):
        tags = addon_output.pop("user_tags", "")
        nl_prompt = addon_output.pop("user_nl", "")
        addon = addon_output
        tag_map = full_output
        prompt_without_extranet = tags
        prompt_parse_strength = parse_prompt_attention(prompt_without_extranet)
        nl_prompt_parse_strength = parse_prompt_attention(nl_prompt)
        nl_prompt = ""
        strength_map_nl = []
        for part, strength in nl_prompt_parse_strength:
            nl_prompt += part
            if strength == 1:
                continue
            strength_map_nl.append((part, strength))
        all_tags = []
        strength_map = {}
        for part, strength in prompt_parse_strength:
            part_tags = [tag.strip() for tag in part.strip().split(",") if tag.strip()]
            all_tags.extend(part_tags)
            if strength == 1:
                continue
            for tag in part_tags:
                strength_map[tag] = strength
        org_tag_map = seperate_tags(all_tags)
        meta, _, general, nl_prompt = parse_tipo_request(org_tag_map, nl_prompt)
        org_formatted_prompt = parse_tipo_result(
            apply_tipo_prompt(meta, general, nl_prompt, "short_to_tag_to_long", "long", True, gen_meta=True)
        )
        org_formatted_prompt = apply_strength(org_formatted_prompt, strength_map, strength_map_nl)
        formatted_prompt_by_user = apply_format(org_formatted_prompt, format)
        unformatted_prompt_by_user = tags + nl_prompt
        formatted_prompt_by_tipo = apply_format(tag_map, format)
        unformatted_prompt_by_tipo = (tags + ", " + ", ".join(addon["tags"]) + "\n" + addon["nl"])
        return (formatted_prompt_by_tipo, formatted_prompt_by_user, unformatted_prompt_by_tipo, unformatted_prompt_by_user)


NODE_CLASS_MAPPINGS = {
    "TIPO": TIPO,
    "TIPONaturalLanguage": TIPONaturalLanguage, # Added new node
    "TIPOOperation": TIPOOperation,
    "TIPOFormat": TIPOFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPO": "TIPO (Wildcard Enabled)",
    "TIPONaturalLanguage": "TIPO Natural Language (Contextual)", # Added new node display name
    "TIPOOperation": "TIPO Single Operation",
    "TIPOFormat": "TIPO Format",
}
