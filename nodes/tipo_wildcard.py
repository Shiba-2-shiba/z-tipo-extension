# -*- coding: utf-8 -*-
import re
from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import parse_tipo_request, tipo_runner
from kgen.formatter import seperate_tags, apply_format
from kgen.logging import logger

# 共通機能を util からインポート
from . import util

class TIPO:
    """
    Preprocessorからの入力を使用して、タグの拡張とフォーマットを行う改修版ノード。
    UIが簡素化され、プロンプト生成のコア機能に特化しています。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # Preprocessorからの単一入力を受け取る
                "tipo_prompts": ("TIPO_PROMPTS",),
                
                # プロンプト生成に関する設定は維持
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
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
                "tag_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "nl_length": (["very_short", "short", "long", "very_long"], {"default": "long"}),
                "seed": ("INT", {"default": 1234, "min": -1, "max": 0xffffffffffffffff}),
                "device": (["cpu", "cuda"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = (
        "STRING", "STRING", "STRING", "STRING",
        "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",
    )
    RETURN_NAMES = (
        "prompt", "user_prompt", "unformatted_prompt", "unformatted_user_prompt",
        "appearance_tags", "clothing_tags", "background_tags", 
        "pose_emotion_tags", "camera_lighting_tags", "art_style_tags",
    )
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(
        self,
        tipo_prompts: dict,
        tipo_model: str,
        format: str,
        width: int, height: int,
        temperature: float, top_p: float, min_p: float, top_k: int,
        tag_length: str, nl_length: str,
        seed: int, device: str,
    ):
        # 不要なタグ（例: "1.0", "<|short|>"）を除去するための正規表現パターン
        invalid_tag_pattern = re.compile(r'^\s*(\d+\.\d+|<\|.*?\|>)\s*$')

        util.load_tipo_model(tipo_model, device)
        
        # --- Preprocessorからのデータを展開 ---
        tags = tipo_prompts.get("main", {}).get("tags", "")
        nl_prompt = tipo_prompts.get("main", {}).get("nl_prompt", "")
        ban_tags = tipo_prompts.get("main", {}).get("ban_tags", "")
        category_prompts = tipo_prompts.get("categories", {})
        
        aspect_ratio = width / height
        
        # --- 修正点 1: ban_tagsをリストとして変数に保持 ---
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        tipo.BAN_TAGS = black_list
        
        final_prompt_parts = {}
        all_original_tags_list = []
        all_addon_tags_list = []

        # Part 1: メインの 'tags' と 'nl_prompt' を処理
        if tags.strip() or nl_prompt.strip():
            prompt_parse_strength = util.parse_prompt_attention(tags)
            nl_prompt_parse_strength = util.parse_prompt_attention(nl_prompt)
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
                tag_map_main['general'] = [
                    tag for tag in tag_map_main['general'] 
                    if not invalid_tag_pattern.match(tag)
                ]

            for key, tag_list in tag_map_main.items():
                if isinstance(tag_list, list):
                    tag_map_main[key] = list(dict.fromkeys(tag_list))

            final_prompt_parts = util.apply_strength(tag_map_main, strength_map, strength_map_nl)

            main_original_tags_set = set(main_all_tags)
            for cate, tag_list in tag_map_main.items():
                if isinstance(tag_list, list):
                    for tag in tag_list:
                        if tag not in main_original_tags_set:
                            all_addon_tags_list.append(tag)

        # Part 2: 各カテゴリを個別に処理
        category_outputs = {}
        current_seed = seed + 1
        for category_name, category_tags in category_prompts.items():
            placeholder_key = f"wildcard_{category_name}"
            
            if not category_tags.strip():
                final_prompt_parts[placeholder_key] = ""
                category_outputs[placeholder_key] = ""
                continue

            # --- 修正点 2: ループ内で毎回ban_tagsを再設定 ---
            tipo.BAN_TAGS = black_list

            logger.info(f"TIPO is extending category: <|{placeholder_key}|>")
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
                        if tag not in original_tags_set and not invalid_tag_pattern.match(tag):
                            addon_tags.append(tag)
            
            combined_tags_list = cat_all_tags + addon_tags
            unique_tags_list = list(dict.fromkeys(combined_tags_list))
            all_addon_tags_list.extend(addon_tags)

            processed_category_tags = ", ".join(unique_tags_list)
            final_prompt_parts[placeholder_key] = processed_category_tags
            category_outputs[placeholder_key] = processed_category_tags
            current_seed += 1

        # Part 3: 最終的な組み立てと他の戻り値の再構築
        final_prompt = apply_format(final_prompt_parts, format)
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        final_prompt = ", ".join(list(dict.fromkeys(final_tags_list)))

        all_original_tags_str = ", ".join(list(dict.fromkeys(all_original_tags_list)))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(all_addon_tags_list)))
        unformatted_prompt_by_tipo = (all_original_tags_str + ", " + unformatted_addon_tags).strip(", ")
        
        user_prompt_parts = seperate_tags(all_original_tags_list)
        formatted_prompt_by_user = apply_format(user_prompt_parts, format)
        
        for category_name, original_tags in category_prompts.items():
            formatted_placeholder = f"<|wildcard_{category_name}|>"
            formatted_prompt_by_user = formatted_prompt_by_user.replace(formatted_placeholder, original_tags)
        
        user_tags_list = [tag.strip() for tag in formatted_prompt_by_user.split(',') if tag.strip()]
        formatted_prompt_by_user = ", ".join(list(dict.fromkeys(user_tags_list)))
        unformatted_prompt_by_user = all_original_tags_str + "\n" + nl_prompt
        
        return (
            final_prompt, formatted_prompt_by_user, unformatted_prompt_by_tipo, unformatted_prompt_by_user,
            category_outputs.get("wildcard_appearance", ""),
            category_outputs.get("wildcard_clothing", ""),
            category_outputs.get("wildcard_background", ""),
            category_outputs.get("wildcard_pose_emotion", ""),
            category_outputs.get("wildcard_camera_lighting", ""),
            category_outputs.get("wildcard_art_style", ""),
        )

# --- ノードのマッピング定義 ---
NODE_CLASS_MAPPINGS = {
    "TIPO": TIPO,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPO": "TIPO Wildcard (Refactored)",
}
