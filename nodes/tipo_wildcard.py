# -*- coding: utf-8 -*-
import re
from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import tipo_single_request, tipo_runner
from kgen.formatter import seperate_tags, apply_format
from kgen.logging import logger

# 共通機能を util からインポート
from . import util

class TIPOWildcard:
    """
    Preprocessorからのカテゴリ別タグ入力を受け取り、それぞれを個別に拡張して出力するノード。
    「メイン」「外見」「ポーズ/表情」を結合して処理し、重複を削減します。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_prompts": ("TIPO_PROMPTS",),
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
                # ★ 変更点: デフォルトのフォーマットを新しい出力に合わせて修正
                "format": (
                    "STRING",
                    {
                        "default": """<|special|>, <|characters|>, <|copyrights|>, <|artist|>, 
<|wildcard_character_pose|>,
<|wildcard_clothing|>, 
<|wildcard_background|>,
<|general|>,
<|quality|>, <|meta|>, <|rating|>""",
                        "multiline": True,
                    },
                ),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                "tag_length": (["short", "medium", "long", "very_long"], {"default": "medium"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    # ★ 変更点: 出力を5つに削減
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "final_prompt",
        "unformatted_prompt_by_tipo",
        "wildcard_character_pose",
        "wildcard_clothing",
        "wildcard_background",
    )
    FUNCTION = "execute"
    CATEGORY = "TIPO"

    def _process_category(self, tags_list, aspect_ratio, tag_length, temperature, seed, top_p, min_p, top_k, black_list):
        """
        指定されたタグリストをKgenライブラリの仕様に沿って適切に拡張する共通関数
        """
        if not tags_list:
            return [], []

        tipo.BAN_TAGS = black_list
        org_tag_map = seperate_tags(tags_list)
        meta, operations, general, _ = tipo_single_request(
            org_tag_map, 
            nl_prompt="",
            tag_length_target=tag_length,
            operation="short_to_tag"
        )

        if not general and not any(org_tag_map.values()):
            logger.warning(f"No processable tags found for input: {tags_list}. Returning original tags.")
            return tags_list, []
        
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        tag_map, _ = tipo_runner(
            meta, operations, general, "",
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k
        )

        original_tags_set = set(tags_list)
        addon_tags = []
        for category, generated_tags in tag_map.items():
            if isinstance(generated_tags, list):
                for tag in generated_tags:
                    tag_stripped = tag.strip()
                    if tag_stripped and tag_stripped not in original_tags_set:
                        addon_tags.append(tag_stripped)
        
        final_addon_tags = [tag for tag in list(dict.fromkeys(addon_tags)) if tag and tag not in black_list]
        return final_addon_tags

    def execute(
        self,
        tipo_prompts: dict,
        tipo_model: str,
        format: str,
        width: int, height: int,
        temperature: float, top_p: float, min_p: float, top_k: int,
        tag_length: str,
        seed: int, device: str,
    ):
        util.load_tipo_model(tipo_model, device)
        
        main_tags_str = tipo_prompts.get("main", {}).get("tags", "")
        ban_tags = tipo_prompts.get("main", {}).get("ban_tags", "")
        category_prompts = tipo_prompts.get("categories", {})
        
        aspect_ratio = width / height
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        
        all_original_tags_list = []
        all_addon_tags_list = []
        final_prompt_parts = {}
        category_outputs = {}
        current_seed = seed

        # --- Part 1: ★★★ メイン、外見、ポーズ/表情を結合して処理 ★★★
        logger.info("TIPO is processing: <|wildcard_character_pose|>")
        appearance_tags_str = category_prompts.pop("appearance", "")
        pose_emotion_tags_str = category_prompts.pop("pose_emotion", "")
        
        combined_main_tags_str = ", ".join(filter(None, [main_tags_str, appearance_tags_str, pose_emotion_tags_str]))
        main_tags_list = [t.strip() for t in combined_main_tags_str.split(',') if t.strip()]
        all_original_tags_list.extend(main_tags_list)

        addon_main_tags = self._process_category(
            main_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
        )
        all_addon_tags_list.extend(addon_main_tags)
        
        processed_main_tags = main_tags_list + addon_main_tags
        main_output_str = ", ".join(list(dict.fromkeys(processed_main_tags)))
        final_prompt_parts["wildcard_character_pose"] = main_output_str
        category_outputs["wildcard_character_pose"] = main_output_str
        current_seed += 1
        
        # --- Part 2: 残りの各カテゴリ（服装、背景）を個別に処理 ---
        for category_name, category_tags_str in category_prompts.items():
            placeholder_key = f"wildcard_{category_name}"
            logger.info(f"TIPO is processing: <|{placeholder_key}|>")

            if not category_tags_str.strip():
                final_prompt_parts[placeholder_key] = ""
                category_outputs[placeholder_key] = ""
                continue

            cat_tags_list = [t.strip() for t in category_tags_str.split(',') if t.strip()]
            all_original_tags_list.extend(cat_tags_list)

            addon_cat_tags = self._process_category(
                cat_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
            )
            all_addon_tags_list.extend(addon_cat_tags)

            processed_cat_tags = cat_tags_list + addon_cat_tags
            cat_output_str = ", ".join(list(dict.fromkeys(processed_cat_tags)))
            final_prompt_parts[placeholder_key] = cat_output_str
            category_outputs[placeholder_key] = cat_output_str
            current_seed += 1

        # --- Part 3: 最終的な組み立て ---
        org_tags_for_format = seperate_tags(all_original_tags_list)
        for key, value in org_tags_for_format.items():
            if f"<|{key}|>" in format:
                 final_prompt_parts[key] = ", ".join(value)

        final_prompt = apply_format(final_prompt_parts, format)
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        final_tags_list = [tag for tag in final_tags_list if tag not in black_list]
        final_prompt = ", ".join(list(dict.fromkeys(final_tags_list)))
        
        all_original_tags_str = ", ".join(list(dict.fromkeys(all_original_tags_list)))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(all_addon_tags_list)))
        unformatted_prompt_by_tipo = (all_original_tags_str + ", " + unformatted_addon_tags).strip(", ")
        
        return (
            final_prompt,
            unformatted_prompt_by_tipo,
            category_outputs.get("wildcard_character_pose", ""),
            category_outputs.get("wildcard_clothing", ""),
            category_outputs.get("wildcard_background", ""),
        )

# --- ノードのマッピング定義 ---
NODE_CLASS_MAPPINGS = {
    "TIPOWildcard": TIPOWildcard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPOWildcard": "TIPO Wildcard (Refactored)",
}
