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
    banタグの最終整形時フィルタを実装し、再混入を防止します。
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
<|characters|>, 
<|copyrights|>, 
<|artist|>, 
<|general|>,
<|wildcard_clothing|>, 
<|wildcard_background|>, 
<|wildcard_pose_emotion|>, 
<|wildcard_camera_lighting|>, 
<|wildcard_art_style|>""",
                        "multiline": True,
                    },
                ),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                "tag_length": ("STRING", {"default": "medium"}),
                "nl_length": ("STRING", {"default": "2 sentences"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "device": ("STRING", {"default": "auto"}),
            }
        }

    # ★ 戻り値の数を 9 に修正
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "final_prompt",
        "formatted_prompt_by_user",
        "unformatted_prompt_by_tipo",
        "main_processed",
        "wildcard_clothing",
        "wildcard_background",
        "wildcard_pose_emotion",
        "wildcard_camera_lighting",
        "wildcard_art_style",
    )
    FUNCTION = "execute"
    CATEGORY = "TIPO"

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
        
        # --- ban_tagsをリストとして保持 ---
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        tipo.BAN_TAGS = black_list
        
        final_prompt_parts = {}
        all_original_tags_list = []
        all_addon_tags_list = []

        # Part 1: メインの 'tags' と 'nl_prompt' を処理
        if tags.strip() or nl_prompt.strip():
            prompt_parse_strength = util.parse_prompt_attention(tags)
            nl_prompt_parse_strength = util.parse_prompt_attention(nl_prompt)

            strength_map = { p[0].strip(): p[1] for p in prompt_parse_strength }
            strength_map_nl = [(p[0].strip(), p[1]) for p in nl_prompt_parse_strength if p[0] != "BREAK"]

            main_all_tags = [p[0].strip() for p in prompt_parse_strength if p[0] and p[0] != "BREAK"]
            all_original_tags_list.extend(main_all_tags)

            meta, operations, general, nl_prompt_processed = parse_tipo_request(
                format, main_all_tags, aspect_ratio,
                tagtop_replace_underscore=True,
                valid_only=True,
                lowercase=True,
                general_by="regex",
                seperate_by=",",
                seperate_space=True,
                unify_space=True,
                remove_whitespace=True,
                check_edge_space=True,
                tag_to_replace=main_all_tags,
                escape_text=True,
                merge_general=True,
                tag_length=tag_length,
                nl_length=nl_length,
                generate_nl_prompt=nl_prompt.strip() != "",
                remove_nl_tags=True,
                text_to_replace=nl_prompt_processed if nl_prompt else "",
                lowercase_nl=True,
                tagtop_replace_space=True,
                texttop_replace_space=True,
                # texttop_to_lowercase は下で一度だけ指定（★重複回避）
                merge_nl_prompt=True,
                replace_text_to_single_space=True,
                tagtop_replace_space_with_underscore=False,
                texttop_replace_space_with_underscore=False,
                generate_extra_tags=("<|extended|>" in format or "<|special|>" in format),
                escape_general=True,
                remove_duplicate=True,
                remove_empty=True,
                replace_space=True,
                replace_underscore=True,
                output_formatter=seperate_tags,
                post_formatter=apply_format,
                post_format_text_replace_underscore=True,
                meta_data={
                    "length": tag_length,
                    "nl_length": nl_length,
                    "replace_underscore": True,
                    "replace_space": True,
                },
                tagtop=main_all_tags,
                texttop=nl_prompt,
                tagtop_replace_space_for_comma=True,
                texttop_replace_space_for_comma=True,
                tagtop_replace_space_with_underscore_for_comma=True,
                texttop_replace_space_with_underscore_for_comma=True,
                lowercase_general=True,
                replace_general_space_with_underscore=True,
                replace_general_space=True,
                replace_texttop_space_for_comma=False,
                replace_tagtop_space_for_comma=False,
                general_by_tagtop=True,
                general_limit=256,
                seed=seed,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                length_scale=1.0,
                texttop_length_scale=1.0,
                tagtop_length_scale=1.0,
                general_length_scale=1.0,
                smart_tagtop=True,
                texttop_replace_space_for_escape=True,
                replace_texttop_space_for_escape=True,
                replace_tagtop_space_for_escape=True,
                tagtop_replace_space_with_underscore_for_escape=True,
                operations=operations,
                lowercase_special=True,
                lowercase_characters=True,
                lowercase_copyrights=True,
                lowercase_artist=True,
                lowercase_general_category=True,
                lowercase_wildcard=True,
                remove_wildcard_duplicate=True,
                remove_special_duplicate=True,
                remove_characters_duplicate=True,
                remove_copyrights_duplicate=True,
                remove_artist_duplicate=True,
                remove_general_duplicate=True,
                remove_duplicate_tags=True,
                remove_duplicate_text=True,
                replace_space_with_underscore=False,
                replace_space_with_underscore_for_escape=True,
                tagtop_to_lowercase=True,
                texttop_to_lowercase=True,  # ★ ここで一度だけ指定
                output_format_replace_space_with_underscore=True,
                output_format_replace_space=True,
                generated_text_replace_space=True,
                generated_text_replace_space_with_underscore=True,
                output_format_lowercase=True,
                output_format_escape=True,
                output_format_replace_space_with_single_space=True,
                output_format_replace_underscore=True,
                texttop_split_by_comma=True,
                tagtop_split_by_comma=True,
                generated_text_split_by_comma=True,
                generated_text_split_by_comma_for_escape=True,
                output_format_replace_space_with_comma=True,
                merge_texttop=True,
                merge_tagtop=True,
                allow_multiple=True,
                replace_comma_with_space=True,
                tagtop_strip_space=True,
                texttop_strip_space=True,
                texttop_replace_commas=True,
                tagtop_replace_commas=True,
                # ここからメタ情報
                meta={
                    "aspect_ratio": f"{aspect_ratio:.1f}",
                    "seed": seed,
                    "temperature": temperature,
                },
                # 生成オプション
                max_new_tokens=128,
                repetition_penalty=1.1,
                device=device,
                replace_underscore=True,
                generate_extra_tags_for_output=True,
                generate_extra_text_for_output=True,
                output_general=True,
                output_special=True,
                output_characters=True,
                output_copyrights=True,
                output_artist=True,
                generate_extra_wildcard=True,
                tagtop_replace_space_with_underscore_for_output=True,
                texttop_replace_space_with_underscore_for_output=True,
                tagtop_replace_underscore_with_space=False,
                texttop_replace_underscore_with_space=False,
                generate_extra_prompt=True,
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
                    tag_list = list(dict.fromkeys(tag_list))
                    # ★ ban除去を追加
                    tag_list = util.filter_banned_tags(tag_list, black_list)
                    tag_map_main[key] = tag_list

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

            # ループ内でも毎回ban_tagsを再設定（保険）
            tipo.BAN_TAGS = black_list

            logger.info(f"TIPO is extending category: <|{placeholder_key}|>")
            cat_all_tags = [t.strip() for t in category_tags.split(',') if t.strip()]
            original_tags_set = set(cat_all_tags)

            # 各カテゴリ用の meta を再構築
            meta, operations, general, nl_prompt_processed = parse_tipo_request(
                format, cat_all_tags, aspect_ratio,
                tagtop_replace_underscore=True,
                valid_only=True,
                lowercase=True,
                general_by="regex",
                seperate_by=",",
                seperate_space=True,
                unify_space=True,
                remove_whitespace=True,
                check_edge_space=True,
                tag_to_replace=cat_all_tags,
                escape_text=True,
                merge_general=True,
                tag_length=tag_length,
                nl_length=nl_length,
                generate_nl_prompt=False,
                remove_nl_tags=True,
                text_to_replace="",
                lowercase_nl=True,
                tagtop_replace_space=True,
                texttop_replace_space=True,
                # texttop_to_lowercase は下で一度だけ指定（★重複回避）
                merge_nl_prompt=False,
                replace_text_to_single_space=True,
                tagtop_replace_space_with_underscore=False,
                texttop_replace_space_with_underscore=False,
                generate_extra_tags=("<|extended|>" in format or "<|special|>" in format),
                escape_general=True,
                remove_duplicate=True,
                remove_empty=True,
                replace_space=True,
                replace_underscore=True,
                output_formatter=seperate_tags,
                post_formatter=apply_format,
                post_format_text_replace_underscore=True,
                meta_data={
                    "length": tag_length,
                    "nl_length": nl_length,
                    "replace_underscore": True,
                    "replace_space": True,
                },
                tagtop=cat_all_tags,
                texttop="",
                tagtop_replace_space_for_comma=True,
                texttop_replace_space_for_comma=True,
                tagtop_replace_space_with_underscore_for_comma=True,
                texttop_replace_space_with_underscore_for_comma=True,
                lowercase_general=True,
                replace_general_space_with_underscore=True,
                replace_general_space=True,
                replace_texttop_space_for_comma=False,
                replace_tagtop_space_for_comma=False,
                general_by_tagtop=True,
                general_limit=256,
                seed=current_seed,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                length_scale=1.0,
                texttop_length_scale=1.0,
                tagtop_length_scale=1.0,
                general_length_scale=1.0,
                smart_tagtop=True,
                texttop_replace_space_for_escape=True,
                replace_texttop_space_for_escape=True,
                replace_tagtop_space_for_escape=True,
                tagtop_replace_space_with_underscore_for_escape=True,
                operations=operations,
                lowercase_special=True,
                lowercase_characters=True,
                lowercase_copyrights=True,
                lowercase_artist=True,
                lowercase_general_category=True,
                lowercase_wildcard=True,
                remove_wildcard_duplicate=True,
                remove_special_duplicate=True,
                remove_characters_duplicate=True,
                remove_copyrights_duplicate=True,
                remove_artist_duplicate=True,
                remove_general_duplicate=True,
                remove_duplicate_tags=True,
                remove_duplicate_text=True,
                replace_space_with_underscore=False,
                replace_space_with_underscore_for_escape=True,
                tagtop_to_lowercase=True,
                texttop_to_lowercase=True,  # ★ ここで一度だけ指定
                output_format_replace_space_with_underscore=True,
                output_format_replace_space=True,
                generated_text_replace_space=True,
                generated_text_replace_space_with_underscore=True,
                output_format_lowercase=True,
                output_format_escape=True,
                output_format_replace_space_with_single_space=True,
                output_format_replace_underscore=True,
                texttop_split_by_comma=True,
                tagtop_split_by_comma=True,
                generated_text_split_by_comma=True,
                generated_text_split_by_comma_for_escape=True,
                output_format_replace_space_with_comma=True,
                merge_texttop=True,
                merge_tagtop=True,
                allow_multiple=True,
                replace_comma_with_space=True,
                tagtop_strip_space=True,
                texttop_strip_space=True,
                texttop_replace_commas=True,
                tagtop_replace_commas=True,
                meta={
                    "aspect_ratio": f"{aspect_ratio:.1f}",
                    "seed": current_seed,
                    "temperature": temperature,
                },
                max_new_tokens=96,
                repetition_penalty=1.1,
                device=device,
                replace_underscore=True,
                generate_extra_tags_for_output=True,
                generate_extra_text_for_output=False,
                output_general=True,
                output_special=True,
                output_characters=True,
                output_copyrights=True,
                output_artist=True,
                generate_extra_wildcard=True,
                tagtop_replace_space_with_underscore_for_output=True,
                texttop_replace_space_with_underscore_for_output=True,
                tagtop_replace_underscore_with_space=False,
                texttop_replace_underscore_with_space=False,
                generate_extra_prompt=True,
                generate_extra_nl_prompt=("<|extended|>" in format or "<|generated|>" in format)
            )
            meta["aspect_ratio"] = f"{aspect_ratio:.1f}"

            cat_tag_map, _ = tipo_runner(
                meta, operations, general, nl_prompt_processed,
                temperature=temperature, seed=current_seed, top_p=top_p, min_p=min_p, top_k=top_k,
            )

            # 生成結果から無効タグを除去
            if 'general' in cat_tag_map and isinstance(cat_tag_map['general'], list):
                cat_tag_map['general'] = [
                    tag for tag in cat_tag_map['general'] 
                    if not invalid_tag_pattern.match(tag)
                ]

            # 生成結果の重複除去
            for key, tag_list in cat_tag_map.items():
                if isinstance(tag_list, list):
                    cat_tag_map[key] = list(dict.fromkeys(tag_list))

            # 新規追加タグを抽出
            addon_tags = []
            for tag_list in cat_tag_map.values():
                if isinstance(tag_list, list):
                    for tag in tag_list:
                        if tag not in original_tags_set and not invalid_tag_pattern.match(tag):
                            addon_tags.append(tag)
            
            combined_tags_list = cat_all_tags + addon_tags
            unique_tags_list = list(dict.fromkeys(combined_tags_list))
            # ★ ban除去を追加
            unique_tags_list = util.filter_banned_tags(unique_tags_list, black_list)
            all_addon_tags_list.extend(addon_tags)

            processed_category_tags = ", ".join(unique_tags_list)
            final_prompt_parts[placeholder_key] = processed_category_tags
            category_outputs[placeholder_key] = processed_category_tags
            current_seed += 1

        # Part 3: 最終的な組み立てと他の戻り値の再構築
        final_prompt = apply_format(final_prompt_parts, format)
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        # ★ ban除去を追加
        final_tags_list = util.filter_banned_tags(final_tags_list, black_list)
        final_prompt = ", ".join(list(dict.fromkeys(final_tags_list)))

        all_original_tags_str = ", ".join(list(dict.fromkeys(all_original_tags_list)))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(all_addon_tags_list)))
        unformatted_prompt_by_tipo = (all_original_tags_str + ", " + unformatted_addon_tags).strip(", ")
        
        user_prompt_parts = seperate_tags(all_original_tags_list)
        formatted_prompt_by_user = apply_format(user_prompt_parts, format)
        
        main_processed = ", ".join(tag_map_main.get("general", [])) if 'tag_map_main' in locals() else ""
        
        return (
            final_prompt,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            main_processed,
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
