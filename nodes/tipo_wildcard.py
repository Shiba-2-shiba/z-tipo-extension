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
    ★ 改善版: AIによる追加タグを含めた全タグを最適配置します。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_prompts": ("TIPO_PROMPTS",),
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
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

    # ★ 変更点: 関数名をより分かりやすくし、戻り値をTIPOのパース結果全体から抽出したクリーンなタグリストに変更
    def _get_expanded_tags(self, tags_list, aspect_ratio, tag_length, temperature, seed, top_p, min_p, top_k, black_list):
        """
        指定されたタグリストをTIPOで拡張し、クリーンなタグリストを返す共通関数
        """
        if not tags_list:
            return []

        tipo.BAN_TAGS = black_list
        org_tag_map = seperate_tags(tags_list)
        
        meta, operations, general, _ = tipo_single_request(
            org_tag_map, 
            nl_prompt="",
            tag_length_target=tag_length,
            operation="short_to_tag"
        )
        
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        
        # tipo_runnerを実行し、パース済みの辞書結果を取得
        parsed_result, _ = tipo_runner(
            meta, operations, general, "",
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k
        )

        # ★ 変更点: パース結果から'general'と'special'タグを抽出し、結合する
        # これにより、プロンプト構築時の命令（aspect_ratioなど）が混入するのを防ぐ
        expanded_tags = parsed_result.get("special", []) + parsed_result.get("general", [])
        
        # 重複を除去し、BANリストにないものだけを返す
        unique_tags = list(dict.fromkeys(expanded_tags))
        final_tags = [tag for tag in unique_tags if tag and tag not in black_list]

        return final_tags

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
        
        # ★ 変更点: 全てのタグを保持するリストを準備
        all_processed_tags_list = []
        category_outputs = {}
        current_seed = seed

        # --- Part 1: メイン、外見、ポーズ/表情を結合して処理 ---
        logger.info("TIPO is processing: <|wildcard_character_pose|>")
        appearance_tags_str = category_prompts.pop("appearance", "")
        pose_emotion_tags_str = category_prompts.pop("pose_emotion", "")
        
        combined_main_tags_str = ", ".join(filter(None, [main_tags_str, appearance_tags_str, pose_emotion_tags_str]))
        main_tags_list = [t.strip() for t in combined_main_tags_str.split(',') if t.strip()]

        # ★ 変更点: クリーンなタグリストを取得
        processed_main_tags = self._get_expanded_tags(
            main_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
        )
        all_processed_tags_list.extend(processed_main_tags)
        
        main_output_str = ", ".join(processed_main_tags)
        category_outputs["wildcard_character_pose"] = main_output_str
        current_seed += 1
        
        # --- Part 2: 残りの各カテゴリ（服装、背景）を個別に処理 ---
        for category_name, category_tags_str in category_prompts.items():
            placeholder_key = f"wildcard_{category_name}"
            logger.info(f"TIPO is processing: <|{placeholder_key}|>")

            if not category_tags_str.strip():
                category_outputs[placeholder_key] = ""
                continue

            cat_tags_list = [t.strip() for t in category_tags_str.split(',') if t.strip()]

            processed_cat_tags = self._get_expanded_tags(
                cat_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
            )
            all_processed_tags_list.extend(processed_cat_tags)

            cat_output_str = ", ".join(processed_cat_tags)
            category_outputs[placeholder_key] = cat_output_str
            current_seed += 1

        # ★★★ 変更点: 全てのタグを再分類し、最終プロンプトを構築 ★★★
        # これが「効果的な配置」を実現する部分
        final_prompt_parts = seperate_tags(all_processed_tags_list)

        # ワイルドカード部分を、処理済みの各カテゴリ文字列で上書き
        final_prompt_parts["wildcard_character_pose"] = category_outputs.get("wildcard_character_pose", "")
        final_prompt_parts["wildcard_clothing"] = category_outputs.get("wildcard_clothing", "")
        final_prompt_parts["wildcard_background"] = category_outputs.get("wildcard_background", "")
        
        # 最終的なプロンプトをフォーマットに従って生成
        final_prompt = apply_format(final_prompt_parts, format)
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        final_tags_list = [tag for tag in final_tags_list if tag not in black_list]
        final_prompt = ", ".join(list(dict.fromkeys(final_tags_list)))
        
        # unformatted_prompt_by_tipoの再計算
        # 元々の入力タグと、AIによって追加されたタグを区別
        original_tags_set = set()
        original_tags_set.update([t.strip() for t in main_tags_str.split(',') if t.strip()])
        for cat_tags in tipo_prompts.get("categories", {}).values():
            original_tags_set.update([t.strip() for t in cat_tags.split(',') if t.strip()])
        
        addon_tags = [tag for tag in all_processed_tags_list if tag not in original_tags_set]
        
        all_original_tags_str = ", ".join(list(dict.fromkeys(sorted(list(original_tags_set)))))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(addon_tags)))
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
