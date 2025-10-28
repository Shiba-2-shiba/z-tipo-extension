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
    「外見」カテゴリはメインタグと結合して処理される。NL機能は削除し、タグ拡張に特化。
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
<|wildcard_appearance|>,
<|wildcard_clothing|>, 
<|wildcard_background|>, 
<|wildcard_pose_emotion|>, 
<|wildcard_camera_lighting|>, 
<|wildcard_art_style|>,
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
                # ★ 変更点: 文字列入力からリスト選択に戻す
                "tag_length": (["short", "medium", "long", "very_long"], {"default": "medium"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
            # ★ 変更点: 不要な nl_length を削除
        }

    # ★ 変更点: wildcard_appearance を含む10個の出力に修正
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "final_prompt",
        "formatted_prompt_by_user",
        "unformatted_prompt_by_tipo",
        "wildcard_appearance", # ★ 追加
        "wildcard_clothing",
        "wildcard_background",
        "wildcard_pose_emotion",
        "wildcard_camera_lighting",
        "wildcard_art_style",
        "main_processed_tags(for debug)",
    )
    FUNCTION = "execute"
    CATEGORY = "TIPO"

    def _process_category(self, tags_list, aspect_ratio, tag_length, temperature, seed, top_p, min_p, top_k, black_list):
        """
        [改善版] 指定されたタグリストをKgenライブラリの仕様に沿って適切に拡張する共通関数
        """
        if not tags_list:
            return [], []

        # 1. BANタグをライブラリに設定
        tipo.BAN_TAGS = black_list

        # 2. ★★★ ライブラリの formatter を使用してタグをカテゴリ分けする ★★★
        # これが最も重要な改善点です。
        org_tag_map = seperate_tags(tags_list)

        # 3. カテゴリ分けされたタグマップを基に、AIへのリクエストを生成
        meta, operations, general, _ = tipo_single_request(
            org_tag_map, 
            nl_prompt="",  # このノードではNLは使用しない
            tag_length_target=tag_length,
            operation="short_to_tag"  # タグ拡張の操作を指定
        )

        # 4. 処理対象のタグがない場合は、早期にリターン（エラー防止）
        if not general and not any(org_tag_map.values()):
            logger.warning(f"No processable tags found for input: {tags_list}. Returning original tags.")
            return tags_list, []
        
        # 5. メタ情報にアスペクト比を追加
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        
        # 6. AIモデルを実行してタグを拡張
        tag_map, _ = tipo_runner(
            meta, operations, general, "",
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k
        )

        # 7. 結果を統合し、クリーンアップする
        original_tags_set = set(tags_list)
        combined_tags = list(tags_list)
        addon_tags = []

        # tag_map には拡張されたタグがカテゴリ別に格納されている
        for category, generated_tags in tag_map.items():
            if isinstance(generated_tags, list):
                for tag in generated_tags:
                    tag_stripped = tag.strip()
                    if tag_stripped and tag_stripped not in original_tags_set:
                        # 元のタグになかったものを追加タグとして記録
                        addon_tags.append(tag_stripped)
                        combined_tags.append(tag_stripped)
        
        # 8. 最終的なフィルタリング（重複除去とBANタグ除去）
        unique_tags = list(dict.fromkeys(combined_tags))
        final_tags = [tag for tag in unique_tags if tag and tag not in black_list]
        
        unique_addon_tags = list(dict.fromkeys(addon_tags))
        final_addon_tags = [tag for tag in unique_addon_tags if tag and tag not in black_list]

        return final_tags, final_addon_tags

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
        
        # --- Preprocessorからのデータを展開 ---
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

        # --- Part 1: 「外見 (appearance)」カテゴリの特別処理 ---
        logger.info("TIPO is processing: <|wildcard_appearance|>")
        appearance_tags_str = category_prompts.pop("appearance", "") # 辞書から取り出し、後でループさせない
        
        # メインタグと外見タグを結合
        combined_appearance_tags_str = ", ".join(filter(None, [main_tags_str, appearance_tags_str]))
        appearance_tags_list = [t.strip() for t in combined_appearance_tags_str.split(',') if t.strip()]
        all_original_tags_list.extend(appearance_tags_list)

        processed_appearance_tags, addon_app_tags = self._process_category(
            appearance_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
        )
        all_addon_tags_list.extend(addon_app_tags)
        
        # 結果を保存
        app_output_str = ", ".join(processed_appearance_tags)
        final_prompt_parts["wildcard_appearance"] = app_output_str
        category_outputs["wildcard_appearance"] = app_output_str
        main_processed_tags_output = app_output_str # main_processedの代替
        current_seed += 1
        
        # --- Part 2: 残りの各カテゴリを個別に処理 ---
        for category_name, category_tags_str in category_prompts.items():
            placeholder_key = f"wildcard_{category_name}"
            logger.info(f"TIPO is processing: <|{placeholder_key}|>")

            if not category_tags_str.strip():
                final_prompt_parts[placeholder_key] = ""
                category_outputs[placeholder_key] = ""
                continue

            cat_tags_list = [t.strip() for t in category_tags_str.split(',') if t.strip()]
            all_original_tags_list.extend(cat_tags_list)

            processed_cat_tags, addon_cat_tags = self._process_category(
                cat_tags_list, aspect_ratio, tag_length, temperature, current_seed, top_p, min_p, top_k, black_list
            )
            all_addon_tags_list.extend(addon_cat_tags)

            cat_output_str = ", ".join(processed_cat_tags)
            final_prompt_parts[placeholder_key] = cat_output_str
            category_outputs[placeholder_key] = cat_output_str
            current_seed += 1

        # --- Part 3: 最終的な組み立てと他の戻り値の再構築 ---
        # final_prompt_partsに元のタグ(generalなど)を追加
        org_tags_for_format = seperate_tags(all_original_tags_list)
        for key, value in org_tags_for_format.items():
            if f"<|{key}|>" in format:
                 final_prompt_parts[key] = ", ".join(value)

        final_prompt = apply_format(final_prompt_parts, format)
        final_tags_list = [tag.strip() for tag in final_prompt.split(',') if tag.strip()]
        final_tags_list = [tag for tag in final_tags_list if tag not in black_list] # 最終確認
        final_prompt = ", ".join(list(dict.fromkeys(final_tags_list)))

        # ユーザー入力時点でのプロンプトを再現
        user_prompt_parts = seperate_tags(all_original_tags_list)
        # カテゴリのプレースホルダーを元のタグで置換
        for cat_name in list(category_prompts.keys()) + ['appearance']:
            placeholder = f"<|wildcard_{cat_name}|>"
            original_tags = tipo_prompts.get("categories", {}).get(cat_name, "")
            if cat_name == 'appearance':
                original_tags = combined_appearance_tags_str
            user_prompt_parts[f"wildcard_{cat_name}"] = original_tags
            
        formatted_prompt_by_user = apply_format(user_prompt_parts, format)
        
        all_original_tags_str = ", ".join(list(dict.fromkeys(all_original_tags_list)))
        unformatted_addon_tags = ", ".join(list(dict.fromkeys(all_addon_tags_list)))
        unformatted_prompt_by_tipo = (all_original_tags_str + ", " + unformatted_addon_tags).strip(", ")
        
        return (
            final_prompt,
            formatted_prompt_by_user,
            unformatted_prompt_by_tipo,
            category_outputs.get("wildcard_appearance", ""),
            category_outputs.get("wildcard_clothing", ""),
            category_outputs.get("wildcard_background", ""),
            category_outputs.get("wildcard_pose_emotion", ""),
            category_outputs.get("wildcard_camera_lighting", ""),
            category_outputs.get("wildcard_art_style", ""),
            main_processed_tags_output,
        )

# --- ノードのマッピング定義 ---
NODE_CLASS_MAPPINGS = {
    "TIPOWildcard": TIPOWildcard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPOWildcard": "TIPO Wildcard (Refactored)",
}


