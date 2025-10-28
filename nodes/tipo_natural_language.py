# -*- coding: utf-8 -*-
import os
import re
from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import tipo_single_request, tipo_runner
from kgen.formatter import seperate_tags
from kgen.logging import logger

# 共通機能を util からインポート
from . import util

class TIPONaturalLanguage:
    """
    Preprocessorからの入力を使用して、文脈に応じた自然言語プロンプトを生成する改修版ノード。
    関連カテゴリを統合処理することで、より一貫性のある自然言語記述を生成します。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_prompts": ("TIPO_PROMPTS",),
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
                "width": ("INT", {"default": 1024, "max": 16384}),
                "height": ("INT", {"default": 1024, "max": 16384}),
                "temperature": ("FLOAT", {"default": 0.6, "step": 0.01, "min": 0.1, "max": 1.5}),
                "top_p": ("FLOAT", {"default": 0.95, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "step": 0.01}),
                "top_k": ("INT", {"default": 80}),
                # ★ 変更点: 全体とカテゴリの長さを分ける必要がなくなったため一本化
                "nl_length": (["short", "medium", "long", "very_long"], {"default": "long"}),
                "seed": ("INT", {"default": 1234, "min": -1, "max": 0xffffffffffffffff}),
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    # ★ 変更点: 不要な出力を削除
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "final_nl", 
        "character_pose_nl", 
        "clothing_nl",
        "background_nl",
    )
    FUNCTION = "execute"
    CATEGORY = "TIPO"

    def _post_process_nl(self, nl_parts: list[str]) -> str:
        """重複や定型文を削除し、自然な文章に整形する"""
        redundant_starters = [
            r'^\s*In this scene,?\s*', r'^\s*The artwork depicts\s*', r'^\s*This image features\s*',
            r'^\s*It features\s*', r'^\s*The image shows\s*', r'^\s*The scene is\s*',
        ]
        pattern = re.compile('|'.join(redundant_starters), re.IGNORECASE)
        seen_sentences = set()
        final_sentences = []
        for nl_part in nl_parts:
            if not nl_part or not nl_part.strip():
                continue
            sentences = re.split(r'(?<=[.!?])\s+', nl_part.strip())
            for sentence in sentences:
                if not sentence or not sentence.strip():
                    continue
                cleaned_sentence = pattern.sub('', sentence).strip()
                if cleaned_sentence:
                    # 文頭を大文字にする
                    cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                # 小文字に変換して重複チェック
                if cleaned_sentence and cleaned_sentence.lower() not in seen_sentences:
                    seen_sentences.add(cleaned_sentence.lower())
                    final_sentences.append(cleaned_sentence)
        return " ".join(final_sentences)

    def _generate_nl_for_category(
        self, tags_list, aspect_ratio, nl_length, base_nl_context,
        temperature, seed, top_p, min_p, top_k
    ):
        """指定されたタグリストから自然言語記述を生成する共通関数"""
        if not tags_list:
            return ""

        org_tag_map = seperate_tags(tags_list)
        
        # 文脈がある場合はそれを活用し、ない場合はタグから直接生成
        operation = "short_to_tag_to_long" if base_nl_context else "tag_to_long"
        
        meta, ops, general, nl_prompt = tipo_single_request(
            org_tag_map, 
            nl_prompt=base_nl_context,
            nl_length_target=nl_length,
            operation=operation
        )
        meta["aspect_ratio"] = f"{aspect_ratio:.1f}"
        
        tag_map, _ = tipo_runner(
            meta, ops, general, nl_prompt,
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        
        return tag_map.get("generated", tag_map.get("extended", ""))


    def execute(
        self,
        tipo_prompts: dict,
        tipo_model: str,
        width: int, height: int,
        temperature: float, top_p: float, min_p: float, top_k: int,
        nl_length: str,
        seed: int, device: str,
    ):
        util.load_tipo_model(tipo_model, device)
        
        main_tags_str = tipo_prompts.get("main", {}).get("tags", "")
        ban_tags = tipo_prompts.get("main", {}).get("ban_tags", "")
        category_prompts = tipo_prompts.get("categories", {})
        
        aspect_ratio = width / height
        black_list = [t.strip() for t in ban_tags.split(",") if t.strip()]
        tipo.BAN_TAGS = black_list

        if seed == -1:
            seed = int.from_bytes(os.urandom(8), 'big')

        category_nls = {}
        current_seed = seed

        # --- Part 1: ★★★ メイン、外見、ポーズ/表情を結合し、核となるNLを生成 ★★★
        logger.info("Generating core NL for character and pose...")
        appearance_tags_str = category_prompts.pop("appearance", "")
        pose_emotion_tags_str = category_prompts.pop("pose_emotion", "")
        
        combined_main_tags_str = ", ".join(filter(None, [main_tags_str, appearance_tags_str, pose_emotion_tags_str]))
        main_tags_list = [t.strip() for t in combined_main_tags_str.split(',') if t.strip()]

        character_pose_nl = self._generate_nl_for_category(
            main_tags_list, aspect_ratio, nl_length, "", # 最初は文脈なし
            temperature, current_seed, top_p, min_p, top_k
        )
        category_nls["character_pose_nl"] = character_pose_nl
        logger.info(f"  -> Core NL: {character_pose_nl[:100]}...")
        current_seed += 1

        # --- Part 2: ★★★ 残りのカテゴリを、核NLを文脈として利用して個別に生成 ★★★
        # 不要なカテゴリを予め除外
        for cat in ["camera_lighting", "art_style"]:
            category_prompts.pop(cat, None)

        for name, category_tags_str in category_prompts.items():
            output_key = name + "_nl"
            if not category_tags_str.strip():
                category_nls[output_key] = ""
                continue
            
            logger.info(f"Generating contextual NL for category: {name}")
            cat_tags = [t.strip() for t in category_tags_str.split(',') if t.strip()]

            # 核となるNLを文脈として渡す
            contextual_nl = self._generate_nl_for_category(
                cat_tags, aspect_ratio, "short", # カテゴリ別は短めに設定
                character_pose_nl, # ★文脈として利用
                temperature, current_seed, top_p, min_p, top_k
            )
            category_nls[output_key] = contextual_nl
            logger.info(f"  -> {name} NL: {contextual_nl[:100]}...")
            current_seed += 1

        # --- Part 3: 全てのNLパーツを統合して最終的な文章を生成 ---
        all_nl_parts = [
            category_nls.get("character_pose_nl", ""),
            category_nls.get("clothing_nl", ""),
            category_nls.get("background_nl", ""),
        ]
        final_nl = self._post_process_nl(all_nl_parts)
        logger.info(f"Final integrated NL: {final_nl[:150]}...")

        return (
            final_nl,
            category_nls.get("character_pose_nl", ""),
            category_nls.get("clothing_nl", ""),
            category_nls.get("background_nl", ""),
        )

# --- ノードのマッピング定義 ---
NODE_CLASS_MAPPINGS = {
    "TIPONaturalLanguage": TIPONaturalLanguage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPONaturalLanguage": "TIPO Natural Language (Refactored)",
}
