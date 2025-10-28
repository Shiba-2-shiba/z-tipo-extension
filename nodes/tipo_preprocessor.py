# -*- coding: utf-8 -*-
"""
TIPO Natural Language (Refactored)
- 文章生成の最終段で ban 語句を含む文をドロップ（またはマスク）する改修を追加。
- util.normalize_tag / filter_banned_tags 相当の正規化を流用し、再混入を抑止。
"""
import os
import re
from typing import Dict, List

from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import tipo_single_request
from kgen.formatter import seperate_tags
from kgen.logging import logger

# 共通ユーティリティ
from . import util


class TIPONaturalLanguage:
    """Preprocessor から受け取ったタグ群を元に NL（文章）を生成するノードの改修版。
    - overall（全体）の文章＋カテゴリ別（appearance/clothing/...）の文章を生成
    - 文章分割・重複除去・冗長書き出しの除去
    - ★ 追加: ban 語句を含む文の除外
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_prompts": ("TIPO_PROMPTS",),
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
                "width": ("INT", {"default": 832, "min": 256, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1216, "min": 256, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000, "step": 1}),
                # NL 長さ（overall / category）
                "overall_nl_length": (["very_short", "short", "long", "very_long"], {"default": "short"}),
                "category_nl_length": (["very_short", "short", "long", "very_long"], {"default": "short"}),
                # 乱数
                "seed": ("INT", {"default": 1234, "min": -1, "max": 0xffffffffffffffff}),
                # 実行デバイス
                "device": (["auto", "cpu", "cuda"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "final_nl", "overall_nl", "appearance_nl", "clothing_nl",
        "background_nl", "pose_emotion_nl", "camera_lighting_nl", "art_style_nl",
    )
    FUNCTION = "execute"
    CATEGORY = "TIPO"

    # --- 追加: ban 語句チェック ---
    def _contains_ban(self, sentence: str, ban_list: List[str]) -> bool:
        if not ban_list or not sentence:
            return False
        s_norm = util.normalize_tag(sentence)
        # ban 側も正規化 + 単純複数形を吸収
        ban_norm = set()
        for b in ban_list:
            n = util.normalize_tag(b)
            if not n:
                continue
            ban_norm.add(n)
            if not n.endswith("s"):
                ban_norm.add(n + "s")
        # 完全一致だけでなく、句中に含まれている場合も除外（NLは句が長い前提）
        return any(n in s_norm for n in ban_norm)

    def _post_process_nl(self, nl_parts: List[str], ban_list: List[str]) -> str:
        """文を分割→冗長な書き出しの除去→重複削除→ban 含有文のドロップ→整形。"""
        # よくある冗長な書き出しを削除
        redundant_starters = [
            r'^\s*In this scene,?\s*', r'^\s*The artwork depicts\s*',
            r'^\s*This image features\s*', r'^\s*It features\s*',
            r'^\s*by the artist\s*,?\s*', r'^\s*by artist\s*,?\s*',
            r'^\s*View from\s*', r'^\s*The image shows\s*', r'^\s*The scene is\s*',
        ]
        starter_pat = re.compile('|'.join(redundant_starters), re.IGNORECASE)

        seen = set()
        final_sentences: List[str] = []
        for part in nl_parts or []:
            if not part or not part.strip():
                continue
            # 文へ分割
            sentences = re.split(r'(?<=[.!?])\s+', part.strip())
            for s in sentences:
                if not s or not s.strip():
                    continue
                # スターター除去
                s2 = starter_pat.sub('', s).strip()
                # ban 語句含有なら捨てる
                if self._contains_ban(s2, ban_list):
                    continue
                # 同一文の重複除去（正規化キーで）
                key = util.normalize_tag(s2)
                if not key or key in seen:
                    continue
                seen.add(key)
                # 文末の終止記号を整える
                if not re.search(r'[.!?]$', s2):
                    s2 += '.'
                final_sentences.append(s2)
        return ' '.join(final_sentences)

    def _gen_nl_from_tags(
        self,
        tag_list: List[str],
        nl_length: str,
        temperature: float, top_p: float, min_p: float, top_k: int,
        seed: int, device: str,
    ) -> str:
        if not tag_list:
            return ""
        tag_map = seperate_tags(tag_list)
        meta, operations, general, _ = tipo_single_request(
            tag_map,
            nl_prompt="",
            tag_length_target="short",  # NL なのでタグ長ターゲットは固定で十分
            nl_length_target=nl_length.replace(' ', '_'),
            aspect_ratio="auto",
            generate_extra_prompt=False,
            generate_extra_text_for_output=True,
            output_general=False,
            output_special=False,
            output_characters=False,
            output_copyrights=False,
            output_artist=False,
            lowercase_general_category=True,
            lowercase_wildcard=True,
            remove_duplicate=True,
            remove_empty=True,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            device=device,
        )
        # 実行
        tag_map_out, text_out = tipo.tipo_runner(
            meta, operations, general, "",
            temperature=temperature, seed=seed, top_p=top_p, min_p=min_p, top_k=top_k,
        )
        # text_out は NL の候補が配列で返る前提
        if isinstance(text_out, list):
            return '\n'.join([t for t in text_out if isinstance(t, str)])
        return str(text_out or "")

    def execute(
        self,
        tipo_prompts: Dict,
        tipo_model: str,
        width: int, height: int,
        temperature: float, top_p: float, min_p: float, top_k: int,
        overall_nl_length: str, category_nl_length: str,
        seed: int, device: str,
    ):
        # モデルロード
        util.load_tipo_model(tipo_model, device)

        # --- Preprocessor から展開 ---
        tags = tipo_prompts.get("main", {}).get("tags", "")
        ban_tags = tipo_prompts.get("main", {}).get("ban_tags", "")
        category_prompts = tipo_prompts.get("categories", {})

        # ban 設定
        black_list = [t.strip() for t in ban_tags.split(',') if t.strip()]
        tipo.BAN_TAGS = black_list  # 生成側の抑制にも反映

        # seed -1 対応
        if seed == -1:
            seed = int.from_bytes(os.urandom(8), 'big')

        # 全体タグを収集（overall NL 用）
        all_tags: List[str] = [t.strip() for t in tags.split(',') if t.strip()]
        for cat_str in category_prompts.values():
            all_tags.extend([t.strip() for t in str(cat_str).split(',') if t.strip()])
        all_tags = list(dict.fromkeys(all_tags))

        # overall NL
        logger.info("Generating overall NL description...")
        overall_raw = self._gen_nl_from_tags(
            all_tags, overall_nl_length,
            temperature, top_p, min_p, top_k,
            seed, device,
        ) if all_tags else ""
        overall_nl = self._post_process_nl([overall_raw], black_list)

        # カテゴリ別 NL
        category_nls: Dict[str, str] = {}
        name_map = {
            'appearance': 'appearance_nl',
            'clothing': 'clothing_nl',
            'background': 'background_nl',
            'pose_emotion': 'pose_emotion_nl',
            'camera_lighting': 'camera_lighting_nl',
            'art_style': 'art_style_nl',
        }
        cur_seed = seed + 1
        for cat_key, out_key in name_map.items():
            src = str(category_prompts.get(cat_key, ""))
            if not src.strip():
                category_nls[out_key] = ""
                continue
            tag_list = [t.strip() for t in src.split(',') if t.strip()]
            raw = self._gen_nl_from_tags(
                tag_list, category_nl_length,
                temperature, top_p, min_p, top_k,
                cur_seed, device,
            )
            category_nls[out_key] = self._post_process_nl([raw], black_list)
            cur_seed += 1

        # 最終 NL（overall とカテゴリを連結して、もう一度 ban 検査を通す）
        final_nl = self._post_process_nl(
            [overall_nl] + list(category_nls.values()), black_list
        )

        return (
            final_nl,
            overall_nl,
            category_nls.get('appearance_nl', ''),
            category_nls.get('clothing_nl', ''),
            category_nls.get('background_nl', ''),
            category_nls.get('pose_emotion_nl', ''),
            category_nls.get('camera_lighting_nl', ''),
            category_nls.get('art_style_nl', ''),
        )


# --- ノードの登録 ---
NODE_CLASS_MAPPINGS = {
    "TIPONaturalLanguage": TIPONaturalLanguage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPONaturalLanguage": "TIPO Natural Language (Refactored)",
}
