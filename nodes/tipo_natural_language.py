import os
import re
from ..tipo_installer import install_tipo_kgen, install_llama_cpp
install_llama_cpp()
install_tipo_kgen()

import kgen.executor.tipo as tipo
from kgen.executor.tipo import tipo_single_request, tipo_runner
from kgen.formatter import seperate_tags
from kgen.logging import logger

from . import util

class TIPONaturalLanguage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tipo_model": (util.MODEL_NAME_LIST, {"default": util.MODEL_NAME_LIST[0]}),
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

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = (
        "final_nl", "overall_nl", "appearance_nl", "clothing_nl",
        "background_nl", "pose_emotion_nl", "camera_lighting_nl", "art_style_nl",
    )
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def _post_process_nl(self, nl_parts: list[str]) -> str:
        redundant_starters = [
            r'^\s*In this scene,?\s*', r'^\s*The artwork depicts\s*',
            r'^\s*This image features\s*', r'^\s*It features\s*',
            r'^\s*by the artist\s*,?\s*', r'^\s*by artist\s*,?\s*',
            r'^\s*View from\s*', r'^\s*The image shows\s*', r'^\s*The scene is\s*',
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
                    cleaned_sentence = cleaned_sentence[0].upper() + cleaned_sentence[1:]
                if cleaned_sentence.lower() not in seen_sentences:
                    seen_sentences.add(cleaned_sentence.lower())
                    final_sentences.append(cleaned_sentence)
        return " ".join(final_sentences)

    def execute(
        self,
        tipo_model: str, tags: str, ban_tags: str, width: int, height: int,
        temperature: float, top_p: float, min_p: float, top_k: int,
        overall_nl_length: str, category_nl_length: str, seed: int, device: str,
        appearance_tags: str = "", clothing_tags: str = "", background_tags: str = "",
        pose_emotion_tags: str = "", camera_lighting_tags: str = "", art_style_tags: str = "",
    ):
        util.load_tipo_model(tipo_model, device)
        
        aspect_ratio = width / height
        tipo.BAN_TAGS = [t.strip() for t in ban_tags.split(",") if t.strip()]
        if seed == -1:
            seed = int.from_bytes(os.urandom(8), 'big')

        all_tags_list = [t.strip() for t in tags.split(',') if t.strip()]
        wildcard_categories = {
            "appearance": appearance_tags, "clothing": clothing_tags,
            "background": background_tags, "pose_emotion": pose_emotion_tags,
            "camera_lighting": camera_lighting_tags, "art_style": art_style_tags,
        }
        for category_tags_str in wildcard_categories.values():
            all_tags_list.extend([t.strip() for t in category_tags_str.split(',') if t.strip()])
        all_tags_list = list(dict.fromkeys(all_tags_list))

        overall_nl = ""
        logger.info("Generating overall NL description...")
        if all_tags_list:
            org_tag_map = seperate_tags(all_tags_list)
            meta, operations, general, _ = tipo_single_request(
                org_tag_map, nl_prompt="", tag_length_target="short",
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
            category_name_for_prompt = name.replace('_', ' ')
            contextual_prompt = f"{overall_nl} In this scene, describe the {category_name_for_prompt}."
            cat_meta, cat_ops, cat_general, cat_nl_prompt = tipo_single_request(
                cat_org_tag_map, nl_prompt=contextual_prompt,
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

        all_nl_parts = [overall_nl] + list(category_nls.values())
        final_nl = self._post_process_nl(all_nl_parts)
        logger.info(f"Final integrated NL: {final_nl[:150]}...")

        return (
            final_nl, overall_nl,
            category_nls.get("appearance_nl", ""), category_nls.get("clothing_nl", ""),
            category_nls.get("background_nl", ""), category_nls.get("pose_emotion_nl", ""),
            category_nls.get("camera_lighting_nl", ""), category_nls.get("art_style_nl", ""),
        )

# Add mappings for this node
NODE_CLASS_MAPPINGS = {
    "TIPONaturalLanguage": TIPONaturalLanguage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPONaturalLanguage": "TIPO Natural Language (Contextual)",
}
