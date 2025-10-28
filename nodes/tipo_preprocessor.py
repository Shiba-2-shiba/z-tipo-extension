# -*- coding: utf-8 -*-

class TIPOPreprocessor:
    """
    全てのタグとプロンプト入力を一つの辞書オブジェクトにまとめるためのノード。
    これにより、後続のノードのUIが簡素化され、ワークフローが整理されます。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tags": ("STRING", {"default": "", "multiline": True, "placeholder": "キャラクター、著作権、一般タグなど..."}),
                "nl_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "自然言語での全体的な指示..."}),
                "ban_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "除外したいタグをカンマ区切りで入力..."}),
            },
            # ★ 変更点: 不要なカテゴリを削除
            "optional": {
                "appearance_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "外見に関するタグ..."}),
                "clothing_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "服装に関するタグ..."}),
                "background_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "背景に関するタグ..."}),
                "pose_emotion_tags": ("STRING", {"default": "", "multiline": True, "placeholder": "ポーズや表情に関するタグ..."}),
            }
        }

    RETURN_TYPES = ("TIPO_PROMPTS",)
    RETURN_NAMES = ("tipo_prompts",)
    FUNCTION = "execute"
    CATEGORY = "utils/promptgen"

    def execute(self, **kwargs):
        prompt_data = {
            "main": {
                "tags": kwargs.get("tags", ""),
                "nl_prompt": kwargs.get("nl_prompt", ""),
                "ban_tags": kwargs.get("ban_tags", ""),
            },
            # ★ 変更点: 不要なカテゴリを削除
            "categories": {
                "appearance": kwargs.get("appearance_tags", ""),
                "clothing": kwargs.get("clothing_tags", ""),
                "background": kwargs.get("background_tags", ""),
                "pose_emotion": kwargs.get("pose_emotion_tags", ""),
            }
        }
        return (prompt_data,)

# --- ノードのマッピング定義 ---
NODE_CLASS_MAPPINGS = {
    "TIPOPreprocessor": TIPOPreprocessor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TIPOPreprocessor": "TIPO Preprocessor",
}
