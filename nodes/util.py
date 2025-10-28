import os
import re
from pathlib import Path

import folder_paths
import kgen.models as models
import kgen.executor.tipo as tipo

# --- Global State ---
current_model = None

# --- Model Management ---
models.model_dir = Path(folder_paths.models_dir) / "kgen"
os.makedirs(models.model_dir, exist_ok=True)

# Generate model list once
model_list = tipo.models.tipo_model_list
MODEL_NAME_LIST = [
    f"{model_name} | {file}".strip("_")
    for model_name, ggufs in models.tipo_model_list
    for file in ggufs
] + [i[0] for i in models.tipo_model_list]


def load_tipo_model(tipo_model: str, device: str):
    """
    Loads the specified TIPO model if it's not already loaded.
    Handles downloading GGUF models as needed.
    """
    global current_model
    if (tipo_model, device) == current_model:
        return

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
    print(f"TIPO model loaded: {tipo_model} on {device}")


# --- Prompt Parsing Utilities ---
attn_syntax = (
    r"\\\(|" r"\\\)|" r"\\\[|" r"\\]|" r"\\\\|" r"\\|"
    r"\(|" r"\[|" r":\s*([+-]?[\.\d]+)\s*\)|" r"\)|" r"]|"
    r"[^\\()\[\]:]+|" r":"
)
re_attention = re.compile(attn_syntax, re.X)
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
        multiply_range(pos, square_brackets_multiplier)

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
    """
    Applies strength values to the tags in the tag_map.
    """
    for cate in tag_map.keys():
        new_list = []
        if isinstance(tag_map[cate], str):
            prompt = tag_map[cate]
            # Check if all parts for strength modification exist in the prompt
            if all(part in prompt for part, strength in strength_map_nl):
                new_prompt = ""
                for part, strength in strength_map_nl:
                    before, prompt = prompt.split(part, 1)
                    new_prompt += before.replace("(", "\\(").replace(")", "\\)")
                    part_escaped = part.replace("(", "\\(").replace(")", "\\)")
                    new_prompt += f"({part_escaped}:{strength})"
                new_prompt += prompt
                tag_map[cate] = new_prompt
            continue
        
        if isinstance(tag_map[cate], list):
            for org_tag in tag_map[cate]:
                tag = org_tag.replace("(", "\\(").replace(")", "\\)")
                if org_tag in strength_map:
                    new_list.append(f"({tag}:{strength_map[org_tag]})")
                else:
                    new_list.append(tag)
            tag_map[cate] = new_list
            
    return tag_map


# --- Ban フィルタ（最終整形での再混入を防ぐ共通ユーティリティ） ---
_normalize_space_underscore = re.compile(r"[\s_]+")
_weight_wrap = re.compile(r"^[\(\[]?(.+?)(?::[0-9.\-]+)?[\)\]]?$")

def normalize_tag(text: str) -> str:
    """タグ比較用の正規化。重み (tag:1.2) や括弧を外し、小文字化し、空白/アンダースコアを単一空白に統一。"""
    if not text:
        return ""
    core = _weight_wrap.sub(r"\1", text.strip())
    core = _normalize_space_underscore.sub(" ", core.lower())
    return core.strip()


def _expand_ban_forms(bans: list[str]) -> set[str]:
    """最低限の同義形を補完（単純複数形）。例: 'no human' -> {'no human', 'no humans'}"""
    out: set[str] = set()
    for b in bans:
        n = normalize_tag(b)
        if not n:
            continue
        out.add(n)
        if not n.endswith("s"):
            out.add(n + "s")
    return out


def is_banned(tag: str, ban_list: list[str]) -> bool:
    if not tag:
        return False
    ntag = normalize_tag(tag)
    ban_norm = _expand_ban_forms(ban_list)
    return ntag in ban_norm


def filter_banned_tags(tags: list[str], ban_list: list[str]) -> list[str]:
    """タグ配列から ban に該当する要素と空要素を除外。正規化後の重複も排除。"""
    seen = set()
    filtered: list[str] = []
    for t in tags:
        if not t:
            continue
        if is_banned(t, ban_list):
            continue
        key = normalize_tag(t)
        if key in seen:
            continue
        seen.add(key)
        filtered.append(t)
    return filtered
