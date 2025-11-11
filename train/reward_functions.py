import json
import re
import jsonlines
from typing import Dict, Any, List, Tuple

JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")

def _safe_last_json(text: str) -> Dict[str, Any]:
    """Extract last JSON object from a text blob. Returns {} if parsing fails."""
    if not text:
        return {}
    matches = list(JSON_OBJ_RE.finditer(text))
    for m in reversed(matches):
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def build_reward_mapping(cleaned_path: str, eval_path: str) -> Dict[str, Dict[str, Any]]:
    """Builds a mapping keyed by prompt string -> metadata used by reward_fn.
    metadata includes teacher_action, student_action, scores, and teacher add_wall_note if parsable.
    """
    cleaned_rows = []
    with jsonlines.open(cleaned_path) as r:
        for row in r:
            cleaned_rows.append(row)
    eval_rows = []
    with jsonlines.open(eval_path) as r:
        for row in r:
            eval_rows.append(row)
    if len(cleaned_rows) != len(eval_rows):
        raise ValueError(f"Length mismatch: {len(cleaned_rows)} vs {len(eval_rows)}")

    mapping: Dict[str, Dict[str, Any]] = {}
    for cleaned, ev in zip(cleaned_rows, eval_rows):
        prompt = cleaned["prompt"]
        teacher_res = cleaned.get("teacher_response", "")
        student_res = cleaned.get("student_response", "")
        teacher_obj = _safe_last_json(teacher_res)
        student_obj = _safe_last_json(student_res)

        mapping[prompt] = {
            "teacher_action": str(teacher_obj.get("action_to_take", "")),
            "teacher_add_wall_note": bool(teacher_obj.get("add_wall_note", False)) if isinstance(teacher_obj.get("add_wall_note"), bool) else False,
            "student_action": str(student_obj.get("action_to_take", "")),
            "student_add_wall_note": bool(student_obj.get("add_wall_note", False)) if isinstance(student_obj.get("add_wall_note"), bool) else False,
            "teacher_score": float(ev.get("Teacher_Score", 0.0)),
            "student_score": float(ev.get("Student_Score", 0.0)),
        }
    return mapping


def make_reward_fn(mapping: Dict[str, Dict[str, Any]]):
    """GRPO-compatible reward function factory.

    The callable signature matches TRL's GRPO expectations: reward_fn(samples: List[str], prompts: List[str]) -> List[float]
    """
    def reward_fn(samples: List[str], prompts: List[str]) -> List[float]:
        rewards: List[float] = []
        for text, prompt in zip(samples, prompts):
            meta = mapping.get(prompt)
            if meta is None:
                rewards.append(0.0)
                continue
            obj = _safe_last_json(text)
            if not obj:
                rewards.append(-0.1)
                continue
            action = str(obj.get("action_to_take", ""))
            add_note = obj.get("add_wall_note", False)
            r = 0.0
            # Exact teacher match gets teacher_score
            if action == meta["teacher_action"]:
                r = meta["teacher_score"]
                # small bonus if wall_note matches teacher
                if isinstance(add_note, bool) and add_note == meta["teacher_add_wall_note"]:
                    r += 0.05
            # Student match gets student_score
            elif action == meta["student_action"]:
                r = meta["student_score"]
            else:
                # weak reward for valid JSON but wrong action
                r = 0.0
            # Bound rewards to a reasonable range
            if r < -1.0:
                r = -1.0
            if r > 1.0:
                r = 1.0
            rewards.append(float(r))
        return rewards
    return reward_fn
