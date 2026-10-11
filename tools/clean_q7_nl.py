"""Conservatively clean Q7 natural-language answers for annotation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re


DEFAULT_INPUT = Path(
    "/Users/wy/Desktop/KDD/processed_data/llm_rewritten_v3/"
    "task_responses_llm_rewritten.csv"
)
DEFAULT_OUTPUT = Path(
    "/Users/wy/Documents/designer_roleplay/processed_data/q7_cleaned/"
    "task_responses_llm_rewritten_q7_cleaned.csv"
)


@dataclass(frozen=True)
class CleanResult:
    text: str
    removed_spans: tuple[str, ...]
    used_fallback: bool = False


@dataclass(frozen=True)
class CleaningReport:
    total_rows: int
    q7_rows: int
    q7_segments: int
    changed_segments: int
    fallback_segments: int
    changed_examples: tuple[tuple[str, str, str, str], ...]


META_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"the audio duration is [\d.]+ seconds, longer than 60 seconds\.\s*",
        r"刚刚那个超过60秒了[，,。！？?\s]*",
        r"(?:Barry,\s*)?(?:怎么)?(?:一定要|还要|非要)?(?:输|写|说)(?:满|够)\s*\d+\s*字[啊额呢吧！？?。]*",
        r"(?:好的?[，,]?\s*)?那就随便说一点吧[，,]?\s*(?:说满\d+字)?[。！？?]*",
        r"我凑不到数了[，,]?",
        r"随便说点吧[，,]?",
        r"[（(][^）)\n]*(?:凑|水|够)字数[^）)\n]*[）)]?",
        r"具体原因我在下一题讲[。！？?]*",
        r"(?:如|就像)(?:我)?(?:上个|上一|上面|前面)(?:问题|题)?所说[，,]?",
        r"就像我上面说的[，,]?",
        r"和之前的问题一样[，,]?",
        r"我已经不知道该说什么[。！？?]*",
        r"还有啥呀[，,]?我感觉我也没话说了[。！？?]*",
        r"大概就是这样[，,]?没有什么其他想法了[，,]?接下来的这些文字其实都是在凑数[，,]?为什么一定要\d+个字[啊呀呢吧]*[，,]?\d+多个字已经是我的极限了呀+",
        r"没有办法能?写出?(?:那么多|这么多)字数[，,]?",
        r"我觉得我的想法（请看第二题我的答案）对于第一个和第三个问题完全不需要这么多的字数要求[。！？?]*",
        r"这个问题只问我第一反应[，,]?",
        r"应该用不了这么多字来写一个第一反应吧[，,]?",
        r"你应该也看出来了我现在在努力的凑字数[，,]?",
        r"啊?这个问题我感觉真没必要用这么多字来说诶[。！？?]*",
        r"这个提问的背景本身就存在很大的漏洞[，,]?",
        r"如果只是单纯的假设我觉得很不专业你的这个问题[，,]?(?:并且)?",
        r"这个问题毫无(?:意|疑)义好吗[？?]?",
        r"所以别扯这些没用的好吗[，,]?你的问题假设太过了且特别不专业[，,]?浪费我时间[。！？?]*",
        r"(?:嗯)?就这样[。！？?]*\s*$",
        r"(?:好了?)?我?(?:又要)?(?:开始)?(?:努力)?(?:凑|水)(?:一下)?字数(?:了)?[啊吧呢]*",
        r"(?:怎么|为什么)?(?:还|会)?有字数要求[啊额呢吧]*",
        r"(?:两百|200|这么多|那么多|好多)(?:个)?字(?:也)?(?:太|真)?(?:多|长)(?:了)?[啊吧呢]*",
        r"字数要求(?:真的|也)?(?:好多|太多|很长)[啊吧呢]*",
        r"(?:这个|第[一二三]?个)?问题(?:完全)?(?:不需要|不用|用不了|没必要)(?:写)?(?:这么多|那么多)字(?:要求)?[啊吧呢]*",
        r"我本来只打算写第一句话(?:的)?",
        r"(?:很多想法)?我(?:已经)?在(?:第二个|第二|上一|上个|前面)(?:问题|题)(?:的回答)?里写清楚了",
        r"(?:请)?看(?:我)?(?:上一个|上个|前一个|第二个|第二|前面)(?:问题|题)(?:中|的)?(?:我的)?(?:回答|答案)?",
        r"这就是我的第一反应了",
    )
)
HESITATION_RE = re.compile(r"(?:嗯+|呃+)(?:[ \t]*[，、])?")
PARTICLE_RE = re.compile(r"(?:啊+|诶+|哎+)")
BOUNDED_E_FILLER_RE = re.compile(
    r"(^|[\s，。！？、；：])额+(?=$|[\s，。！？、；：])"
)
ROMAN_FILLER_RE = re.compile(
    r"(^|[\s，。！？、；：])e?m{2,}", re.IGNORECASE
)
PADDING_RE = re.compile(r"(?:[.．·…]{6,}|[_—-]{8,})")
EXCESS_BLANK_RE = re.compile(r"\n(?:[ \t]*\n){3,}")


def _delete_matches(
    text: str, pattern: re.Pattern[str], removed: list[str]
) -> str:
    def replace(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return ""

    return pattern.sub(replace, text)


def _delete_fillers(text: str, removed: list[str]) -> str:
    text = _delete_matches(text, HESITATION_RE, removed)
    text = _delete_matches(text, PARTICLE_RE, removed)

    def replace(match: re.Match[str]) -> str:
        removed.append(match.group(0)[len(match.group(1)) :])
        return match.group(1)

    text = ROMAN_FILLER_RE.sub(replace, text)
    return BOUNDED_E_FILLER_RE.sub(replace, text)


def _compress_excess_blank_lines(text: str, removed: list[str]) -> str:
    def replace(match: re.Match[str]) -> str:
        removed.append(match.group(0))
        return "\n\n"

    return EXCESS_BLANK_RE.sub(replace, text)


def _normalize_after_deletion(text: str) -> str:
    text = re.sub(r"([。！？；：])\s*，", r"\1", text)
    text = re.sub(r"，\s*([。！？；：])", r"\1", text)
    text = re.sub(r"，{2,}", "，", text)
    text = re.sub(r"。{2,}", "。", text)
    text = re.sub(r"(?m)^[ \t]*[，。；：、]+[ \t]*", "", text)
    return text.strip(" \t\n，；：、")


def clean_q7_text(text: str) -> CleanResult:
    """Remove only Q7 speech fillers, response meta-talk, and padding."""

    original = text
    cleaned = text
    removed: list[str] = []
    for pattern in META_PATTERNS:
        cleaned = _delete_matches(cleaned, pattern, removed)
    cleaned = _delete_matches(cleaned, PADDING_RE, removed)
    cleaned = _compress_excess_blank_lines(cleaned, removed)
    cleaned = _delete_fillers(cleaned, removed)
    if not removed:
        return CleanResult(original, ())
    cleaned = _normalize_after_deletion(cleaned)
    if not cleaned:
        return CleanResult(original, tuple(removed), used_fallback=True)
    return CleanResult(cleaned, tuple(removed))


def clean_csv(input_path: Path, output_path: Path) -> CleaningReport:
    """Write a new CSV after cleaning only Q7 segment text values."""

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError("Input CSV has no header")
        rows = list(reader)

    changed_examples: list[tuple[str, str, str, str]] = []
    q7_rows = 0
    q7_segments = 0
    changed_segments = 0
    fallback_segments = 0
    for row in rows:
        if row.get("question_id") != "Q7":
            continue
        q7_rows += 1
        user_id = row.get("user_id", "")
        try:
            payload = json.loads(row["answer_value"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid Q7 JSON for user_id={user_id}") from exc

        canonical: dict[str, dict[str, object]] = {}
        for segment_id in ("seg1", "seg2", "seg3"):
            segment = payload.get(segment_id)
            if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                raise ValueError(f"Missing Q7 {segment_id}.text for user_id={user_id}")
            q7_segments += 1
            before = segment["text"]
            result = clean_q7_text(before)
            updated = dict(segment)
            updated["text"] = result.text
            canonical[segment_id] = updated
            if result.used_fallback:
                fallback_segments += 1
            if result.text != before:
                changed_segments += 1
                changed_examples.append(
                    (user_id, segment_id, before, result.text)
                )
        row["answer_value"] = json.dumps(
            canonical, ensure_ascii=False, separators=(",", ":")
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    return CleaningReport(
        total_rows=len(rows),
        q7_rows=q7_rows,
        q7_segments=q7_segments,
        changed_segments=changed_segments,
        fallback_segments=fallback_segments,
        changed_examples=tuple(changed_examples),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservatively clean Q7 NL text")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = clean_csv(args.input, args.output)
    print(
        f"total_rows={report.total_rows} q7_rows={report.q7_rows} "
        f"q7_segments={report.q7_segments} "
        f"changed_segments={report.changed_segments} "
        f"fallback_segments={report.fallback_segments}"
    )
    for user_id, segment_id, before, after in report.changed_examples:
        print(
            json.dumps(
                {
                    "user_id": user_id,
                    "segment_id": segment_id,
                    "before": before,
                    "after": after,
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
