import csv
import json
import tempfile
from pathlib import Path
import unittest

from tools.clean_q7_nl import clean_csv, clean_q7_text


class CleanQ7TextTests(unittest.TestCase):
    def test_removes_standalone_fillers_meta_talk_and_padding(self):
        source = "嗯，我支持保留人工通道。好了我又要开始凑字数了……………………"

        result = clean_q7_text(source)

        self.assertEqual(result.text, "我支持保留人工通道。")
        self.assertIn("凑字数", "".join(result.removed_spans))

    def test_preserves_substantive_meta_k_and_ambiguous_oral_words(self):
        source = "我没做过这种设计，其实这就是经验边界吧。"

        self.assertEqual(clean_q7_text(source).text, source)

    def test_removes_only_meta_clause_from_mixed_sentence(self):
        source = "这个问题不用写这么多字，我认为必须保留现金支付。"

        self.assertEqual(clean_q7_text(source).text, "我认为必须保留现金支付。")

    def test_empty_result_falls_back_to_original(self):
        source = "嗯。"

        result = clean_q7_text(source)

        self.assertEqual(result.text, source)
        self.assertTrue(result.used_fallback)

    def test_preserves_unrelated_spacing_when_no_target_is_removed(self):
        source = "商业要求为先，  这部分空格保留。"

        self.assertEqual(clean_q7_text(source).text, source)

    def test_removes_embedded_fillers_without_punctuation_artifacts(self):
        source = "这是。呃，方案，而且我会呃先调研。嗯并且保留人工。"

        self.assertEqual(
            clean_q7_text(source).text,
            "这是。方案，而且我会先调研。并且保留人工。",
        )

    def test_removes_hesitation_commas_but_keeps_existing_clause_comma(self):
        source = "我会呃，先调研，嗯，再保留人工。"

        self.assertEqual(clean_q7_text(source).text, "我会先调研，再保留人工。")

    def test_removes_romanized_hesitation_at_start(self):
        source = "emmmmmmmm第一反应是保留人工。"

        self.assertEqual(clean_q7_text(source).text, "第一反应是保留人工。")

    def test_removes_audio_and_word_count_meta_talk(self):
        source = (
            "the audio duration is 65.519000 seconds, longer than 60 seconds. "
            "啊，刚刚那个超过60秒了，首先我会调研。"
            "好的，那就随便说一点吧，说满300字。"
        )

        self.assertEqual(clean_q7_text(source).text, "首先我会调研。")

    def test_removes_parenthesized_padding_meta_talk(self):
        source = "我支持保留人工。（没话说了原谅这句用来凑字数吧"

        self.assertEqual(clean_q7_text(source).text, "我支持保留人工。")

    def test_removes_cross_answer_reference_only(self):
        source = "如上个问题所说，这个项目需要快速落地。"

        self.assertEqual(clean_q7_text(source).text, "这个项目需要快速落地。")

    def test_removes_cross_answer_reference_inside_reasoning(self):
        source = "因为就像我上面说的，诱导问题没有那么严重。"

        self.assertEqual(clean_q7_text(source).text, "因为诱导问题没有那么严重。")

    def test_removes_transcription_prefix_and_trailing_meta_talk(self):
        source = "我会先调研。Barry, 怎么一定要输满300字啊？嗯就这样"

        self.assertEqual(clean_q7_text(source).text, "我会先调研。")

    def test_removes_direct_task_complaint_tail(self):
        source = (
            "成交额可能有，但绝对亏本。"
            "所以别扯这些没用的好吗，你的问题假设太过了且特别不专业，浪费我时间。"
        )

        self.assertEqual(clean_q7_text(source).text, "成交额可能有，但绝对亏本。")

    def test_removes_no_more_to_say_without_dropping_following_content(self):
        source = "方案可以成立。还有啥呀，我感觉我也没话说了。我会做用户测试。"

        self.assertEqual(
            clean_q7_text(source).text,
            "方案可以成立。我会做用户测试。",
        )

    def test_removes_declared_padding_tail(self):
        source = (
            "最后结合经验做决策。大概就是这样，没有什么其他想法了，"
            "接下来的这些文字其实都是在凑数，为什么一定要300个字啊，"
            "200多个字已经是我的极限了呀呀呀"
        )

        self.assertEqual(clean_q7_text(source).text, "最后结合经验做决策。")

    def test_preserves_experience_boundary_around_word_count_meta(self):
        source = (
            "因为我对整体系统没有太多了解，所以方案非常浅显，"
            "没有办法能写出那么多字数，我只能根据经验提出建议。"
        )

        self.assertEqual(
            clean_q7_text(source).text,
            "因为我对整体系统没有太多了解，所以方案非常浅显，"
            "我只能根据经验提出建议。",
        )


class CleanQ7CsvTests(unittest.TestCase):
    fieldnames = [
        "user_id",
        "question_id",
        "answer_value",
        "text_supplement",
        "duration_seconds",
        "device_type",
        "submitted_at",
    ]

    def _write_rows(self, path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_changes_only_q7_text_and_canonicalizes_segment_order(self):
        q1 = {
            "user_id": "U1",
            "question_id": "Q1",
            "answer_value": "A",
            "text_supplement": "原样保留",
            "duration_seconds": "1",
            "device_type": "desktop",
            "submitted_at": "2026-01-01",
        }
        payload = {
            "seg3": {"text": "嗯，保留现金支付。", "method": "text"},
            "seg2": {"text": "我没做过这种设计。", "method": "text"},
            "seg1": {
                "text": "支持人工通道。凑字数……………………",
                "method": "text",
            },
        }
        q7 = {
            "user_id": "U1",
            "question_id": "Q7",
            "answer_value": json.dumps(payload, ensure_ascii=False),
            "text_supplement": "",
            "duration_seconds": "1",
            "device_type": "desktop",
            "submitted_at": "2026-01-01",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "output.csv"
            self._write_rows(source, [q1, q7])

            report = clean_csv(source, output)
            rows = self._read_rows(output)

        self.assertEqual(rows[0], q1)
        self.assertEqual(report.total_rows, 2)
        self.assertEqual(report.q7_rows, 1)
        cleaned = json.loads(rows[1]["answer_value"])
        self.assertEqual(list(cleaned), ["seg1", "seg2", "seg3"])
        self.assertEqual(cleaned["seg1"]["text"], "支持人工通道。")
        self.assertEqual(cleaned["seg2"], payload["seg2"])
        self.assertEqual(cleaned["seg3"]["text"], "保留现金支付。")
        self.assertEqual(cleaned["seg3"]["method"], "text")
        for key in self.fieldnames:
            if key != "answer_value":
                self.assertEqual(rows[1][key], q7[key])

    def test_rejects_invalid_q7_json(self):
        bad_q7 = dict.fromkeys(self.fieldnames, "")
        bad_q7.update(
            {"user_id": "BROKEN", "question_id": "Q7", "answer_value": "{"}
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            output = Path(directory) / "output.csv"
            self._write_rows(source, [bad_q7])

            with self.assertRaisesRegex(ValueError, "user_id=BROKEN"):
                clean_csv(source, output)


if __name__ == "__main__":
    unittest.main()
