"""Tests for scripts.run_demo module — _TeeStream and formatting helpers."""

from __future__ import annotations

import io
from io import TextIOBase

import pytest

from scripts.run_demo import _TeeStream, _compact_rows, _format_evidence, _hr, _truncate


class TestTeeStream:
    """Tests for _TeeStream dual-output stream."""

    def _make_streams(self) -> tuple[io.StringIO, io.StringIO, _TeeStream]:
        stream = io.StringIO()
        log = io.StringIO()
        tee = _TeeStream(stream, log)
        return stream, log, tee

    def test_write_goes_to_both(self) -> None:
        stream, log, tee = self._make_streams()
        tee.write("hello")
        assert stream.getvalue() == "hello"
        assert log.getvalue() == "hello"

    def test_write_returns_length(self) -> None:
        _, _, tee = self._make_streams()
        assert tee.write("abcde") == 5
        assert tee.write("") == 0

    def test_multiple_writes_accumulate(self) -> None:
        stream, log, tee = self._make_streams()
        tee.write("line1\n")
        tee.write("line2\n")
        expected = "line1\nline2\n"
        assert stream.getvalue() == expected
        assert log.getvalue() == expected

    def test_flush_does_not_raise(self) -> None:
        _, _, tee = self._make_streams()
        tee.write("data")
        tee.flush()  # should not raise

    def test_writable_returns_true(self) -> None:
        _, _, tee = self._make_streams()
        assert tee.writable() is True

    def test_readable_returns_false(self) -> None:
        _, _, tee = self._make_streams()
        assert tee.readable() is False

    def test_encoding_property_defaults(self) -> None:
        _, _, tee = self._make_streams()
        # StringIO has no 'encoding' attr, so fallback to 'utf-8'
        assert tee.encoding == "utf-8"

    def test_encoding_from_stream(self) -> None:
        """If the underlying stream has an encoding attribute, use it."""

        class FakeStream(TextIOBase):
            encoding = "ascii"  # type: ignore[assignment]

            def write(self, s: str) -> int:
                return len(s)

        log = io.StringIO()
        tee = _TeeStream(FakeStream(), log)
        assert tee.encoding == "ascii"

    def test_is_text_io_base_subclass(self) -> None:
        _, _, tee = self._make_streams()
        assert isinstance(tee, TextIOBase)

    def test_unicode_content(self) -> None:
        stream, log, tee = self._make_streams()
        text = "中文测试 🚀 émojis"
        tee.write(text)
        assert stream.getvalue() == text
        assert log.getvalue() == text


class TestFormatHelpers:
    """Tests for run_demo formatting utilities."""

    def test_hr_default(self) -> None:
        result = _hr()
        assert result == "─" * 60

    def test_hr_custom(self) -> None:
        assert _hr("═", 10) == "═" * 10

    def test_truncate_short(self) -> None:
        assert _truncate("short") == "short"

    def test_truncate_long(self) -> None:
        text = "a" * 100
        result = _truncate(text, max_len=20)
        assert len(result) == 20
        assert result.endswith("…")
        assert result == "a" * 19 + "…"

    def test_truncate_exact_boundary(self) -> None:
        text = "a" * 50
        assert _truncate(text) == text  # exactly at limit

    def test_compact_rows_single_row(self) -> None:
        items = ["A(x)", "B(y)", "C(z)"]
        rows = _compact_rows(items)
        assert len(rows) == 1
        assert rows[0] == "A(x) · B(y) · C(z)"

    def test_compact_rows_wrap(self) -> None:
        items = ["Entity" * 5 + "(type)"] * 5  # long items
        rows = _compact_rows(items, row_width=30)
        assert len(rows) > 1  # should wrap

    def test_compact_rows_max_items(self) -> None:
        items = [f"E{i}(T)" for i in range(20)]
        rows = _compact_rows(items, max_items=5)
        last = rows[-1]
        assert "共 20 个" in last

    def test_compact_rows_no_limit(self) -> None:
        items = [f"E{i}(T)" for i in range(20)]
        rows = _compact_rows(items, max_items=0)
        all_text = " · ".join(items)
        # All items should appear (possibly across rows)
        joined = " · ".join(rows)
        for item in items:
            assert item in joined


class TestFormatEvidence:
    """Tests for _format_evidence display."""

    def _make_chain(
        self,
        *,
        edges: list | None = None,
        steps: list | None = None,
    ) -> EvidenceChain:
        from src.reasoning.evidence_chain import (
            EvidenceChain,
            EvidenceEdge,
            EvidenceNode,
            EvidenceStep,
        )

        default_edges = edges if edges is not None else [
            EvidenceEdge(
                source="A", target="B", relation_type="rel1",
                confidence=0.9, hop=1,
            ),
            EvidenceEdge(
                source="B", target="C", relation_type="rel2",
                confidence=0.8, hop=2,
            ),
        ]
        default_nodes = [
            EvidenceNode(name="A", label="type", hop=0),
            EvidenceNode(name="B", label="type", hop=1),
            EvidenceNode(name="C", label="type", hop=2),
        ]
        default_steps = steps if steps is not None else [
            EvidenceStep(
                hop_number=1, action="explore",
                nodes_explored=["B"], relation_used="rel1",
                reasoning="通过 rel1 找到 B",
            ),
        ]
        return EvidenceChain(
            nodes=default_nodes,
            edges=default_edges,
            steps=default_steps,
            start_entity="A",
            end_entity="C",
            total_confidence=0.72,
        )

    def test_empty_edges(self) -> None:
        from src.reasoning.evidence_chain import EvidenceChain

        chain = EvidenceChain(
            nodes=[], edges=[], steps=[],
            start_entity="X", end_entity=None, total_confidence=0.0,
        )
        lines = _format_evidence(chain)
        assert len(lines) == 1
        assert "无推理边" in lines[0]

    def test_basic_format(self) -> None:
        chain = self._make_chain()
        lines = _format_evidence(chain, verbose=False)
        # Should have hop lines + separator + summary
        hop_lines = [l for l in lines if l.startswith("hop ")]
        assert len(hop_lines) == 2
        assert "A ──[rel1]──→ B" in hop_lines[0]
        assert "B ──[rel2]──→ C" in hop_lines[1]
        # No confidence in non-verbose
        assert "(90%)" not in hop_lines[0]

    def test_verbose_shows_confidence(self) -> None:
        chain = self._make_chain()
        lines = _format_evidence(chain, verbose=True)
        hop_lines = [l for l in lines if l.startswith("hop ")]
        assert "(90%)" in hop_lines[0]
        assert "(80%)" in hop_lines[1]

    def test_verbose_shows_reasoning_steps(self) -> None:
        chain = self._make_chain()
        lines = _format_evidence(chain, verbose=True)
        joined = "\n".join(lines)
        assert "推理步骤" in joined
        assert "通过 rel1 找到 B" in joined

    def test_summary_line(self) -> None:
        chain = self._make_chain()
        lines = _format_evidence(chain, verbose=False)
        summary = lines[-1]
        assert "节点: 3" in summary
        assert "边: 2" in summary
        assert "置信度: 72.0%" in summary
