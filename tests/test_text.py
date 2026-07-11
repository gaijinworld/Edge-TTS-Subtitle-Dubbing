from __future__ import annotations

from edge_tts_subtitle_dubbing.text import clean_subtitle_text, parse_duration_str, time_str_to_seconds


class TestCleanSubtitleText:
    def test_strips_whitespace(self):
        assert clean_subtitle_text("  hello  ") == "hello"

    def test_replaces_newlines(self):
        assert clean_subtitle_text("line one\nline two") == "line one line two"

    def test_converts_ellipsis_to_empty(self):
        assert clean_subtitle_text("...") == ""

    def test_empty_string(self):
        assert clean_subtitle_text("") == ""


class TestParseDurationStr:
    def test_hh_mm_ss(self):
        assert parse_duration_str("01:30:45") == 5445.0

    def test_mm_ss(self):
        assert parse_duration_str("05:30") == 330.0

    def test_seconds(self):
        assert parse_duration_str("120.5") == 120.5

    def test_none(self):
        assert parse_duration_str(None) == 0.0

    def test_empty(self):
        assert parse_duration_str("") == 0.0

    def test_invalid(self):
        assert parse_duration_str("abc") == 0.0


class TestTimeStrToSeconds:
    def test_basic(self):
        class MockTime:
            hours = 0
            minutes = 1
            seconds = 30
            milliseconds = 500
        assert time_str_to_seconds(MockTime()) == 90.5

    def test_hours(self):
        class MockTime:
            hours = 1
            minutes = 0
            seconds = 0
            milliseconds = 0
        assert time_str_to_seconds(MockTime()) == 3600.0
