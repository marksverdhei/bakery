"""Tests for bakery.data.load_eval_data."""

import json
import os
import tempfile

import pytest

from bakery.data import load_eval_data


class TestLoadEvalDataNone:
    def test_returns_empty_list_when_no_file(self):
        assert load_eval_data(None) == []

    def test_returns_empty_list_for_empty_string(self):
        # Empty string is falsy → same guard
        assert load_eval_data("") == []


class TestLoadEvalDataList:
    def _write(self, data):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def teardown_method(self):
        pass  # tmp files cleaned in each test

    def test_basic_question_answer_pair(self):
        data = [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 1
            question, keywords = pairs[0]
            assert question == "What is AI?"
            assert "artificial intelligence" in keywords
        finally:
            os.unlink(path)

    def test_returns_list_of_tuples(self):
        data = [{"question": "Q", "answer": "A"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert isinstance(pairs, list)
            assert isinstance(pairs[0], tuple)
            assert len(pairs[0]) == 2
        finally:
            os.unlink(path)

    def test_multiple_pairs(self):
        data = [
            {"question": "Q1", "answer": "A1"},
            {"question": "Q2", "answer": "A2"},
            {"question": "Q3", "answer": "A3"},
        ]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 3
            assert pairs[1][0] == "Q2"
        finally:
            os.unlink(path)

    def test_answer_lowercased_as_keyword(self):
        data = [{"question": "Who invented Python?", "answer": "Guido van Rossum"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            _, keywords = pairs[0]
            assert "guido van rossum" in keywords
        finally:
            os.unlink(path)

    def test_answer_list_becomes_multiple_keywords(self):
        data = [{"question": "Name a color", "answer": ["Red", "Blue", "Green"]}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            _, keywords = pairs[0]
            assert "red" in keywords
            assert "blue" in keywords
            assert "green" in keywords
        finally:
            os.unlink(path)

    def test_uses_input_key_as_question(self):
        data = [{"input": "What is 2+2?", "answer": "4"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert pairs[0][0] == "What is 2+2?"
        finally:
            os.unlink(path)

    def test_uses_expected_key_as_answer(self):
        data = [{"question": "Capital of France?", "expected": "Paris"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            _, keywords = pairs[0]
            assert "paris" in keywords
        finally:
            os.unlink(path)

    def test_uses_target_key_as_answer(self):
        data = [{"question": "Largest planet?", "target": "Jupiter"}]
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            _, keywords = pairs[0]
            assert "jupiter" in keywords
        finally:
            os.unlink(path)

    def test_nested_evaluation_samples_key(self):
        data = {"evaluation_samples": [{"question": "Q", "answer": "A"}]}
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 1
            assert pairs[0][0] == "Q"
        finally:
            os.unlink(path)

    def test_nested_test_samples_key(self):
        data = {"test_samples": [{"question": "Q", "answer": "A"}]}
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 1
        finally:
            os.unlink(path)

    def test_nested_eval_key(self):
        data = {"eval": [{"question": "Q", "answer": "A"}]}
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 1
        finally:
            os.unlink(path)

    def test_nested_qa_pairs_key(self):
        data = {"qa_pairs": [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]}
        path = self._write(data)
        try:
            pairs = load_eval_data(path)
            assert len(pairs) == 2
        finally:
            os.unlink(path)
