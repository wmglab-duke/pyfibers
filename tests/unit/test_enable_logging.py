"""Tests for pyfibers.enable_logging.

The copyrights of this software are owned by Duke University.
See LICENSE for licensing instructions.
Source code: https://github.com/wmglab-duke/pyfibers
"""

from __future__ import annotations

import io
import logging
import sys

import pyfibers


def _restore_logger(logger, handlers, level):
    logger.handlers[:] = handlers
    logger.setLevel(level)


def test_enable_logging_adds_streamhandler():
    logger = logging.getLogger("pyfibers")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    try:
        pyfibers.enable_logging()
        stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler)]
        assert len(stream_handlers) >= 1
        assert logger.level == logging.INFO
    finally:
        _restore_logger(logger, original_handlers, original_level)


def test_enable_logging_idempotent():
    logger = logging.getLogger("pyfibers")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    try:
        pyfibers.enable_logging()
        pyfibers.enable_logging()
        stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
        assert len(stream_handlers) == 1
    finally:
        _restore_logger(logger, original_handlers, original_level)


def test_enable_logging_custom_stream():
    logger = logging.getLogger("pyfibers")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    buf = io.StringIO()
    try:
        pyfibers.enable_logging(stream=buf)
        logger.info("hello-from-test")
        assert "hello-from-test" in buf.getvalue()
    finally:
        _restore_logger(logger, original_handlers, original_level)


def test_enable_logging_docs_build_stdout(monkeypatch):
    logger = logging.getLogger("pyfibers")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    monkeypatch.setenv("PYFIBERS_DOCS_BUILD", "1")
    try:
        pyfibers.enable_logging()
        stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
        assert stream_handlers[0].stream is sys.stdout
    finally:
        _restore_logger(logger, original_handlers, original_level)


def test_enable_logging_custom_format_and_level():
    logger = logging.getLogger("pyfibers")
    original_handlers = list(logger.handlers)
    original_level = logger.level
    buf = io.StringIO()
    try:
        pyfibers.enable_logging(level=logging.WARNING, format_string="%(levelname)s-%(message)s", stream=buf)
        logger.warning("fmt-check")
        logger.info("should-not-appear")
        assert "WARNING-fmt-check" in buf.getvalue()
        assert "should-not-appear" not in buf.getvalue()
    finally:
        _restore_logger(logger, original_handlers, original_level)
