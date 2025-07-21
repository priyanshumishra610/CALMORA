"""
Tests for PanicGuard (panic-aware phrasing).
"""
import pytest
from app.core.panic_guard import PanicGuard

def test_rephrase():
    guard = PanicGuard()
    msg = guard.rephrase({"flu": 0.8})
    assert "calm" in msg or "seek medical" in msg or "Please remain calm" in msg 