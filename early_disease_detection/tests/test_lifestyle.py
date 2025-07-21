"""
Tests for LifestyleRecommender (lifestyle & health tips).
"""
import pytest
from app.core.lifestyle import LifestyleRecommender

def test_recommend():
    recommender = LifestyleRecommender()
    tips = recommender.recommend(["fever"], {"flu": 0.8})
    assert any("hydrated" in t for t in tips) 