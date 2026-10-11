"""
generation/base.py
Base class for LLM response generation using retrieved documents and chunks.
"""

from abc import ABC, abstractmethod

class BaseGenerator(ABC):
    @abstractmethod
    def __init__(self, client, model):
        self.client = client
        self.model = model

    @abstractmethod
    def generate(self, query, scored_chunks):
        pass