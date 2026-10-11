"""
generation/google.py
Google provided model implementation for LLM response generation using retrieved documents and chunks.
"""

import os

from google import genai
from google.genai import types

from generation.base import BaseGenerator


class GeminiGenerator(BaseGenerator):
    def __init__(self, model: str = "gemini-3.8-flash", model_config: types.GenerateContentConfig = None):
        client = genai.Client()
        super().__init__(client, model)
        if model_config is None:
            self.config = types.GenerateContentConfig(
                system_instruction="You are a helpful assistant that provides accurate and concise answers based on the provided context.",
                temperature=0.0
            )
        else:
            self.config = model_config

    def generate(self, query, scored_chunks):
        context_str = "\n\n".join([f"Context chunk {i + 1}:\n{chunk.chunk.content}\n" for i, chunk in enumerate(scored_chunks)])
        prompt = f"""
Based on the following context, answer the question carefully. If the context does not contain enough information, say "I don't know".
{context_str}

Question: {query}
Answer:
"""
        response = self.client.models.generate_content(
            model=self.model,
            config=self.config,
            contents=prompt
        )

        return response.text