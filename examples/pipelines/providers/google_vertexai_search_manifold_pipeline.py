"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: google-cloud-aiplatform
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
import uuid
from typing import Iterator, List, Union, Optional, Dict, Any
from pydantic import BaseModel, Field
import logging
import time

import vertexai
from pydantic import BaseModel, Field
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerationResponse,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
    Tool,
    grounding,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Pipeline:
    """Google GenAI pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_PROJECT_ID: str = ""
        GOOGLE_CLOUD_REGION: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)

    def __init__(self):
        self.type = "manifold"
        self.name = "vertexai: "
        self.html_filename = "grounding_with_google_search.html"

        self.valves = self.Valves(
            **{
                "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
                "GOOGLE_CLOUD_REGION": os.getenv("GOOGLE_CLOUD_REGION", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.pipelines = [
            {"id": "gemini-1.5-flash-002", "name": "Gemini 1.5 Flash 002-Search"},
            {"id": "gemini-1.5-pro-002", "name": "Gemini 1.5 Pro 002-Search"},
            {"id": "gemini-1.5-flash", "name": "Gemini 1.5 Flash-Search"},
            {"id": "gemini-1.5-pro", "name": "Gemini 1.5 Pro-Search"},
            {"id": "gemini-flash-experimental", "name": "Gemini 1.5 Flash Experimental-Search"},
            {"id": "gemini-pro-experimental", "name": "Gemini 1.5 Pro Experimental-Search"},
            #{"id": "gemini-2.0-flash-001", "name": "Gemini 2.0 Flash 001-Search"},
            #{"id": "gemini-2.0-pro-exp-02-05", "name": "Gemini 2.0 Pro Exp 0205-Search"},
            #{"id": "gemini-2.0-flash-thinking-exp-01-21", "name": "Gemini 2.0 Flash Thinking Exp 0121-Search"},
        ]

        self.tool = Tool.from_google_search_retrieval(grounding.GoogleSearchRetrieval())

    async def on_startup(self) -> None:
        """This function is called when the server is started."""

        print(f"on_startup:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    async def on_shutdown(self) -> None:
        """This function is called when the server is stopped."""
        print(f"on_shutdown:{__name__}")

    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        vertexai.init(
            project=self.valves.GOOGLE_PROJECT_ID,
            location=self.valves.GOOGLE_CLOUD_REGION,
        )

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict,
    ) -> Union[str, Iterator]:
        try:
            if not model_id.startswith("gemini-"):
                return f"Error: Invalid model name format: {model_id}"

            print(f"Pipe function called for model: {model_id}")
            print(f"Stream mode: {body.get('stream', False)}")

            system_message = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            model = GenerativeModel(
                model_name=model_id,
                system_instruction=system_message,
            )

            if body.get("title", False):  # If chat title generation is requested
                contents = [Content(role="user", parts=[Part.from_text(user_message)])]
            else:
                contents = self.build_conversation_history(messages)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.7),
                top_p=body.get("top_p", 0.9),
                top_k=body.get("top_k", 40),
                max_output_tokens=body.get("max_tokens", 8192),
                stop_sequences=body.get("stop", []),
            )

            if self.valves.USE_PERMISSIVE_SAFETY:
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                }
            else:
                safety_settings = body.get("safety_settings")

            response = model.generate_content(
                contents,
                stream=body.get("stream", False),
                generation_config=generation_config,
                safety_settings=safety_settings,
                tools=[self.tool],
            )

            if body.get("stream", False):
                return self.stream_response(response)
            else:
                return response.text

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def print_grounding_chunk(self, chunk):
        """Prints Gemini response chunk with grounding citations."""
        grounding_metadata = chunk.candidates[0].grounding_metadata

        markdown_text = ""

        if grounding_metadata.web_search_queries:
            markdown_text += (
                f"\n**Web Search Queries:** {grounding_metadata.web_search_queries}\n"
            )
            #if grounding_metadata.search_entry_point:
            #    markdown_text += f"\n**Search Entry Point:**\n {grounding_metadata.search_entry_point.rendered_content}\n"

        elif grounding_metadata.retrieval_queries:
            markdown_text += (
                f"\n**Retrieval Queries:** {grounding_metadata.retrieval_queries}\n"
            )

        markdown_text += "**Grounding Sources**\n"

        for index, grounding_chunk in enumerate(
            grounding_metadata.grounding_chunks, start=1
        ):
            context = grounding_chunk.web or grounding_chunk.retrieved_context
            if not context:
                print(f"Skipping Grounding Chunk {grounding_chunk}")
                continue

            markdown_text += f"{index}. [{context.title}]({context.uri})\n"

        return markdown_text

    def stream_response(self, response):
        for chunk in response:
            if chunk.candidates[0].grounding_metadata:
                grounding_info = self.print_grounding_chunk(chunk)
                print(f"Chunk: {chunk.text}")
                print(grounding_info)
                yield chunk.text + "\n\n" + grounding_info
            elif chunk.text:
                print(f"Chunk: {chunk.text}")
                yield chunk.text

    def build_conversation_history(self, messages: List[dict]) -> List[Content]:
        contents = []

        for message in messages:
            if message["role"] == "system":
                continue

            parts = []

            if isinstance(message.get("content"), list):
                for content in message["content"]:
                    if content["type"] == "text":
                        parts.append(Part.from_text(content["text"]))
                    elif content["type"] == "image_url":
                        image_url = content["image_url"]["url"]
                        if image_url.startswith("data:image"):
                            image_data = image_url.split(",")[1]
                            parts.append(Part.from_image(image_data))
                        else:
                            parts.append(Part.from_uri(image_url))
            else:
                parts = [Part.from_text(message["content"])]

            role = "user" if message["role"] == "user" else "model"
            contents.append(Content(role=role, parts=parts))

        return contents
    