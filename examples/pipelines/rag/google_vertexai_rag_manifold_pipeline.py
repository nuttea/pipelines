"""
title: Google GenAI (Vertex AI) Manifold Pipeline
author: Hiromasa Kakehashi
date: 2024-09-19
version: 1.0
license: MIT
description: A pipeline for generating text using Google's GenAI models in Open-WebUI.
requirements: vertexai
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
usage_instructions:
  To use Gemini with the Vertex AI API, a service account with the appropriate role (e.g., `roles/aiplatform.user`) is required.
  - For deployment on Google Cloud: Associate the service account with the deployment.
  - For use outside of Google Cloud: Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of the service account key file.
"""

import os
import uuid
from typing import List, Union, Generator, Iterator, Optional
from pprint import pprint

import vertexai
from pydantic import BaseModel, Field
from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

PROMPT_TEMPLATE = """Given the conversation between a user and a helpful assistant and some search results, create a final answer for the assistant.
The answer should use all relevant information from the search results, not introduce any additional information, and use exactly the same words as the search results when possible. 
The assistant's answer should be no more than 30 sentences.
The assistant's answer should be main topic with transaction code and follow by step-by-step instructions formatted as a bulleted list.
Each list item should start with the "-" symbol.

Build the answer specificly to the user persona:
<PERSONA>
{persona}
</PERSONA>

{context}

User Question: {search_query}

Answer in the same language of user question."""

class Pipeline:
    """Search RAG pipeline"""

    class Valves(BaseModel):
        """Options to change from the WebUI"""

        GOOGLE_PROJECT_ID: str = ""
        GOOGLE_CLOUD_REGION: str = ""
        USE_PERMISSIVE_SAFETY: bool = Field(default=False)
        LOCATION: str = ""
        ENGINE_ID: str = ""
        PERSONA: str = ""

    def __init__(self):
        self.type = "manifold"
        self.name = "vertexai: "

        self.valves = self.Valves(
            **{
                "GOOGLE_PROJECT_ID": os.getenv("GOOGLE_PROJECT_ID", ""),
                "GOOGLE_CLOUD_REGION": os.getenv("GOOGLE_CLOUD_REGION", ""),
                "USE_PERMISSIVE_SAFETY": False,
            }
        )
        self.pipelines = [
            {"id": "gemini-1.5-flash", "name": "RAG Gemini 1.5 Flash"},
            {"id": "gemini-1.5-pro", "name": "RAG Gemini 1.5 Pro"},
        ]

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

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        # Check for presence of required keys and generate chat_id if missing
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")
        return body

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
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
                return self.title_generation(body, user_message)
            else:
                contents = self.build_conversation_history(messages)

            generation_config = GenerationConfig(
                temperature=body.get("temperature", 0.1),
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

            # RAG on last message before generate_content
            results_chunks = self.retriever(
                self.valves.GOOGLE_PROJECT_ID,
                self.valves.LOCATION,
                self.valves.ENGINE_ID,
                user_message,
            )

            context, citations_list = self.parse_results(results_chunks.results)

            prompt = PROMPT_TEMPLATE.format(
                persona=self.valves.PERSONA,
                context=context,
                search_query=user_message,
            )

            contents[-1] = Content(role="user", parts=[Part.from_text(prompt)])

            response = model.generate_content(
                contents,
                stream=body.get("stream", False),
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if body.get("stream", False):
                return self.stream_response(response, citations_list)
            else:
                return response.text + "\n" + citations_list

        except Exception as e:
            print(f"Error generating content: {e}")
            return f"An error occurred: {str(e)}"

    def stream_response(self, response, citations_list):
        for chunk in response:
            if chunk.text:
                print(f"Chunk: {chunk.text}")
                yield chunk.text
        yield "\n\n" + citations_list

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
    
    def title_generation(self, body, user_message):
        # Title Generation
        print("Title Generation Request")
        model = GenerativeModel(
            model_name="gemini-1.5-flash",
        )

        generation_config = GenerationConfig(
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            top_k=body.get("top_k", 40),
            max_output_tokens=body.get("max_tokens", 8192),
            stop_sequences=body.get("stop", []),
        )

        contents = [Content(role="user", parts=[Part.from_text(user_message)])]

        response = model.generate_content(
            contents,
            stream=False,
            generation_config=generation_config,
        )

        return response.text
    
    def retriever(
        self,
        project_id: str,
        location: str,
        engine_id: str,
        search_query: str,
    ) -> List[discoveryengine.SearchResponse]:
        #  For more information, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/locations#specify_a_multi-region_for_your_data_store
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )

        # Create a client
        client = discoveryengine.SearchServiceClient(client_options=client_options)

        # The full resource name of the search app serving config
        serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_serving_config"

        # Refer to the `ContentSearchSpec` reference for all supported fields:
        # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest.ContentSearchSpec
        content_search_spec = discoveryengine.SearchRequest.ContentSearchSpec(
            # For information about snippets, refer to:
            # https://cloud.google.com/generative-ai-app-builder/docs/snippets
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            ),
            # SearchResultMode https://cloud.google.com/generative-ai-app-builder/docs/reference/rest/v1/SearchResultMode
            search_result_mode=discoveryengine.SearchRequest.ContentSearchSpec.SearchResultMode.CHUNKS,  
        )
        # Refer to the `SearchRequest` reference for all supported fields:
        # https://cloud.google.com/python/docs/reference/discoveryengine/latest/google.cloud.discoveryengine_v1.types.SearchRequest
        request = discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=search_query,
            page_size=20,
            content_search_spec=content_search_spec,
            spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
                mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
            ),
        )

        return client.search(request)

    def parse_results(self, results):
        context = "<CONTEXT>\n"
        citations_list = "> ##### References:\n"
        for i, result in enumerate(results):
            context += "Chunk: " + str(i+1) + "\n"
            context += "Title: " + result.chunk.document_metadata.title + "\n"
            context += "Relevance Score: " + str(result.chunk.relevance_score) + "\n"
            context += "Content: " + result.chunk.content + "\n\n"
            citations_list += f"> [{i+1}] - Page {result.chunk.page_span.page_start}-{result.chunk.page_span.page_end} - [{result.chunk.document_metadata.title}]({result.chunk.document_metadata.uri})\n\n"
        context += "</CONTEXT>"

        return context, citations_list

    def rag_query(
        self,
        search_query: str,
        persona: str,
        prompt_template: str,
        llm: GenerativeModel,
    ):
        results_chunks = retriever(
            PROJECT_ID,
            REGION,
            ENGINE_ID,
            SEARCH_QUERY,
        )

        context, citations_list = parse_results(results_chunks.results)

        prompt = prompt_template.format(
            persona=persona,
            context=context,
            search_query=search_query,
        )

        response = llm.generate_content(
            [prompt],
        )

        return response.text
