"""
title: Vertex AI Search Streaming Answer Pipeline
author: nuttee
date: 2024-11-05
version: 1.0
license: Apache 2.0
description: A pipeline for retrieving relevant information or summary answer from a Vertex AI Search App.
requirements: vertexai, google-cloud-aiplatform, google-cloud-discoveryengine, google-api-core, google-auth, requests
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
"""

import os
import uuid
from typing import List, Union, Generator, Iterator, Optional
from pprint import pprint

from pydantic import BaseModel, Field
import urllib.parse
import codecs

from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions
import google.auth
import google.auth.transport.requests

import vertexai
import requests
from vertexai.generative_models import (
    Content,
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

class Pipeline:

    class Valves(BaseModel):
        priority: int = Field(
            default=100, description="Priority level for the filter operations."
        )
        PROJECT_ID: str
        REGION: str
        LOCATION: str
        ENGINE_ID: str

    def __init__(self):
        self.name = "Vertex AI Streaming Answer"
        self.documents = None
        self.index = None
        self.debug = True
        self.creds = None
        self.auth_req = None
        self.url = None


        self.valves = self.Valves(
            **{
                "PROJECT_ID": os.getenv("PROJECT_ID", ""),
                "REGION": os.getenv("REGION", ""),
                "LOCATION": os.getenv("LOCATION", ""),
                "ENGINE_ID": os.getenv("ENGINE_ID", ""),
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        vertexai.init(
            project=self.valves.PROJECT_ID,
            location=self.valves.REGION,
        )

        # Programmatically get an access token
        self.creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.auth_req = google.auth.transport.requests.Request()
        self.creds.refresh(self.auth_req)

        serving_config = f"projects/{self.valves.PROJECT_ID}/locations/{self.valves.LOCATION}/collections/default_collection/engines/{self.valves.ENGINE_ID}/servingConfigs/default_search:streamAnswer"
        self.url = f"https://discoveryengine.googleapis.com/v1/{serving_config}"


    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
    
    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")

        vertexai.init(
            project=self.valves.PROJECT_ID,
            location=self.valves.REGION,
        )

        serving_config = f"projects/{self.valves.PROJECT_ID}/locations/{self.valves.LOCATION}/collections/default_collection/engines/{self.valves.ENGINE_ID}/servingConfigs/default_search:streamAnswer"
        self.url = f"https://discoveryengine.googleapis.com/v1/{serving_config}"
    
    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        # Check for presence of required keys and generate chat_id if missing
        if "chat_id" not in body:
            unique_id = f"SYSTEM MESSAGE {uuid.uuid4()}"
            body["chat_id"] = unique_id
            print(f"chat_id was missing, set to: {unique_id}")
        if self.debug:
            print(f"inlet: {__name__} - body:")
            pprint(body)
            print(f"inlet: {__name__} - user:")
            pprint(user)
        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        print(f"outlet: {__name__}")
        if self.debug:
            print(f"outlet: {__name__} - body:")
            pprint(body)
            print(f"outlet: {__name__} - user:")
            pprint(user)
        return body
    
    def stream_answer_response(
        self,
        search_query: str,
        url: str,
    ):
        # Check creds validity
        if self.creds.expired:
            print("Credentials are expired, refreshing...")
            self.creds.refresh(self.auth_req)
            print("Credentials refreshed successfully.")
        else:
            print("Credentials are still valid.")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.creds.token}",
            # Include any additional headers you may need for authentication, etc.
        }

        data = {
            "query": {"text": search_query}
        }

        with requests.post(url, headers=headers, json=data, stream=True) as responses:
            decoder = codecs.getincrementaldecoder("utf-8")()
            response = ""

            for chunk in responses.iter_lines():
                try:
                    text = decoder.decode(chunk, final=False)  # Decode with final=False
                    response += text
                    if text.startswith('    "answerText":'):
                        res = text.split('"')[3].replace("\\n", "\n")
                        print("AnswerText:" + res)
                        yield res
                except UnicodeDecodeError:
                    # Incomplete data, keep accumulating in the buffer
                    pass

    def pipe(
        self,
        user_message: str,
        model_id: str,
        messages: List[dict],
        body: dict,
        __event_emitter__=None,
        __event_call__=None,
        __task__=None,
        __valves__=None,
        
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        if self.debug:
            print(f"pipe: {__name__} - received message from user: {user_message}")
            print("## BODY ##")
            print(body)
            print("## MESSAGES ##")
            print(messages)
            print("## USER MESSAGE ##")
            print(user_message)

        # If you'd like to check for title generation, you can add the following check
        if user_message.startswith("Create a concise, 3-5 word title") and body.get("max_tokens") == 50:
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
        else:
            # Streaming Answer Query
            return self.stream_answer_response(user_message, self.url)