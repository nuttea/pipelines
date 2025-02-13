"""
title: Vertex AI Search Answer with Follow-up Pipeline
author: nuttee
date: 2024-11-05
version: 1.0
license: Apache 2.0
description: A pipeline for retrieving relevant information or summary answer from a Vertex AI Search App.
requirements: vertexai, google-cloud-aiplatform, google-cloud-discoveryengine, google-api-core
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
"""

import os
import uuid
from typing import List, Union, Generator, Iterator, Optional
from pprint import pprint

from pydantic import BaseModel, Field
import urllib.parse

from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from google.api_core.client_options import ClientOptions

import vertexai
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
        ANSWER_PREAMBLE: str

    def __init__(self):
        self.name = "Vertex AI Search Answer with Follow-up"
        self.documents = None
        self.index = None
        self.debug = True

        self.valves = self.Valves(
            **{
                "PROJECT_ID": os.getenv("PROJECT_ID", ""),
                "REGION": os.getenv("REGION", ""),
                "LOCATION": os.getenv("LOCATION", ""),
                "ENGINE_ID": os.getenv("ENGINE_ID", ""),
                "ANSWER_PREAMBLE": "",
            }
        )

    async def on_startup(self):
        # This function is called when the server is started.
        vertexai.init(
            project=self.valves.PROJECT_ID,
            location=self.valves.LOCATION,
        )

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
    
    async def on_valves_updated(self) -> None:
        """This function is called when the valves are updated."""
        print(f"on_valves_updated:{__name__}")
        vertexai.init(
            project=self.valves.PROJECT_ID,
            location=self.valves.LOCATION,
        )
    
    def answer(
        self,
        project_id: str,
        location: str,
        engine_id: str,
        search_query: str,
        user_pseudo_id: str,
        session: str = None,
        query_rephraser: bool = False,
        max_rephrase_steps: int = 2,
        model_version: str = "gemini-1.5-flash-002/answer_gen/v1",
        answer_preamble: str = None,
        answer_language_code: str = None,
        include_grounding: bool = True,
        filter_level: str = "FILTERING_LEVEL_LOW",
        **kwargs,
    ) -> discoveryengine.AnswerQueryResponse:
        """
        Answers a search query using the Discovery Engine API.

        Args:
            project_id: The Google Cloud project ID.
            location: The location of the engine.
            engine_id: The ID of the engine to use.
            search_query: The search query.
            user_pseudo_id: The user pseudo ID.
            session: The session ID.
            query_rephraser: The query rephraser.
            max_rephrase_steps: The maximum number of rephrase steps.
            model_version: The model version.
            answer_preamble: The answer preamble.
            answer_language_code: The answer language code.
            include_grounding: Whether to include grounding.
            filter_level: The filter level.

        Returns:
            The answer query response.
        """

        #  For more information, refer to:
        # https://cloud.google.com/generative-ai-app-builder/docs/samples/genappbuilder-answer-query
        client_options = (
            ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
            if location != "global"
            else None
        )

        # Create a client
        client = discoveryengine.ConversationalSearchServiceClient(
            client_options=client_options
        )

        # The full resource name of the search app serving config
        serving_config = f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}/servingConfigs/default_serving_config"

        # Create session if has User Pseudo ID and no session
        if session is None:
            session = client.create_session(
                # The full resource name of the engine
                parent=f"projects/{project_id}/locations/{location}/collections/default_collection/engines/{engine_id}",
                session=discoveryengine.Session(user_pseudo_id=user_pseudo_id),
            )
            print(f"Session: {session.name}")

        # Optional: Options for query phase
        query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
            query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
                disable=query_rephraser,  # Optional: Disable query rephraser
                max_rephrase_steps=max_rephrase_steps,  # Optional: Number of rephrase steps
            ),
            # Optional: Classify query types
            query_classification_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec(
                types=[
                    discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.ADVERSARIAL_QUERY,
                    discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.NON_ANSWER_SEEKING_QUERY,
                ]  # Options: ADVERSARIAL_QUERY, NON_ANSWER_SEEKING_QUERY or both
            ),
        )

        # Optional: Options for answer phase
        if answer_preamble == "":
            answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
                ignore_adversarial_query=True,
                ignore_non_answer_seeking_query=True,
                ignore_low_relevant_content=True,
                include_citations=True,
                model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                    model_version=model_version
                ),
            )
        else:
            answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
                ignore_adversarial_query=True,
                ignore_non_answer_seeking_query=True,
                ignore_low_relevant_content=True,
                include_citations=True,
                model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                    model_version=model_version
                ),
                prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
                    preamble=answer_preamble,
                    # Example """Given the conversation between a user and a helpful assistant and some search results, create a final answer for the assistant. The answer should use all relevant information from the search results, not introduce any additional information, and use exactly the same words as the search results when possible. The assistant's answer should be no more than 30 sentences. The assistant's answer should be main topic with transaction code and follow by step-by-step instructions formatted as a bulleted list. Each list item should start with the "-" symbol."""
                ),
            )

        request = discoveryengine.AnswerQueryRequest(
            serving_config=serving_config,
            query=discoveryengine.Query(text=search_query),
            user_pseudo_id=user_pseudo_id,
            session=session.name,
            answer_generation_spec=answer_generation_spec,
            query_understanding_spec=query_understanding_spec,
            **kwargs,
        )

        response = client.answer_query(request)

        return response
    
    def add_references_to_answer(self, answer):
        """
        Adds reference IDs to the answer text in Markdown format.

        Args:
        answer: A dictionary containing the answer text and citation information.

        Returns:
        A string with the answer text and reference IDs in Markdown format.
        """

        answer_text = answer.answer_text
        citations = answer.citations
        references = answer.references

        # Sort citations by start_index in descending order to avoid index issues
        # when inserting reference IDs.
        citations.sort(key=lambda x: x.end_index, reverse=True)
        all_ref_ids = []
        for citation in citations:
            start_index = citation.start_index
            end_index = citation.end_index
            ref_ids = sorted([source.reference_id for source in citation.sources])
            all_ref_ids.extend(ref_ids)
            ref_string = f"***[{', '.join(ref_ids)}]***"
            answer_text = answer_text[:end_index] + ref_string + answer_text[end_index:]

        # Add references after the footnotes
        distinct_ref_ids = sorted(list(set(all_ref_ids)))
        citations_list = []
        for ref_id in distinct_ref_ids:
            title = references[int(ref_id)].chunk_info.document_metadata.title
            url = references[int(ref_id)].chunk_info.document_metadata.uri
            #print(f"URL: {url}")
            # PPTX in GCS
            authenticated_url = urllib.parse.quote(url).replace("gs%3A//", "https://storage.mtls.cloud.google.com/")
            # PDF in GCS
            #authenticated_url = urllib.parse.quote(url).replace("gs%3A//", "https://storage.mtls.cloud.google.com/").replace("/pptx/", "/pdf/").replace(".pptx", ".pptx.pdf")
            #print(f"Authenticated URL: {authenticated_url}")
            citations_list.append(f"> {ref_id} - [{title}]({authenticated_url})")

        return answer_text, citations_list

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

        # Answer Query
        # Set session to None if not found in body
        if "session" not in body:
            body["session"] = None

        response = self.answer(
            project_id=self.valves.PROJECT_ID,
            location=self.valves.REGION,
            engine_id=self.valves.ENGINE_ID,
            search_query=user_message,
            user_pseudo_id=body["user"]["id"],
            session=body["session"],
            answer_preamble=self.valves.ANSWER_PREAMBLE,
        )

        # Set session name to body["session"]
        if body["session"] is None:
            body["session"] = response.session.name
        
        #answer, citations_list = self.add_references_to_answer(response.answer)
        #return answer + "\n\n" + "> ##### References:\n" + "\n".join(citations_list)

        return response.answer.answer_text
