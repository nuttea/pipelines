"""
title: Vertex AI Search Pipeline
author: nuttee
date: 2024-11-05
version: 1.0
license: Apache 2.0
description: A pipeline for retrieving relevant information or summary answer from a Vertex AI Search App.
requirements: google-cloud-aiplatform, google-cloud-discoveryengine, google-api-core
environment_variables: GOOGLE_PROJECT_ID, GOOGLE_CLOUD_REGION
"""

import os
from typing import List, Union, Generator, Iterator, Optional, Callable, Awaitable

from pydantic import BaseModel

from google.cloud import aiplatform
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import discoveryengine_v1alpha
from google.api_core.client_options import ClientOptions

class Pipeline:

    class Valves(BaseModel):
        PROJECT_ID: str
        REGION: str
        LOCATION: str
        ENGINE_ID: str

    def __init__(self):
        self.documents = None
        self.index = None

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
        aiplatform.init(project=self.valves.PROJECT_ID, location=self.valves.LOCATION)

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass
    
    def answer_query(
        self,
        project_id: str,
        location: str,
        engine_id: str,
        search_query: str,
    ) -> List[discoveryengine.SearchResponse]:
        """Answers a query using Vertex AI Search's Generative Answers API.

        This function sends a search query to a Vertex AI Search Engine and returns a
        list of search responses. It includes options for query understanding and
        answer generation.

        Args:
            project_id: The Google Cloud project ID.
            location: The location of the Vertex AI Search engine.
            engine_id: The ID of the Vertex AI Search engine.
            search_query: The search query text.

        Returns:
            A list of Google Cloud Discovery Engine SearchResponse objects.
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

        # Optional: Options for query phase
        query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
            query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
                disable=False,  # Optional: Disable query rephraser
                max_rephrase_steps=1,  # Optional: Number of rephrase steps
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
        answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
            ignore_adversarial_query=True,
            ignore_non_answer_seeking_query=True,
            ignore_low_relevant_content=True,
            include_citations=True,
            model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
                model_version="gemini-1.5-flash-002/answer_gen/v1"
            ),
            prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
                preamble="""Given the conversation between a user and a helpful assistant and some search results, create a final answer for the assistant. The answer should use all relevant information from the search results, not introduce any additional information, and use exactly the same words as the search results when possible. The assistant's answer should be no more than 30 sentences. The assistant's answer should be main topic with transaction code and follow by step-by-step instructions formatted as a bulleted list. Each list item should start with the "-" symbol."""
            ),
        )

        request = discoveryengine.AnswerQueryRequest(
            serving_config=serving_config,
            query=discoveryengine.Query(text=search_query),
            session=None,
            query_understanding_spec=query_understanding_spec,
            answer_generation_spec=answer_generation_spec,
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
            ref_string = f"^[{', '.join(ref_ids)}]^"
            answer_text = answer_text[:end_index] + ref_string + answer_text[end_index:]
            answer_text += "\n\n"
        
        # Add references after the footnotes
        answer_text += "References:\n"
        distinct_ref_ids = sorted(list(set(all_ref_ids)))
        citations_list = []
        for ref_id in distinct_ref_ids:
            title = references[int(ref_id)].chunk_info.document_metadata.title
            uri = references[int(ref_id)].chunk_info.document_metadata.uri
            citations_list.append(f"=={ref_id}== [{title}]({uri})")
            #answer_text += f"- [{ref_id}]: [{title}]({uri})\n"

        return answer_text, citations_list

    async def pipe(
        self, user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict,
         __user__: Optional[dict] = None,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
        __event_call__: Callable[[dict], Awaitable[dict]] = None,
        
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "in_progress",
                            "description": "Thinking...",
                            "done": False
                            },
                    }
                )

        response = self.answer_query(project_id=self.valves.PROJECT_ID,location=self.valves.REGION,engine_id=self.valves.ENGINE_ID,search_query=user_message)

        answer, citations_list = self.add_references_to_answer(response.answer)

        for citation in citations_list:
            await __event_emitter__(
                        {
                            "type": "message",
                            "data": {"content": citation},
                        }
                    )

        await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "status": "complete",
                            "description": "Generated Answer",
                            "done": False
                            },
                    }
                )

        return answer
