import dataclasses
import datetime
import os
import secrets
import time
from typing import Optional, Any, List, Optional, Dict, Union
from typing_extensions import TypedDict, NotRequired, Literal

from langchain.llms import VertexAIModelGarden
from langchain.chains import ConversationChain
from langchain.chat_models import ChatVertexAI
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel


# class Message(BaseModel):
#     role: str
#     content: str

# prefer openai_types message
from rift.llm.openai_types import Message


class ChatBody(BaseModel):
    messages: List[Message]
    model: str
    stream: Optional[bool] = False
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]


@dataclasses.dataclass
class VertexLlmWrapper:
    model_name: str
    project: str
    location: str
    temperature = os.environ.get("TEMPERATURE", "0.2")
    top_k = os.environ.get("TOP_K", "40")
    top_p = os.environ.get("TOP_P", "0.8")
    max_output_tokens = os.environ.get("MAX_OUTPUT_TOKENS", "512")

    # post init
    llm: VertexAIModelGarden = dataclasses.field(init=False)
    memory: ConversationBufferMemory = dataclasses.field(init=False)
    conversation: ConversationChain = dataclasses.field(init=False)

    def __post_init__(self):
        self.llm = VertexAIModelGarden(
            model_name=self.model_name,
            endpoint_id=6383221352123334656,
            project=self.project,
            location=self.location,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            max_output_tokens=self.max_output_tokens
        )

        self.memory = ConversationBufferMemory(
            memory_key="history",
            max_token_limit=2048,
            return_messages=True
        )

        self.memory.chat_memory.add_user_message("What day is today?")
        self.memory.chat_memory.add_ai_message(
            datetime.date.today().strftime("Today is %A, %B %d, %Y")
        )

        # Get Vertex AI output
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
        )

    def prompt(self, prompt: str):
        res = self.conversation.predict(input=prompt)
        return self.generate_response(res)

    def generate_response(self, content: str) -> "ChatCompletion":
        """ This is mostly fake """
        ts = int(time.time())
        id = f"cmpl-{secrets.token_hex(12)}"
        return {
            "id": id,
            "created": ts,
            "object": "chat.completion",
            "model": self.model_name,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "choices": [{
                "message": {"text": content},
                "finish_reason": "stop", "index": 0}
            ]
        }

### llama types

class ChatCompletionFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionResponseMessage(TypedDict):
    role: Literal["assistant", "user", "system", "function"]
    content: Optional[str]
    user: NotRequired[str]
    function_call: NotRequired[ChatCompletionFunctionCall]


ChatCompletionMessage = ChatCompletionResponseMessage

class ChatCompletionResponseChoice(TypedDict):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str]


ChatCompletionChoice = ChatCompletionResponseChoice


class CompletionUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: Dict[str, Any]  # TODO: make this more specific


ChatCompletionFunction = ChatCompletionResponseFunction


class CreateChatCompletionResponse(TypedDict):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: CompletionUsage


ChatCompletion = CreateChatCompletionResponse
