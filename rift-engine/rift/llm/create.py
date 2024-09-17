import functools
import logging
import os
import weakref
from pathlib import Path
from typing import Literal, Optional, Tuple

morph_model_dir = Path.home().joinpath(".morph", "models")

from pydantic import BaseModel, SecretStr

from rift.llm.abstract import AbstractChatCompletionProvider, AbstractCodeCompletionProvider

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    chatModel: str
    codeEditModel: str
    openaiKey: Optional[SecretStr] = None

    def __hash__(self):
        return hash((self.chatModel, self.codeEditModel))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def create_chat(self) -> AbstractChatCompletionProvider:
        logger.info(f"creating chat {self.chatModel}")
        c = create_client(self.chatModel, self.openaiKey)
        assert isinstance(c, AbstractChatCompletionProvider)
        return c

    def create_code_edit(self) -> AbstractCodeCompletionProvider:
        logger.info(f"creating code edit {self.codeEditModel}")
        return create_client(self.codeEditModel, self.openaiKey)

    @classmethod
    def default(cls):
        # we ignore the model name
        return ModelConfig(
            codeEditModel="vertexaillama:any",
            chatModel="vertexaillama:any",
        )


CLIENTS = weakref.WeakValueDictionary()


def create_client(
    config: str, openai_api_key: Optional[SecretStr] = None
) -> AbstractCodeCompletionProvider:
    """Create a client for the given config. If the client has already been created, then it will return a cached one.

    Note that it uses a WeakValueDictionary, so if the client is no longer referenced, it will be garbage collected.
    This is useful because it means you can call create_client multiple times without allocating the same model, but
    if you need to dispose a model this won't keep a reference that prevents it from being garbage collected.
    """
    global CLIENTS

    if config in CLIENTS:
        return CLIENTS[config]
    else:
        client = create_client_core(config, openai_api_key)
        CLIENTS[config] = client
        return client


def parse_type_name_path(config: str) -> Tuple[str, str, str]:
    assert ":" in config, f"Invalid config: {config}"
    type, rest = config.split(":", 1)
    type = type.strip()
    if "@" in rest:
        name, path = rest.split("@", 1)
    else:
        name = rest
        path = ""
    name = name.strip()
    path = path.strip()
    return (type, name, path)


def create_client_core(
    config: str, openai_api_key: Optional[SecretStr]
) -> AbstractCodeCompletionProvider:
    """
    The function parses the `config` string to extract the `type` and the rest of the configuration. It then checks the `type` and based on that, returns different instances of code completion providers.

    For example, if the `type` is `"hf"`, it imports and returns an instance of `HuggingFaceClient` from `rift.llm.hf_client`. If the `type` is `"openai"`, it imports and returns an instance of `OpenAIClient` from `rift.llm.openai_client` with some additional keyword arguments. If the `type` is `"gpt4all"`, it imports and returns an instance of `Gpt4AllModel` from `rift.llm.gpt4all_model` with some additional settings and keyword arguments.

    If the `type` is none of the above, it raises a `ValueError` with a message indicating that the model is unknown.
    """
    type, name, path = parse_type_name_path(config)
    logger.info(f"{type=} {name=} {path=}")
    from rift.llm.vertexai_llama_client import VertexAiLlamaClient

    if type == "openai":
        from rift.llm.openai_client import OpenAIClient
        kwargs = {}
        kwargs["default_model"] = "not-used"
        kwargs["api_key"] = "not-used"
        return OpenAIClient.parse_obj(kwargs)

    elif type == "gpt4all":
        from rift.llm.gpt4all_model import Gpt4AllModel, Gpt4AllSettings

        kwargs = {}
        if name:
            kwargs["model_name"] = name
        if path:
            kwargs["model_path"] = path
        settings = Gpt4AllSettings.parse_obj(kwargs)
        return Gpt4AllModel(settings)
    elif type == "llama":  # llama-cpp-python
        from rift.llm.llama_client import LlamaClient

    elif type == "vertexaillama":
        parsed_path = Path(path if path else name)
        if not parsed_path.is_absolute():
            parsed_path = morph_model_dir.joinpath(parsed_path)
        logger.info(f"Creating LLaMa client with model located at {path}")

        return VertexAiLlamaClient(name=name, model_path=str(parsed_path) if path or name else None)
