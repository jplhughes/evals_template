from enum import Enum
from typing import Callable, Self, Sequence

import anthropic.types
import openai.types.chat
import pydantic
from termcolor import cprint

from .hashable import HashableBaseModel
from .inference import LLMResponse

PRINT_COLORS = {
    "user": "cyan",
    "system": "magenta",
    "assistant": "light_green",
    "none": "cyan",
}


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def openai_format(self) -> openai.types.chat.ChatCompletionMessageParam:
        return {"role": self.role.value, "content": self.content}

    def anthropic_format(self) -> anthropic.types.MessageParam:
        assert self.role.value in ("user", "assistant")
        return anthropic.types.MessageParam(content=self.content, role=self.role.value)

    def remove_role(self) -> Self:
        return self.__class__(role=MessageRole.none, content=self.content)


class PromptTemplate(pydantic.BaseModel):
    method: str
    messages: Sequence[ChatMessage]
    messages_followup: Sequence[ChatMessage] | None = None
    extra: dict[str, str] = {}


class Prompt(HashableBaseModel):
    messages: Sequence[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            if msg.role != MessageRole.none:
                out += f"\n\n{msg.role.value}: {msg.content}"
            else:
                out += f"\n{msg.content}"
        return out.strip()

    def __add__(self, other: Self) -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    @classmethod
    def from_prompt(cls, prompt: Self) -> Self:
        return cls(messages=prompt.messages)

    @classmethod
    def from_almj_prompt_format(
        cls,
        text: str,
        sep: str = 8 * "=",
        strip_content: bool = False,
    ) -> Self:
        if not text.startswith(sep):
            return cls(
                messages=[
                    ChatMessage(
                        role=MessageRole.user,
                        content=text.strip() if strip_content else text,
                    )
                ]
            )

        messages = []
        for role_content_str in ("\n" + text).split("\n" + sep):
            if role_content_str == "":
                continue

            role, content = role_content_str.split(sep + "\n")
            if strip_content:
                content = content.strip()

            messages.append(ChatMessage(role=MessageRole[role], content=content))

        return cls(messages=messages)

    def is_none_in_messages(self) -> bool:
        return any(msg.role == MessageRole.none for msg in self.messages)

    def is_last_message_assistant(self) -> bool:
        return self.messages[-1].role == MessageRole.assistant

    def add_assistant_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=message)])

    def add_user_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.user, content=message)])

    def hf_format(self, hf_model_id: str) -> str:
        match hf_model_id:
            case "cais/zephyr_7b_r2d2" | "HuggingFaceH4/zephyr-7b-beta":
                # See https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
                # and https://github.com/centerforaisafety/HarmBench/blob/1751dd591e3be4bb52cab4d926977a61e304aba5/baselines/model_utils.py#L124-L127
                # for prompt format.
                rendered_prompt = ""
                for msg in self.messages:
                    match msg.role:
                        case MessageRole.system:
                            rendered_prompt += "<|system|>"
                        case MessageRole.user:
                            rendered_prompt += "<|user|>"
                        case MessageRole.assistant:
                            rendered_prompt += "<|assistant|>"
                        case _:
                            raise ValueError(f"Invalid role {msg.role} in prompt")

                    rendered_prompt += f"\n{msg.content}</s>\n"

                match self.messages[-1].role:
                    case MessageRole.user:
                        rendered_prompt += "<|assistant|>\n"
                    case _:
                        raise ValueError("Last message in prompt must be user. " f"Got {self.messages[-1].role}")

                return rendered_prompt

            case _:
                return "\n\n".join(msg.content for msg in self.messages)

    def openai_format(
        self,
    ) -> list[openai.types.chat.ChatCompletionMessageParam]:
        if self.is_last_message_assistant():
            raise ValueError(
                f"OpenAI chat prompts cannot end with an assistant message. Got {self.messages[-1].role}: {self.messages[-1].content}"
            )
        if self.is_none_in_messages():
            raise ValueError(f"OpenAI chat prompts cannot have a None role. Got {self.messages}")
        return [msg.openai_format() for msg in self.messages]

    def anthropic_format(
        self,
    ) -> tuple[str | None, list[anthropic.types.MessageParam]]:
        """Returns (system_message (optional), chat_messages)"""
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic chat prompts cannot have a None role. Got {self.messages}")

        if len(self.messages) == 0:
            return None, []

        if self.messages[0].role == MessageRole.system:
            return self.messages[0].content, [msg.anthropic_format() for msg in self.messages[1:]]

        return None, [msg.anthropic_format() for msg in self.messages]

    def pretty_print(self, responses: list[LLMResponse], print_fn: Callable | None = None) -> None:
        if print_fn is None:
            print_fn = cprint

        for msg in self.messages:
            if msg.role != MessageRole.none:
                print_fn(f"=={msg.role.upper()}:", "white")
            print_fn(msg.content, PRINT_COLORS[msg.role])
        for i, response in enumerate(responses):
            print_fn(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            print_fn(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print_fn("")
