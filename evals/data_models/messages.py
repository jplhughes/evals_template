from enum import Enum
from typing import Sequence, Optional, Dict, Self
from pydantic import BaseModel
import anthropic
from termcolor import cprint

from evals.data_models.hashable import HashableBaseModel
from evals.data_models.inference import LLMResponse

PRINT_COLORS = {"user": "cyan", "system": "magenta", "assistant": "light_green", "none": "cyan"}


class MessageRole(str, Enum):
    user = "user"
    system = "system"
    assistant = "assistant"
    # none is designed for completion tasks where no role / tag will be added
    none = "none"


class ChatMessage(HashableBaseModel):
    role: MessageRole
    content: str
    # model_config = ConfigDict(frozen=True)

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def remove_role(self) -> "ChatMessage":
        return ChatMessage(role=MessageRole.none, content=self.content)


class PromptTemplate(BaseModel):
    method: str
    messages: Sequence[ChatMessage]
    messages_followup: Optional[Sequence[ChatMessage]] = None
    extra: Dict[str, str] = {}


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

    def __add__(self, other: "Prompt") -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def is_none_in_messages(self) -> bool:
        return any(msg.role == MessageRole.none for msg in self.messages)

    def is_last_message_assistant(self) -> bool:
        return self.messages[-1].role == MessageRole.assistant

    def add_assistant_message(self, message: str) -> "Prompt":
        return self + Prompt(messages=[ChatMessage(role=MessageRole.assistant, content=message)])

    def openai_format(self) -> list[Dict]:
        if self.is_last_message_assistant():
            raise ValueError(
                f"OpenAI chat prompts cannot end with an assistant message. Got {self.messages[-1].role}: {self.messages[-1].content}"
            )
        if self.is_none_in_messages():
            raise ValueError(f"OpenAI chat prompts cannot have a None role. Got {self.messages}")
        return [msg.model_dump() for msg in self.messages]

    def anthropic_format(self) -> str:
        if self.is_none_in_messages():
            raise ValueError(f"Anthropic chat prompts cannot have a None role. Got {self.messages}")
        message = ""
        for msg in self.messages:
            match msg.role:
                case MessageRole.user:
                    message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
                case MessageRole.assistant:
                    message += f"{anthropic.AI_PROMPT} {msg.content}"
                case MessageRole.none:
                    message += f"\n\n{msg.content}"
                case MessageRole.system:
                    # No need to add something infront for system messages
                    message += f"\n\n{msg.content}"
        # Add the required empty assistant tag for Claude models if the last message does not have the assistant role
        if self.messages[-1].role != MessageRole.assistant:
            message += f"{anthropic.AI_PROMPT}"
        return message

    def pretty_print(self, responses: list[LLMResponse]) -> None:
        for msg in self.messages:
            if msg.role != MessageRole.none:
                cprint(f"=={msg.role.upper()}:", "white")
            cprint(msg.content, PRINT_COLORS[msg.role])
        for i, response in enumerate(responses):
            cprint(f"==RESPONSE {i + 1} ({response.model_id}):", "white")
            cprint(response.completion, PRINT_COLORS["assistant"], attrs=["bold"])
        print()
