from enum import Enum
from typing import Sequence, Optional, Dict, Self
from pydantic import BaseModel
import anthropic

from evals.data_models.hashable import HashableBaseModel


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


class Prompt(BaseModel):
    messages: Sequence[ChatMessage]

    def __str__(self) -> str:
        out = ""
        for msg in self.messages:
            if msg.role != MessageRole.none:
                out += f"\n\n{msg.role.value}: {msg.content}"
            else:
                out += f"\n{msg.content}"
        return out.strip()

    @classmethod
    def from_prompt(cls, prompt: "Prompt") -> Self:
        return cls(messages=prompt.messages)

    def __add__(self, other: "Prompt") -> Self:
        return self.__class__(messages=list(self.messages) + list(other.messages))

    def openai_format(self) -> list[Dict]:
        return [msg.dict() for msg in self.messages]

    def anthropic_format(self) -> str:
        # Add the required empty assistant tag for Claude models if the last message does not have the assistant role
        if self.messages[-1].role == MessageRole.user:
            self.messages.append(ChatMessage(role=MessageRole.assistant, content=""))

        message = ""
        for msg in self.messages:
            match msg.role:
                case MessageRole.user:
                    message += f"{anthropic.HUMAN_PROMPT} {msg.content}"
                case MessageRole.assistant:
                    message += f"{anthropic.AI_PROMPT} {msg.content}"
                case MessageRole.none:
                    raise ValueError(f"Anthropic chat messages cannot have a None role. Got {self.messages}")
                case MessageRole.system:
                    # No need to add something infront for system messages
                    message += f"\n\n{msg.content}"
        return message


class PromptTemplate(BaseModel):
    method: str
    messages: Sequence[ChatMessage]
    messages_followup: Optional[Sequence[ChatMessage]] = None
    extra: Dict[str, str] = {}
