from typing import Optional, List
from pydantic import BaseModel

from evals.data_models.inference import LLMParams, LLMResponse
from evals.data_models.messages import Prompt


class LLMCache(BaseModel):
    params: LLMParams
    prompt: Prompt
    responses: Optional[List[LLMResponse]] = None
