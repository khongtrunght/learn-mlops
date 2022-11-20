from typing import List

from fastapi import Query
from pydantic import BaseModel


class Text(BaseModel):
    text: str = Query(None, min_length=1)


class PredictPayload(BaseModel):
    texts: List[Text]
