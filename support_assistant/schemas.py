from pydantic import BaseModel, Field
from typing import List


class SupportResponse(BaseModel):
    answer: str = Field(
        description="Final answer to the user's question"
    )

    sources: List[str] = Field(
        description="List of source document or chunk IDs used"
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1"
    )


# Test
if __name__ == "__main__":

    response = SupportResponse(
        answer="Grocery and perishable items can be returned within 24 hours if damaged, spoiled, or incorrect.",
        sources=["doc_02"],
        confidence=1.0
    )

    print(response.model_dump_json(indent=2))