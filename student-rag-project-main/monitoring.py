# monitoring.py
# -------------
# This file monitors the quality of our RAG app's responses.
#
# What is hallucination?
# Even when we give an LLM context documents, it sometimes generates
# information that isn't actually in those documents. It "fills in the gaps"
# with plausible-sounding but unverified facts. This is called hallucination.
#
# How do we detect it?
# We use a technique called "LLM-as-judge": we send the answer AND the
# source documents back to Gemini and ask it to evaluate whether the answer
# is actually supported by the context. This is a common pattern in
# production RAG systems.

from google import genai
from google.genai import types
from config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)


def check_hallucination(question, context_docs, answer, _client=_client):
    """
    Ask Gemini to evaluate whether the generated answer is grounded in
    the source documents that were retrieved.
    """
    try:
        # 1. Format documents into a single string
        context = "\n\n".join([
            f"Document {i+1}: {doc}"
            for i, doc in enumerate(context_docs)
        ])

        # 2. Build the LLM-as-judge prompt
        prompt = f"""
You are an evaluator. Compare the answer to the provided source documents.

Question:
{question}

Source Documents:
{context}

Answer:
{answer}

Classification rules:
- GROUNDED: The answer is fully supported by the documents.
- PARTIAL: The answer is somewhat supported, but includes extra information.
- HALLUCINATED: The answer is not supported by the documents at all.

Respond with EXACTLY ONE WORD:
GROUNDED, PARTIAL, or HALLUCINATED.
"""

        # 3. Call Gemini with temperature=0.0
        response = _client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            generation_config={"temperature": 0.0}
        )

        verdict = response.text.strip().upper()

        # 4. Validate verdict
        valid = {"GROUNDED", "PARTIAL", "HALLUCINATED"}
        if verdict not in valid:
            verdict = "PARTIAL"

        # 5. Set warning message
        if verdict == "GROUNDED":
            warning = ""
            is_grounded = True
        elif verdict == "PARTIAL":
            warning = "Note: This answer may include some information beyond the provided sources."
            is_grounded = False
        else:  # HALLUCINATED
            warning = "Warning: This answer may contain information not found in the source documents."
            is_grounded = False

        # 6. Return result
        return {
            "verdict": verdict,
            "is_grounded": is_grounded,
            "warning": warning
        }

    except Exception:
        return {
            "verdict": "UNKNOWN",
            "is_grounded": True,
            "warning": ""
        }

    # TODO (Week 13): Implement LLM-as-judge hallucination detection.
    #
    # --- The RAG concept ---
    # This is a key quality-control technique in RAG systems. We use Gemini
    # to judge Gemini's own output — asking it to compare the answer against
    # the source documents and decide if the answer stayed within what the
    # sources actually say.
    #
    # Why temperature=0.0?
    # We want a consistent classification result, not a creative one.
    # Setting temperature to 0 makes the model deterministic and precise.
    #
    # Steps:
    #   1. Format context_docs into a single string:
    #      context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context_docs)])
    #
    #   2. Build a prompt that shows Gemini the context and answer, and asks
    #      it to respond with exactly one word: GROUNDED, PARTIAL, or HALLUCINATED.
    #      Explain what each verdict means in the prompt.
    #
    #   3. Call _client.models.generate_content() with temperature=0.0
    #      Get the verdict with: verdict = response.text.strip().upper()
    #
    #   4. If the verdict is not one of the three valid words, default to "PARTIAL"
    #      (the model sometimes adds punctuation or extra words)
    #
    #   5. Set the "warning" string:
    #      - GROUNDED     → warning = ""
    #      - PARTIAL      → warning = "Note: This answer may include some information beyond the provided sources."
    #      - HALLUCINATED → warning = "Warning: This answer may contain information not found in the source documents."
    #
    #   6. Return the dict. Wrap everything in try/except — if this call fails,
    #      return: {"verdict": "UNKNOWN", "is_grounded": True, "warning": ""}
    #
    #return {"verdict": "UNKNOWN", "is_grounded": True, "warning": ""}  # placeholder


def calculate_confidence(distances):
    """
    Convert ChromaDB similarity distances into a 0–1 confidence score.

    Args:
        distances: A list of L2 distance values from ChromaDB.
                   0.0 = identical vectors, 2.0 = completely different.

    Returns:
        A float between 0.0 (not confident) and 1.0 (very confident).
    """

    # 1. If distances is empty, return 0.0
    if not distances:
        return 0.0

    # 2. Compute the average distance
    avg_distance = sum(distances) / len(distances)

    # 3. Apply the formula: confidence = max(0.0, 1.0 - (avg_distance / 2.0))
    confidence = max(0.0, 1.0 - (avg_distance / 2.0))
    

    # 4. Return rounded to 2 decimal places
    return round(confidence, 2)

    # TODO (Week 13): Implement the confidence score calculation.
    #
    # --- The RAG concept ---
    # When ChromaDB retrieves documents, it returns a "distance" for each one.
    # Distance measures how far apart two vectors are in embedding space.
    # A low distance means the document is very similar to the query —
    # which means we can be more confident the answer will be relevant.
    #
    # The formula to convert distance to confidence:
    #   confidence = max(0.0, 1.0 - (avg_distance / 2.0))
    #
    # Why divide by 2? ChromaDB L2 distances range from 0 to 2, so
    # dividing by 2 scales the result to a 0–1 range.
    #
    # Steps:
    #   1. If distances is empty, return 0.0
    #   2. Compute the average: avg_distance = sum(distances) / len(distances)
    #   3. Apply the formula above
    #   4. Return the result rounded to 2 decimal places: round(confidence, 2)
    #
    #return 0.0  # placeholder — replace with your implementation
