QA_PROMPT = """You are a helpful AI assistant. Answer the following question based ONLY on the provided evidence.

Instructions:
- Answer using only the provided evidence.
- Do not invent unsupported facts.
- Produce concise factual statements.
- If the evidence does not contain the answer, state that you do not have enough information.

Evidence:
{evidence}

Question: {query}
Answer:"""

CLAIM_DECOMPOSITION_PROMPT = """Decompose the following answer into atomic claims.
Each claim must be:
- atomic
- independently verifiable
- self-contained
- factual
- free of unresolved pronouns
- concise

Return the result as a valid JSON object matching this schema:
{{
  "claims": [
    {{
      "claim_id": "c1",
      "text": "...",
      "source_sentence": "..."
    }}
  ]
}}

Answer to decompose:
{answer}
"""

REGENERATE_CLAIM_PROMPT = """Given the original claim and new evidence, rewrite the claim so it is factually accurate and supported by the new evidence.
If the new evidence cannot support the claim or any factually true variation of it, return INSUFFICIENT_EVIDENCE.

Original claim: {claim}

New evidence:
{evidence}

Return the result as a valid JSON object matching this schema:
{{
  "status": "success",
  "claim": "..."
}}
or:
{{
  "status": "insufficient_evidence",
  "claim": null
}}
"""
