# Faithfulness-Gated Adaptive RAG (FG-ARAG)

## Overview
A closed-loop, claim-level, faithfulness-gated retrieval-augmented generation system where faithfulness verification controls retrieval, regeneration and abstention.

## Architecture
Query → Preprocessing → Hybrid Retrieval → Generator → Claim Decomposition → NLI → Faithfulness Score → Three-way Gate (Accept/Re-Retrieve/Abstain)

## Setup
1. Clone the repository.
2. `pip install -e .`
3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

## Usage
TBD .
