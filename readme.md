# 🌌 Nexus AI — The Technical PM Toolkit

A secure, single-user, local AI assistant built specifically for **Technical, Data, AI/ML, and Platform Product Managers**. 

Nexus acts as a specialized PM co-pilot. It routes requests between market research (web search), PRD/FRD generation (structured writing), and RAG over uploaded documents (specs, data, guidelines). 

## 🏗️ Architecture

This is not a simple wrapper around an LLM. It is a multi-agent system orchestrated with **LangGraph**.

```text
User (Chainlit)
      │
      ▼
[ Guardrail ] ──(unsafe)──► [ Blocked ]
      │
   (safe)
      ▼
[ Manager ] ──┬──► [ Web Worker ]    (Tavily Search + Groq)
              ├──► [ Writer Worker ] (Groq)
              └──► [ Doc Worker ]    (ChromaDB RAG + Groq)