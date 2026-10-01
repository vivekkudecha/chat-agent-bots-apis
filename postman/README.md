# Chat Agent Bots API - Postman Collection & Environment

This directory contains the official Postman collection and local environment for testing and integrating with the **Chat Agent Bots API** (FastAPI, PostgreSQL, Qdrant, and vLLM).

---

## Files

| File | Description |
|---|---|
| [`chat-agent-bots-api.postman_collection.json`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman/chat-agent-bots-api.postman_collection.json) | Complete Postman Collection v2.1.0 with 52 requests across all 10 API modules, pre-configured authentication, and automated variable extraction scripts. |
| [`chat-agent-bots-local.postman_environment.json`](file:///Users/vivek/tatatel/chat-agent-bots-apis/postman/chat-agent-bots-local.postman_environment.json) | Local development environment containing default server URLs and placeholder credentials. |

---

## Key Features

1. **100% OpenAPI Coverage (52 Requests)**:
   - **01. Authentication** (Register, Login, Refresh Token)
   - **02. Users** (Get Profile, Update Profile, Change Password)
   - **03. AI Models** (Create Model, List Models, Get Model, Update Model, Delete Model, Configure Model for Bot, List Bot Model Configs, Remove Model from Bot)
   - **04. Bots** (Create Bot, List Bots, Get Bot, Update Bot, Delete Bot, Publish Version, List Versions, Attach KB, Detach KB)
   - **05. Knowledge Bases** (Create KB, List KBs, Get KB, Update KB, Delete KB)
   - **06. Documents** (Upload Document multipart/form-data, List KB Documents, Get Document, Retry Document Processing, Delete Document)
   - **07. Guardrails** (Create Guardrail, List Guardrails, Get Guardrail, Update Guardrail, Attach Guardrail to Bot, List Bot Guardrails, Update Bot Guardrail, Remove Guardrail from Bot, Test Guardrails)
   - **08. Conversations** (Create Conversation, List Conversations, Get Conversation, Get Messages, Update Conversation, Delete Conversation)
   - **09. Chat** (Send Chat Message in existing thread, Send Chat Message with auto-created thread)
   - **10. System** (Root Status, Health Check)

2. **Automated Token & UUID Capture (Zero Copy-Pasting)**:
   - When you run **Login** or **Register**, the Postman test script automatically stores `access_token` and `refresh_token` in Collection Variables.
   - When you create a **Bot**, **Knowledge Base**, **Document**, **Model**, **Guardrail**, or **Conversation**, their respective UUIDs are automatically saved to `bot_id`, `knowledge_base_id`, `document_id`, `model_id`, `guardrail_id`, and `conversation_id`.
   - Subsequent requests automatically reference these variables via `{{bot_id}}`, `{{knowledge_base_id}}`, etc.

3. **Collection-Level Bearer Token Authentication**:
   - The collection uses `Bearer {{access_token}}` at the root.
   - Public endpoints (Register, Login, Refresh, System) are configured with `noauth` to bypass authentication headers.

---

## Quick Start Guide

### 1. Import into Postman
1. Open Postman.
2. Click **Import** (top left).
3. Drag & drop or select both:
   - `postman/chat-agent-bots-api.postman_collection.json`
   - `postman/chat-agent-bots-local.postman_environment.json`
4. Select the environment **Chat Agent Bots - Local Development** from the environment dropdown in Postman.

### 2. Recommended Execution Sequence
To test the complete platform workflow from end to end:

1. **Register or Login**:
   - Run `01. Authentication -> Login` (or `Register User`).
   - Notice that `{{access_token}}` is automatically populated.

2. **Register an AI Model**:
   - Run `03. AI Models -> Create AI Model` (e.g. OpenAI `gpt-4o-mini` or vLLM endpoint).
   - `{{model_id}}` is saved automatically.

3. **Create a Knowledge Base**:
   - Run `05. Knowledge Bases -> Create Knowledge Base`.
   - `{{knowledge_base_id}}` is saved automatically.

4. **Upload a Document (Optional)**:
   - Open `06. Documents -> Upload Document`.
   - In the **Body** tab, choose a file for the `file` form-data key and send.
   - `{{document_id}}` is saved automatically.

5. **Create a Bot**:
   - Run `04. Bots -> Create Bot`.
   - `{{bot_id}}` is saved automatically.

6. **Configure Bot Model & Attach Knowledge Base**:
   - Run `03. AI Models -> Configure Model for Bot`.
   - Run `04. Bots -> Attach Knowledge Base to Bot`.

7. **Attach and Test Guardrails**:
   - Run `07. Guardrails -> Create Guardrail Definition`.
   - Run `07. Guardrails -> Attach Guardrail to Bot`.
   - Run `07. Guardrails -> Test Bot Guardrails` with a sample prompt.

8. **Start a Conversation and Chat**:
   - Run `08. Conversations -> Create Conversation`.
   - Run `09. Chat -> Send Chat Message (Existing Thread)` to interact with the bot!

---

## Running with Newman (CLI)

You can run automated test collections using Newman:

```bash
npm install -g newman

newman run postman/chat-agent-bots-api.postman_collection.json \
  -e postman/chat-agent-bots-local.postman_environment.json \
  --delay-request 100
```
