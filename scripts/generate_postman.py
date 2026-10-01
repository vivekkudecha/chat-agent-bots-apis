import json
import uuid

def create_collection():
    collection_id = str(uuid.uuid4())
    
    collection = {
        "info": {
            "_postman_id": collection_id,
            "name": "Chat Agent Bots API",
            "description": "### Multi-Bot AI Platform API Collection\n\nComprehensive Postman Collection covering all endpoints and payloads for **Chat Agent Bots API** powered by FastAPI, PostgreSQL, Qdrant, and vLLM.\n\n#### Features\n- **Pre-configured Bearer Token Authentication** inheriting across the collection.\n- **Automated Token & UUID Capture**: Registration, Login, Create Bot, Create KB, and other creation endpoints automatically save tokens and UUIDs into Collection Variables (`access_token`, `bot_id`, `knowledge_base_id`, `conversation_id`, etc.).\n- **Complete Request Payloads**: Every POST, PUT, and PATCH endpoint contains ready-to-test JSON payloads matching the FastAPI Pydantic schema validation rules.\n- **Query & Path Parameters**: All standard pagination, filtering, and path parameters are defined with sensible defaults.",
            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
        },
        "auth": {
            "type": "bearer",
            "bearer": [
                {
                    "key": "token",
                    "value": "{{access_token}}",
                    "type": "string"
                }
            ]
        },
        "variable": [
            {
                "key": "baseUrl",
                "value": "http://localhost:8000",
                "type": "string",
                "description": "Base URL of the FastAPI backend server"
            },
            {
                "key": "access_token",
                "value": "",
                "type": "string",
                "description": "JWT Access Token populated automatically upon login or registration"
            },
            {
                "key": "refresh_token",
                "value": "",
                "type": "string",
                "description": "JWT Refresh Token populated automatically upon login or registration"
            },
            {
                "key": "bot_id",
                "value": "11111111-1111-1111-1111-111111111111",
                "type": "string",
                "description": "UUID of the active bot"
            },
            {
                "key": "knowledge_base_id",
                "value": "22222222-2222-2222-2222-222222222222",
                "type": "string",
                "description": "UUID of the active knowledge base"
            },
            {
                "key": "document_id",
                "value": "33333333-3333-3333-3333-333333333333",
                "type": "string",
                "description": "UUID of the uploaded document"
            },
            {
                "key": "conversation_id",
                "value": "44444444-4444-4444-4444-444444444444",
                "type": "string",
                "description": "UUID of the active conversation thread"
            },
            {
                "key": "model_id",
                "value": "55555555-5555-5555-5555-555555555555",
                "type": "string",
                "description": "UUID of the registered AI model"
            },
            {
                "key": "guardrail_id",
                "value": "66666666-6666-6666-6666-666666666666",
                "type": "string",
                "description": "UUID of the guardrail definition"
            },
            {
                "key": "bot_guardrail_id",
                "value": "77777777-7777-7777-7777-777777777777",
                "type": "string",
                "description": "UUID of the attached bot-guardrail association"
            },
            {
                "key": "version_id",
                "value": "88888888-8888-8888-8888-888888888888",
                "type": "string",
                "description": "UUID of the published bot version"
            }
        ],
        "item": []
    }

    def url_obj(path_str, query_params=None, path_vars=None):
        clean_path = path_str.strip("/")
        parts = clean_path.split("/") if clean_path else []
        url = {
            "raw": "{{baseUrl}}" + path_str + (("?" + "&".join([f"{k}={v}" for k, v, *_ in query_params])) if query_params else ""),
            "host": ["{{baseUrl}}"],
            "path": parts
        }
        if query_params:
            url["query"] = [{"key": k, "value": str(v), "description": (desc[0] if desc else "")} for k, v, *desc in query_params]
        if path_vars:
            url["variable"] = [{"key": k, "value": f"{{{{{v}}}}}", "description": (desc[0] if desc else "")} for k, v, *desc in path_vars]
        return url

    def json_body(payload_dict):
        return {
            "mode": "raw",
            "raw": json.dumps(payload_dict, indent=2),
            "options": {
                "raw": {
                    "language": "json"
                }
            }
        }

    # -------------------------------------------------------------
    # 01. Authentication
    # -------------------------------------------------------------
    auth_items = [
        {
            "name": "Register User",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201 || pm.response.code === 200) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.access_token) {",
                            "        pm.collectionVariables.set('access_token', jsonData.access_token);",
                            "        console.log('Access token saved:', jsonData.access_token);",
                            "    }",
                            "    if (jsonData.refresh_token) {",
                            "        pm.collectionVariables.set('refresh_token', jsonData.refresh_token);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "auth": {"type": "noauth"},
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Admin User",
                    "email": "admin@example.com",
                    "password": "SecurePassword123!"
                }),
                "url": url_obj("/api/v1/auth/register"),
                "description": "Registers a new user account with email, name, and password. Returns user profile details."
            },
            "response": []
        },
        {
            "name": "Login",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 200) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.access_token) {",
                            "        pm.collectionVariables.set('access_token', jsonData.access_token);",
                            "        console.log('Access token saved to collection variable');",
                            "    }",
                            "    if (jsonData.refresh_token) {",
                            "        pm.collectionVariables.set('refresh_token', jsonData.refresh_token);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "auth": {"type": "noauth"},
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "email": "admin@example.com",
                    "password": "SecurePassword123!"
                }),
                "url": url_obj("/api/v1/auth/login"),
                "description": "Authenticates user credentials and returns JWT access_token and refresh_token. Automatically sets collection variables on success."
            },
            "response": []
        },
        {
            "name": "Refresh Token",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 200) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.access_token) {",
                            "        pm.collectionVariables.set('access_token', jsonData.access_token);",
                            "    }",
                            "    if (jsonData.refresh_token) {",
                            "        pm.collectionVariables.set('refresh_token', jsonData.refresh_token);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "auth": {"type": "noauth"},
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "refresh_token": "{{refresh_token}}"
                }),
                "url": url_obj("/api/v1/auth/refresh"),
                "description": "Exchanges a valid refresh token for a new access token and refresh token pair."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 02. Users
    # -------------------------------------------------------------
    user_items = [
        {
            "name": "Get Current User Profile",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj("/api/v1/users/me"),
                "description": "Returns details of the currently authenticated user based on the Bearer JWT token."
            },
            "response": []
        },
        {
            "name": "Update Profile",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Admin User Updated",
                    "email": "admin.updated@example.com"
                }),
                "url": url_obj("/api/v1/users/me"),
                "description": "Updates current user's profile details such as name and email."
            },
            "response": []
        },
        {
            "name": "Change Password",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "current_password": "SecurePassword123!",
                    "new_password": "NewSecurePassword456!"
                }),
                "url": url_obj("/api/v1/users/me/change-password"),
                "description": "Changes the password of the currently authenticated user. Requires current password verification."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 03. AI Models
    # -------------------------------------------------------------
    model_items = [
        {
            "name": "Create AI Model",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('model_id', jsonData.id);",
                            "        console.log('Saved model_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "provider": "openai",
                    "model_key": "gpt-4o-mini",
                    "display_name": "GPT-4o Mini",
                    "name": "gpt-4o-mini",
                    "model_type": "chat",
                    "endpoint_url": "https://api.openai.com/v1",
                    "context_window": 128000,
                    "capabilities": {
                        "streaming": True,
                        "tools": True,
                        "vision": True,
                        "json_mode": True
                    }
                }),
                "url": url_obj("/api/v1/models"),
                "description": "Registers a new AI model provider and configuration. Supports providers like openai, anthropic, vllm, ollama, etc."
            },
            "response": []
        },
        {
            "name": "List AI Models",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/models",
                    query_params=[
                        ("active_only", "true", "Filter only active models"),
                        ("page", "1", "Page number"),
                        ("page_size", "20", "Items per page (max 100)")
                    ]
                ),
                "description": "Retrieves a paginated list of registered AI models."
            },
            "response": []
        },
        {
            "name": "Get AI Model by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/models/:model_id",
                    path_vars=[("model_id", "model_id", "UUID of the AI model")]
                ),
                "description": "Fetches detailed metadata of a specific AI model by UUID."
            },
            "response": []
        },
        {
            "name": "Update AI Model",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "display_name": "GPT-4o Mini (Updated Production)",
                    "name": "gpt-4o-mini-prod",
                    "endpoint_url": "https://api.openai.com/v1",
                    "context_window": 128000,
                    "capabilities": {
                        "streaming": True,
                        "tools": True,
                        "vision": True,
                        "json_mode": True
                    },
                    "is_active": True
                }),
                "url": url_obj(
                    "/api/v1/models/:model_id",
                    path_vars=[("model_id", "model_id", "UUID of the AI model to update")]
                ),
                "description": "Updates AI model configurations, active state, context window or capabilities."
            },
            "response": []
        },
        {
            "name": "Configure Model for Bot",
            "request": {
                "method": "PUT",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "model_id": "{{model_id}}",
                    "is_primary": True,
                    "temperature": 0.7,
                    "top_p": 1.0,
                    "max_tokens": 2048,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stop": ["<|endoftext|>"],
                    "extra_config": {
                        "response_format": {"type": "text"}
                    }
                }),
                "url": url_obj(
                    "/api/v1/models/bots/:bot_id",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot to configure")]
                ),
                "description": "Binds an AI model to a bot with specific inference parameters (temperature, max_tokens, penalty, primary flag)."
            },
            "response": []
        },
        {
            "name": "List Bot Model Configurations",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/models/bots/:bot_id/configs",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Lists all model configurations and hyperparameter settings assigned to a specific bot."
            },
            "response": []
        },
        {
            "name": "Remove Model from Bot",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/models/bots/:bot_id/:model_id",
                    path_vars=[
                        ("bot_id", "bot_id", "UUID of the bot"),
                        ("model_id", "model_id", "UUID of the model to unbind")
                    ]
                ),
                "description": "Removes a specific AI model configuration binding from a bot."
            },
            "response": []
        },
        {
            "name": "Delete AI Model",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/models/:model_id",
                    path_vars=[("model_id", "model_id", "UUID of the model to delete")]
                ),
                "description": "Permanently deletes an AI model registration."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 04. Bots
    # -------------------------------------------------------------
    bot_items = [
        {
            "name": "Create Bot",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('bot_id', jsonData.id);",
                            "        console.log('Saved bot_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Enterprise Support Assistant",
                    "slug": "enterprise-support-assistant",
                    "description": "AI-powered customer and technical support bot with knowledge base access",
                    "system_instruction": "You are a professional enterprise support assistant. Answer queries clearly, helpfully, and concisely based on provided context.",
                    "welcome_message": "Hello! How can I help you today with our platform?",
                    "conversation_starters": [
                        "How do I set up SSO authentication?",
                        "Where can I find API rate limit guidelines?",
                        "Can you help troubleshoot connection errors?"
                    ],
                    "visibility": "private",
                    "avatar_url": "https://example.com/assets/bot-avatar.png",
                    "metadata": {
                        "department": "Technical Support",
                        "tier": "enterprise"
                    }
                }),
                "url": url_obj("/api/v1/bots"),
                "description": "Creates a new bot with system instructions, persona configuration, starter questions, and metadata."
            },
            "response": []
        },
        {
            "name": "List Bots",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots",
                    query_params=[
                        ("page", "1", "Page number"),
                        ("page_size", "20", "Items per page (max 100)")
                    ]
                ),
                "description": "Lists all bots owned by the authenticated user with pagination."
            },
            "response": []
        },
        {
            "name": "Get Bot by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Retrieves complete details of a bot including attached knowledge bases and active models."
            },
            "response": []
        },
        {
            "name": "Update Bot",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Enterprise Support Assistant (v2)",
                    "description": "Updated enterprise support assistant with specialized cloud troubleshooting capabilities",
                    "system_instruction": "You are an advanced enterprise support agent. Provide accurate, polite, and direct technical answers.",
                    "welcome_message": "Welcome back! What platform issue can I help resolve today?",
                    "conversation_starters": [
                        "SSO configuration guide",
                        "Troubleshoot database timeout",
                        "Check billing and invoices"
                    ],
                    "visibility": "private",
                    "is_api_enabled": True
                }),
                "url": url_obj(
                    "/api/v1/bots/:bot_id",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot to update")]
                ),
                "description": "Updates bot profile, system prompt, conversation starters, or visibility."
            },
            "response": []
        },
        {
            "name": "Publish Bot Version",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 200 || pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('version_id', jsonData.id);",
                            "        console.log('Saved version_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id/publish",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot to publish")]
                ),
                "description": "Creates an immutable snapshot version of the bot configuration, prompts, and attached resources."
            },
            "response": []
        },
        {
            "name": "List Bot Versions",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id/versions",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Retrieves the historical published versions of a bot."
            },
            "response": []
        },
        {
            "name": "Attach Knowledge Base to Bot",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id/knowledge-bases/:knowledge_base_id",
                    path_vars=[
                        ("bot_id", "bot_id", "UUID of the bot"),
                        ("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base to attach")
                    ]
                ),
                "description": "Associates a knowledge base with the bot so that user queries can retrieve context via RAG."
            },
            "response": []
        },
        {
            "name": "Detach Knowledge Base from Bot",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id/knowledge-bases/:knowledge_base_id",
                    path_vars=[
                        ("bot_id", "bot_id", "UUID of the bot"),
                        ("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base to detach")
                    ]
                ),
                "description": "Detaches a knowledge base from the bot."
            },
            "response": []
        },
        {
            "name": "Delete Bot",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/bots/:bot_id",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot to delete")]
                ),
                "description": "Deletes a bot and its associated configurations."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 05. Knowledge Bases
    # -------------------------------------------------------------
    kb_items = [
        {
            "name": "Create Knowledge Base",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('knowledge_base_id', jsonData.id);",
                            "        console.log('Saved knowledge_base_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Platform Docs & Runbooks",
                    "description": "Technical documentation, architecture guides, and operational runbooks",
                    "embedding_model": "text-embedding-3-small",
                    "chunking_config": {
                        "strategy": "recursive",
                        "chunk_size": 800,
                        "chunk_overlap": 100,
                        "separators": ["\n\n", "\n", " ", ""]
                    },
                    "retrieval_config": {
                        "search_type": "semantic",
                        "top_k": 5,
                        "score_threshold": 0.75,
                        "enable_reranking": False,
                        "reranker_model": None,
                        "hybrid_alpha": 0.5
                    }
                }),
                "url": url_obj("/api/v1/knowledge-bases"),
                "description": "Creates a new Knowledge Base with chunking strategies (chunk_size, chunk_overlap) and retrieval parameters (search_type: semantic|keyword|hybrid, top_k, threshold)."
            },
            "response": []
        },
        {
            "name": "List Knowledge Bases",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/knowledge-bases",
                    query_params=[
                        ("page", "1", "Page number"),
                        ("page_size", "20", "Items per page (max 100)")
                    ]
                ),
                "description": "Lists all knowledge bases belonging to the user."
            },
            "response": []
        },
        {
            "name": "Get Knowledge Base by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/knowledge-bases/:knowledge_base_id",
                    path_vars=[("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base")]
                ),
                "description": "Retrieves knowledge base details, document counts, and chunk statistics."
            },
            "response": []
        },
        {
            "name": "Update Knowledge Base",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Platform Docs & Runbooks (Hybrid Search)",
                    "description": "Updated platform knowledge base with hybrid retrieval enabled",
                    "chunking_config": {
                        "strategy": "recursive",
                        "chunk_size": 1000,
                        "chunk_overlap": 150
                    },
                    "retrieval_config": {
                        "search_type": "hybrid",
                        "top_k": 8,
                        "score_threshold": 0.7,
                        "enable_reranking": True,
                        "reranker_model": "bge-reranker-large",
                        "hybrid_alpha": 0.6
                    }
                }),
                "url": url_obj(
                    "/api/v1/knowledge-bases/:knowledge_base_id",
                    path_vars=[("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base to update")]
                ),
                "description": "Updates chunking or retrieval parameters for the knowledge base."
            },
            "response": []
        },
        {
            "name": "Delete Knowledge Base",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/knowledge-bases/:knowledge_base_id",
                    path_vars=[("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base to delete")]
                ),
                "description": "Deletes the knowledge base and its indexed vectors."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 06. Documents
    # -------------------------------------------------------------
    doc_items = [
        {
            "name": "Upload Document",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 202) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('document_id', jsonData.id);",
                            "        console.log('Saved document_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json"}],
                "body": {
                    "mode": "formdata",
                    "formdata": [
                        {
                            "key": "file",
                            "type": "file",
                            "description": "Select a file to upload (PDF, TXT, DOCX, MD, HTML)"
                        }
                    ]
                },
                "url": url_obj(
                    "/api/v1/documents/knowledge-bases/:knowledge_base_id",
                    path_vars=[("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base to upload into")]
                ),
                "description": "Uploads a document file to the specified knowledge base. Processing and vector indexing are performed asynchronously in Celery."
            },
            "response": []
        },
        {
            "name": "List Documents in Knowledge Base",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/documents/knowledge-bases/:knowledge_base_id",
                    query_params=[
                        ("page", "1", "Page number"),
                        ("page_size", "20", "Items per page (max 100)")
                    ],
                    path_vars=[("knowledge_base_id", "knowledge_base_id", "UUID of the knowledge base")]
                ),
                "description": "Lists all documents uploaded to a specific knowledge base along with status (pending, processing, completed, failed) and chunk counts."
            },
            "response": []
        },
        {
            "name": "Get Document by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/documents/:document_id",
                    path_vars=[("document_id", "document_id", "UUID of the document")]
                ),
                "description": "Retrieves metadata, processing status, and error logs for a specific document."
            },
            "response": []
        },
        {
            "name": "Retry Document Processing",
            "request": {
                "method": "POST",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/documents/:document_id/retry",
                    path_vars=[("document_id", "document_id", "UUID of the failed document to retry")]
                ),
                "description": "Triggers a retry of the background extraction, chunking, and embedding task for a document."
            },
            "response": []
        },
        {
            "name": "Delete Document",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/documents/:document_id",
                    path_vars=[("document_id", "document_id", "UUID of the document to delete")]
                ),
                "description": "Deletes the document and removes its corresponding chunks from Qdrant vector storage."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 07. Guardrails
    # -------------------------------------------------------------
    guardrail_items = [
        {
            "name": "Create Guardrail Definition",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('guardrail_id', jsonData.id);",
                            "        console.log('Saved guardrail_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "code": "content_safety_filter",
                    "name": "Content Safety and Blocked Terms Filter",
                    "description": "Detects and blocks restricted keywords, sensitive phrases, and disallowed content",
                    "guardrail_type": "input",
                    "handler": "blocked_terms",
                    "default_config": {
                        "terms": ["confidential", "internal only", "password", "secret"],
                        "case_sensitive": False,
                        "whole_word": True,
                        "replacement": "[REDACTED]"
                    }
                }),
                "url": url_obj("/api/v1/guardrails"),
                "description": "Registers a new system guardrail. Handler must be one of: 'blocked_terms', 'pii_detection', 'prompt_injection', 'secret_detection'."
            },
            "response": []
        },
        {
            "name": "List Guardrail Definitions",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/guardrails",
                    query_params=[
                        ("active_only", "true", "Filter active guardrails"),
                        ("page", "1", "Page number"),
                        ("page_size", "50", "Items per page (max 100)")
                    ]
                ),
                "description": "Lists all registered guardrail definitions."
            },
            "response": []
        },
        {
            "name": "Get Guardrail by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/guardrails/:guardrail_id",
                    path_vars=[("guardrail_id", "guardrail_id", "UUID of the guardrail definition")]
                ),
                "description": "Retrieves a guardrail definition by ID."
            },
            "response": []
        },
        {
            "name": "Update Guardrail Definition",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "name": "Content Safety Filter (Updated)",
                    "description": "Expanded list of restricted terms for production compliance",
                    "default_config": {
                        "terms": ["confidential", "internal only", "password", "secret", "private key"],
                        "case_sensitive": False,
                        "whole_word": True,
                        "replacement": "[BLOCKED]"
                    },
                    "is_active": True
                }),
                "url": url_obj(
                    "/api/v1/guardrails/:guardrail_id",
                    path_vars=[("guardrail_id", "guardrail_id", "UUID of the guardrail to update")]
                ),
                "description": "Updates guardrail name, handler default configs, or active status."
            },
            "response": []
        },
        {
            "name": "Attach Guardrail to Bot",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('bot_guardrail_id', jsonData.id);",
                            "        console.log('Saved bot_guardrail_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "guardrail_id": "{{guardrail_id}}",
                    "action": "block",
                    "priority": 100,
                    "is_enabled": True,
                    "config": {
                        "terms": ["confidential", "password", "api_key", "internal secret"],
                        "replacement": "[BLOCKED]"
                    }
                }),
                "url": url_obj(
                    "/api/v1/guardrails/bots/:bot_id",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Attaches a guardrail to a specific bot with action: 'block', 'redact', 'warn', or 'log'."
            },
            "response": []
        },
        {
            "name": "List Guardrails for Bot",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/guardrails/bots/:bot_id/configs",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Lists all guardrails active and attached to the target bot."
            },
            "response": []
        },
        {
            "name": "Update Bot Guardrail Configuration",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "action": "redact",
                    "priority": 50,
                    "is_enabled": True,
                    "config": {
                        "terms": ["confidential", "password", "api_key", "internal secret"],
                        "replacement": "[REDACTED]"
                    }
                }),
                "url": url_obj(
                    "/api/v1/guardrails/bots/:bot_id/:bot_guardrail_id",
                    path_vars=[
                        ("bot_id", "bot_id", "UUID of the bot"),
                        ("bot_guardrail_id", "bot_guardrail_id", "UUID of the bot-guardrail association")
                    ]
                ),
                "description": "Modifies priority, action, or custom config for an attached bot guardrail."
            },
            "response": []
        },
        {
            "name": "Test Bot Guardrails",
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "stage": "input",
                    "text": "Please provide the confidential admin password and internal secret credentials."
                }),
                "url": url_obj(
                    "/api/v1/guardrails/bots/:bot_id/test",
                    path_vars=[("bot_id", "bot_id", "UUID of the bot")]
                ),
                "description": "Simulates guardrail evaluation on sample text at stage: 'input', 'retrieval', 'tool', or 'output'."
            },
            "response": []
        },
        {
            "name": "Remove Guardrail from Bot",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/guardrails/bots/:bot_id/:bot_guardrail_id",
                    path_vars=[
                        ("bot_id", "bot_id", "UUID of the bot"),
                        ("bot_guardrail_id", "bot_guardrail_id", "UUID of the bot-guardrail association")
                    ]
                ),
                "description": "Detaches and removes a guardrail configuration from a bot."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 08. Conversations
    # -------------------------------------------------------------
    conv_items = [
        {
            "name": "Create Conversation",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 201) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.id) {",
                            "        pm.collectionVariables.set('conversation_id', jsonData.id);",
                            "        console.log('Saved conversation_id:', jsonData.id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "bot_id": "{{bot_id}}",
                    "title": "Cloud Infrastructure Setup Discussion",
                    "metadata": {
                        "channel": "web_chat",
                        "user_locale": "en-US"
                    }
                }),
                "url": url_obj("/api/v1/conversations"),
                "description": "Creates a new conversation session associated with a specific bot."
            },
            "response": []
        },
        {
            "name": "List Conversations",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/conversations",
                    query_params=[
                        ("bot_id", "{{bot_id}}", "Optional filter by bot UUID"),
                        ("page", "1", "Page number"),
                        ("page_size", "20", "Items per page (max 100)")
                    ]
                ),
                "description": "Retrieves conversations for the current user, optionally filtered by bot_id."
            },
            "response": []
        },
        {
            "name": "Get Conversation by ID",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/conversations/:conversation_id",
                    path_vars=[("conversation_id", "conversation_id", "UUID of the conversation")]
                ),
                "description": "Fetches conversation metadata and summary."
            },
            "response": []
        },
        {
            "name": "Get Conversation Messages",
            "request": {
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/conversations/:conversation_id/messages",
                    query_params=[
                        ("limit", "50", "Max messages to return (1-200)")
                    ],
                    path_vars=[("conversation_id", "conversation_id", "UUID of the conversation")]
                ),
                "description": "Fetches chronologically ordered messages for the conversation with token usage and latency stats."
            },
            "response": []
        },
        {
            "name": "Update Conversation",
            "request": {
                "method": "PATCH",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "title": "Cloud Infrastructure Setup (Resolved)",
                    "metadata": {
                        "status": "closed",
                        "resolution": "RESOLVED_BY_BOT"
                    }
                }),
                "url": url_obj(
                    "/api/v1/conversations/:conversation_id",
                    path_vars=[("conversation_id", "conversation_id", "UUID of the conversation to update")]
                ),
                "description": "Updates conversation title or custom metadata."
            },
            "response": []
        },
        {
            "name": "Delete Conversation",
            "request": {
                "method": "DELETE",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj(
                    "/api/v1/conversations/:conversation_id",
                    path_vars=[("conversation_id", "conversation_id", "UUID of the conversation to delete")]
                ),
                "description": "Deletes the conversation and its message history."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 09. Chat
    # -------------------------------------------------------------
    chat_items = [
        {
            "name": "Send Chat Message (Existing Thread)",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 200) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.conversation_id) {",
                            "        pm.collectionVariables.set('conversation_id', jsonData.conversation_id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "bot_id": "{{bot_id}}",
                    "conversation_id": "{{conversation_id}}",
                    "message": "Can you summarize the architecture and key benefits of containerized microservices?"
                }),
                "url": url_obj("/api/v1/chat"),
                "description": "Sends a user message in an existing conversation. Executes guardrail checks, semantic knowledge base retrieval, and LLM completion."
            },
            "response": []
        },
        {
            "name": "Send Chat Message (New Conversation Auto-Create)",
            "event": [
                {
                    "listen": "test",
                    "script": {
                        "exec": [
                            "if (pm.response.code === 200) {",
                            "    var jsonData = pm.response.json();",
                            "    if (jsonData.conversation_id) {",
                            "        pm.collectionVariables.set('conversation_id', jsonData.conversation_id);",
                            "        console.log('Created and saved conversation_id:', jsonData.conversation_id);",
                            "    }",
                            "}"
                        ],
                        "type": "text/javascript"
                    }
                }
            ],
            "request": {
                "method": "POST",
                "header": [
                    {"key": "Content-Type", "value": "application/json"},
                    {"key": "Accept", "value": "application/json"}
                ],
                "body": json_body({
                    "bot_id": "{{bot_id}}",
                    "message": "Hello! How does vector search find relevant documents when answering user questions?"
                }),
                "url": url_obj("/api/v1/chat"),
                "description": "Sends a message without conversation_id. Automatically initializes a new conversation session and saves its UUID."
            },
            "response": []
        }
    ]

    # -------------------------------------------------------------
    # 10. System
    # -------------------------------------------------------------
    system_items = [
        {
            "name": "Root Status",
            "request": {
                "auth": {"type": "noauth"},
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj("/"),
                "description": "Returns application service name, current version, and running status."
            },
            "response": []
        },
        {
            "name": "Health Check",
            "request": {
                "auth": {"type": "noauth"},
                "method": "GET",
                "header": [{"key": "Accept", "value": "application/json"}],
                "url": url_obj("/health"),
                "description": "System health probe used by load balancers and orchestrators."
            },
            "response": []
        }
    ]

    collection["item"] = [
        {"name": "01. Authentication", "item": auth_items, "description": "User registration, login, and JWT token refresh endpoints."},
        {"name": "02. Users", "item": user_items, "description": "Profile inspection, profile update, and password management."},
        {"name": "03. AI Models", "item": model_items, "description": "AI model registry and per-bot hyperparameter configuration."},
        {"name": "04. Bots", "item": bot_items, "description": "Bot creation, version publishing, and knowledge base binding."},
        {"name": "05. Knowledge Bases", "item": kb_items, "description": "Knowledge base management, chunking strategy, and retrieval settings."},
        {"name": "06. Documents", "item": doc_items, "description": "Document ingestion, status checking, retry, and deletion."},
        {"name": "07. Guardrails", "item": guardrail_items, "description": "Safety filter definitions, bot attachment, and testing."},
        {"name": "08. Conversations", "item": conv_items, "description": "Conversation lifecycle and message history."},
        {"name": "09. Chat", "item": chat_items, "description": "End-to-end RAG chat execution and message generation."},
        {"name": "10. System", "item": system_items, "description": "Service root info and health probe."}
    ]

    return collection

def create_environment():
    env_id = str(uuid.uuid4())
    environment = {
        "id": env_id,
        "name": "Chat Agent Bots - Local Development",
        "values": [
            {
                "key": "baseUrl",
                "value": "http://localhost:8000",
                "type": "default",
                "enabled": True
            },
            {
                "key": "access_token",
                "value": "",
                "type": "secret",
                "enabled": True
            },
            {
                "key": "refresh_token",
                "value": "",
                "type": "secret",
                "enabled": True
            },
            {
                "key": "bot_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "knowledge_base_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "document_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "conversation_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "model_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "guardrail_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "bot_guardrail_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "version_id",
                "value": "",
                "type": "default",
                "enabled": True
            },
            {
                "key": "user_email",
                "value": "admin@example.com",
                "type": "default",
                "enabled": True
            },
            {
                "key": "user_password",
                "value": "SecurePassword123!",
                "type": "secret",
                "enabled": True
            }
        ],
        "_postman_variable_scope": "environment"
    }
    return environment

if __name__ == "__main__":
    col = create_collection()
    with open("postman/chat-agent-bots-api.postman_collection.json", "w", encoding="utf-8") as f:
        json.dump(col, f, indent=2)
    print("Generated postman/chat-agent-bots-api.postman_collection.json")

    env = create_environment()
    with open("postman/chat-agent-bots-local.postman_environment.json", "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2)
    print("Generated postman/chat-agent-bots-local.postman_environment.json")
