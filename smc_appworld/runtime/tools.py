"""Tool definitions for AppWorld in OpenAI tool-calling format.

Instead of 453 individual tools, we use a small set of meta-tools:
- Discovery tools: show_app_descriptions, show_api_descriptions, show_api_doc, search_api_docs
- Execution tool: execute_api (generic, calls any AppWorld API)
- Supervisor shortcuts: show_account_passwords, show_active_task, complete_task, show_profile
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_app_descriptions",
            "description": "Show descriptions of all available apps. Call this first to understand what apps are available.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_api_descriptions",
            "description": "Show descriptions of all APIs in a specific app. Use this to discover what operations are available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app (e.g., 'spotify', 'gmail', 'amazon').",
                    }
                },
                "required": ["app_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_api_doc",
            "description": "Show the full documentation of a specific API, including parameters and response schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app.",
                    },
                    "api_name": {
                        "type": "string",
                        "description": "Name of the API.",
                    },
                },
                "required": ["app_name", "api_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_api_docs",
            "description": "Search API documentation across all apps with a natural language query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'send email', 'search songs').",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_api",
            "description": "Execute an API call on a specific app. First use show_api_doc to understand the required parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_name": {
                        "type": "string",
                        "description": "Name of the app (e.g., 'spotify', 'gmail').",
                    },
                    "api_name": {
                        "type": "string",
                        "description": "Name of the API to call (e.g., 'login', 'search_songs').",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "API parameters as key-value pairs. Check show_api_doc for required parameters.",
                        "additionalProperties": True,
                    },
                },
                "required": ["app_name", "api_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_account_passwords",
            "description": "Show your supervisor's app account passwords. Use this to get login credentials for apps.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_active_task",
            "description": "Show the current task description that you need to complete.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_task",
            "description": "Mark the task as complete and submit the final output/answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output": {
                        "type": "string",
                        "description": "The final answer or output for the task.",
                    }
                },
                "required": ["output"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_profile",
            "description": "Show your supervisor's personal profile information.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_addresses",
            "description": "Show your supervisor's saved addresses.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_payment_cards",
            "description": "Show your supervisor's saved payment cards.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]


def get_tool_names():
    """Return list of tool names."""
    return [t["function"]["name"] for t in TOOLS]
