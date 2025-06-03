get_relevant_data_function_calling = {
    "name": "get_relevant_data",
    "description": "Retrieve the top 5 most relevant law articles based on the input question.",
    "parameters": {
        "type": "object",
        "properties": {
            "articles": {
                "type": "array",
                "description": "List of the 5 most relevant law articles matching the input question.",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": "Unique identifier for the article (e.g., 'dieu_1', 'dieu_234')."
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the law article."
                        },
                        "text": {
                            "type": "string",
                            "description": "Full textual content of the law article."
                        }
                    },
                    "required": ["key", "title", "text"]
                }
            },
            "reasoning": {
                "type": "string",
                "description": "Explain the reason why you extract those values"
            }
        },
        "required": ["articles", "reasoning"]
    }
}

rewrite_question_function_calling = {
    "name": "rewrite_question",
    "description": "Rewrite the input question more clearly",
    "parameters": {
        "type": "object",
        "properties": {
            "rewrote_question": {
                "type": "string",
                "description": "The rewrote question response"
            },
            "reasoning": {
                "type": "string",
                "description": "Explain the reason why you extract those values"
            }
        },
        "required": ["rewrote_question", "reasoning"]
    }
}
