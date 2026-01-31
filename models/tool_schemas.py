"""
Tool schema definitions for NTILC.
Supports both Python-style and JSON format tool calls.
"""

import json
from enum import Enum
from typing import List, Dict


class OutputFormat(Enum):
    """Output format for tool calls."""
    PYTHON = "python"  # search(query='...', max_results=10)
    JSON = "json"      # {"tool": "search", "arguments": {...}}


TOOL_SCHEMAS = {
    "search": {
        "name": "search",
        "description": "Search for information using a query string",
        "parameters": {
            "query": {
                "type": "str",
                "description": "The search query string",
                "required": True
            },
            "max_results": {
                "type": "int",
                "description": "Maximum number of results to return",
                "required": True,
                "default": 10
            },
            "date_filter": {
                "type": "DateRange",
                "description": "Optional date range filter",
                "required": False
            }
        }
    },
    "calculate": {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "expression": {
                "type": "str",
                "description": "Mathematical expression to evaluate",
                "required": True
            }
        }
    },
    "database_query": {
        "name": "database_query",
        "description": "Execute a SQL query on the database",
        "parameters": {
            "sql": {
                "type": "str",
                "description": "SQL query string",
                "required": True
            },
            "timeout": {
                "type": "int",
                "description": "Query timeout in seconds",
                "required": True,
                "default": 30
            }
        }
    },
    "send_email": {
        "name": "send_email",
        "description": "Send an email message",
        "parameters": {
            "to": {
                "type": "email",
                "description": "Recipient email address",
                "required": True
            },
            "subject": {
                "type": "str",
                "description": "Email subject line",
                "required": True
            },
            "body": {
                "type": "str",
                "description": "Email body content",
                "required": True
            },
            "cc": {
                "type": "List[email]",
                "description": "Optional CC recipients",
                "required": False
            }
        }
    },
    "web_fetch": {
        "name": "web_fetch",
        "description": "Fetch content from a URL",
        "parameters": {
            "url": {
                "type": "str",
                "description": "URL to fetch",
                "required": True
            },
            "method": {
                "type": "enum",
                "description": "HTTP method",
                "required": True,
                "options": ["GET", "POST"],
                "default": "GET"
            }
        }
    },
    "file_read": {
        "name": "file_read",
        "description": "Read contents of a file",
        "parameters": {
            "path": {
                "type": "str",
                "description": "File path to read",
                "required": True
            },
            "encoding": {
                "type": "str",
                "description": "File encoding",
                "required": True,
                "default": "utf-8",
                "options": ["utf-8", "ascii", "latin-1", "utf-16"]
            }
        }
    }
}


def format_tools_for_prompt(output_format: OutputFormat = OutputFormat.JSON) -> str:
    """
    Format tool schemas as a prompt-friendly string.
    
    Args:
        output_format: Whether to show JSON or Python format examples
    
    Returns:
        Formatted string describing all available tools
    """
    lines = ["Available Tools:", ""]
    
    for tool_name, tool_schema in TOOL_SCHEMAS.items():
        lines.append(f"Tool: {tool_name}")
        lines.append(f"  Description: {tool_schema['description']}")
        lines.append("  Parameters:")
        
        for param_name, param_info in tool_schema["parameters"].items():
            required = param_info.get("required", False)
            param_type = param_info.get("type", "str")
            description = param_info.get("description", "")
            default = param_info.get("default", None)
            
            req_str = "required" if required else "optional"
            default_str = f" (default: {default})" if default is not None else ""
            lines.append(f"    - {param_name} ({param_type}, {req_str}): {description}{default_str}")
        
        lines.append("")
    
    # Add format hint
    if output_format == OutputFormat.JSON:
        lines.append("Output format: JSON")
        lines.append('Example: {"tool": "search", "arguments": {"query": "...", "max_results": 10}}')
    else:
        lines.append("Output format: Python function call")
        lines.append("Example: search(query='...', max_results=10)")
    
    return "\n".join(lines)


def get_tool_examples(output_format: OutputFormat = OutputFormat.JSON) -> Dict[str, List[str]]:
    """
    Get example tool calls for each tool type.
    
    Args:
        output_format: Format for examples (JSON or Python)
    
    Returns:
        Dictionary mapping tool names to example calls
    """
    if output_format == OutputFormat.JSON:
        return {
            "search": [
                json.dumps({"tool": "search", "arguments": {"query": "machine learning", "max_results": 10}}),
                json.dumps({"tool": "search", "arguments": {"query": "quantum computing papers", "max_results": 5, "date_filter": {"from_date": "2024-01-01", "to_date": "2024-12-31"}}})
            ],
            "calculate": [
                json.dumps({"tool": "calculate", "arguments": {"expression": "2 + 2"}}),
                json.dumps({"tool": "calculate", "arguments": {"expression": "sqrt(144)"}})
            ],
            "database_query": [
                json.dumps({"tool": "database_query", "arguments": {"sql": "SELECT * FROM users LIMIT 10", "timeout": 30}}),
                json.dumps({"tool": "database_query", "arguments": {"sql": "SELECT COUNT(*) FROM orders WHERE status = 'active'", "timeout": 60}})
            ],
            "send_email": [
                json.dumps({"tool": "send_email", "arguments": {"to": "user@example.com", "subject": "Hello", "body": "This is a test email"}}),
                json.dumps({"tool": "send_email", "arguments": {"to": "admin@example.com", "subject": "Report", "body": "Monthly report attached", "cc": ["manager@example.com"]}})
            ],
            "web_fetch": [
                json.dumps({"tool": "web_fetch", "arguments": {"url": "https://api.example.com/data", "method": "GET"}}),
                json.dumps({"tool": "web_fetch", "arguments": {"url": "https://api.example.com/data", "method": "POST"}})
            ],
            "file_read": [
                json.dumps({"tool": "file_read", "arguments": {"path": "/tmp/file.txt", "encoding": "utf-8"}}),
                json.dumps({"tool": "file_read", "arguments": {"path": "./data/input.csv", "encoding": "utf-8"}})
            ]
        }
    
    return {
        "search": [
            "search(query='machine learning', max_results=10)",
            "search(query='quantum computing papers', max_results=5, date_filter=DateRange(from_date='2024-01-01', to_date='2024-12-31'))"
        ],
        "calculate": [
            "calculate(expression='2 + 2')",
            "calculate(expression='sqrt(144)')"
        ],
        "database_query": [
            "database_query(sql='SELECT * FROM users LIMIT 10', timeout=30)",
            "database_query(sql='SELECT COUNT(*) FROM orders WHERE status = \\'active\\'', timeout=60)"
        ],
        "send_email": [
            "send_email(to='user@example.com', subject='Hello', body='This is a test email')",
            "send_email(to='admin@example.com', subject='Report', body='Monthly report attached', cc=['manager@example.com'])"
        ],
        "web_fetch": [
            "web_fetch(url='https://api.example.com/data', method='GET')",
            "web_fetch(url='https://api.example.com/data', method='POST')"
        ],
        "file_read": [
            "file_read(path='/tmp/file.txt', encoding='utf-8')",
            "file_read(path='./data/input.csv', encoding='utf-8')"
        ]
    }
