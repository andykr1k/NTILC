"""
Tool schema definitions for ablation studies.
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
                json.dumps({"tool": "calculate", "arguments": {"expression": "sqrt(144)"}}),
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
                json.dumps({"tool": "web_fetch", "arguments": {"url": "https://api.github.com/users", "method": "GET"}}),
                json.dumps({"tool": "web_fetch", "arguments": {"url": "https://api.example.com/data", "method": "POST"}})
            ],
            "file_read": [
                json.dumps({"tool": "file_read", "arguments": {"path": "/home/user/log.txt", "encoding": "utf-8"}}),
                json.dumps({"tool": "file_read", "arguments": {"path": "./config.json", "encoding": "utf-8"}})
            ]
        }
    else:
        # Python format (legacy)
        return {
            "search": [
                "search(query='machine learning', max_results=10)",
                "search(query='quantum computing papers', max_results=5, date_filter=DateRange(from_date='2024-01-01', to_date='2024-12-31'))"
            ],
            "calculate": [
                "calculate(expression='2 + 2')",
                "calculate(expression='sqrt(144)')",
                "calculate(expression='sin(3.14159)')"
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
                "web_fetch(url='https://api.github.com/users', method='GET')",
                "web_fetch(url='https://api.example.com/data', method='POST')"
            ],
            "file_read": [
                "file_read(path='/home/user/log.txt', encoding='utf-8')",
                "file_read(path='./config.json', encoding='utf-8')"
            ]
        }


def extract_tool_call(text: str, expected_format: OutputFormat = OutputFormat.JSON) -> str:
    """
    Extract tool call from generated text.
    Handles both JSON and Python formats.
    
    Args:
        text: Generated text that may contain tool call
        expected_format: Expected format of the tool call
        
    Returns:
        Extracted tool call string
    """
    text = text.strip()
    
    # Try to extract JSON format
    if expected_format == OutputFormat.JSON:
        # Find JSON object in text
        start = text.find('{')
        if start != -1:
            # Find matching closing brace
            depth = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = text[start:i+1]
                        # Validate it's valid JSON with expected structure
                        try:
                            parsed = json.loads(candidate)
                            if "tool" in parsed:
                                return candidate
                        except json.JSONDecodeError:
                            pass
                        break
        
        # Fallback: try to parse the whole text as JSON
        try:
            parsed = json.loads(text)
            if "tool" in parsed:
                return text
        except json.JSONDecodeError:
            pass
    
    # Try Python format extraction
    # Find tool_name(...)
    import re
    pattern = r'(\w+)\s*\([^)]*\)'
    match = re.search(pattern, text)
    if match:
        # Find the complete function call with balanced parens
        func_start = match.start()
        paren_start = text.find('(', func_start)
        if paren_start != -1:
            depth = 0
            in_string = False
            string_char = None
            escape_next = False
            
            for i in range(paren_start, len(text)):
                char = text[i]
                
                if escape_next:
                    escape_next = False
                    continue
                
                if char == '\\':
                    escape_next = True
                    continue
                
                if char in ('"', "'") and not escape_next:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                
                if not in_string:
                    if char == '(':
                        depth += 1
                    elif char == ')':
                        depth -= 1
                        if depth == 0:
                            return text[func_start:i+1].strip()
    
    # Last resort: return cleaned text
    # Remove common prefixes
    for prefix in ["Tool call:", "Output:", "Result:"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    
    return text.split('\n')[0].strip()


def extract_tool_name(tool_call: str) -> str:
    """
    Extract the tool name from a tool call string.
    Handles both JSON and Python formats.
    
    Args:
        tool_call: Tool call string
        
    Returns:
        Tool name or empty string if not found
    """
    tool_call = tool_call.strip()
    
    # Try JSON format
    if tool_call.startswith('{'):
        try:
            parsed = json.loads(tool_call)
            return parsed.get("tool", "")
        except json.JSONDecodeError:
            pass
    
    # Try Python format
    paren_idx = tool_call.find('(')
    if paren_idx > 0:
        return tool_call[:paren_idx].strip()
    
    return ""
