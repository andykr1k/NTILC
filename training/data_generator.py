"""
Synthetic data generator for tool invocation training data.
Supports both Python-style and JSON format outputs.
"""

import random
import json
from typing import List, Dict, Any, Optional
from faker import Faker
from datetime import datetime, timedelta
from enum import Enum

from .config import DataGeneratorConfig


class OutputFormat(Enum):
    """Output format for tool calls."""
    PYTHON = "python"  # search(query='...', max_results=10)
    JSON = "json"      # {"tool": "search", "arguments": {...}}


class ToolInvocationGenerator:
    """
    Generates synthetic tool invocation strings for training.
    
    Supports multiple tool types with realistic parameter distributions.
    Can output in Python-style or JSON format.
    """

    def __init__(self, config: DataGeneratorConfig = None, output_format: OutputFormat = OutputFormat.JSON):
        """
        Args:
            config: Configuration for data generation
            output_format: Format for tool call output (PYTHON or JSON)
        """
        self.config = config or DataGeneratorConfig()
        self.output_format = output_format
        self.faker = Faker()
        Faker.seed(42)  # For reproducibility
        random.seed(42)

    def _format_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Format a tool call in the configured output format.
        
        Args:
            tool_name: Name of the tool
            arguments: Dictionary of arguments
            
        Returns:
            Formatted tool call string
        """
        if self.output_format == OutputFormat.JSON:
            return json.dumps({
                "tool": tool_name,
                "arguments": arguments
            }, ensure_ascii=False)
        else:  # Python format
            args_str = ", ".join([
                f"{k}={self._format_python_value(v)}" 
                for k, v in arguments.items()
            ])
            return f"{tool_name}({args_str})"
    
    def _format_python_value(self, value: Any) -> str:
        """Format a value for Python-style output."""
        if isinstance(value, str):
            return f"'{value}'"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted = [self._format_python_value(v) for v in value]
            return "[" + ", ".join(formatted) + "]"
        elif isinstance(value, dict):
            # Handle nested dicts like DateRange
            if "from_date" in value and "to_date" in value:
                return f"DateRange(from_date='{value['from_date']}', to_date='{value['to_date']}')"
            return str(value)
        return str(value)

    def generate_search(self) -> str:
        """Generate search(query, max_results, date_filter) invocation."""
        topics = [
            "machine learning", "quantum computing", "recent AI papers",
            "Python tutorials", "climate change", "renewable energy",
            "neural networks", "natural language processing", "deep learning",
            "computer vision", "data science", "web development",
            "cloud computing", "cybersecurity", "blockchain technology",
            "software engineering", "database design", "API development",
            "mobile app development", "DevOps practices", "containerization",
            "microservices architecture", "agile methodology", "test-driven development",
            "code review best practices", "version control systems", "CI/CD pipelines",
            "cloud infrastructure", "serverless computing", "edge computing",
            "IoT devices", "robotics", "augmented reality", "virtual reality",
            "game development", "UI/UX design", "frontend frameworks",
            "backend architecture", "distributed systems", "system design patterns"
        ]

        topic = random.choice(topics)
        max_results = random.randint(self.config.min_max_results, self.config.max_max_results)

        arguments = {
            "query": topic,
            "max_results": max_results
        }

        # Optional date filter (30% chance)
        if random.random() < 0.3:
            days_ago = random.randint(1, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            arguments["date_filter"] = {
                "from_date": date_from,
                "to_date": date_to
            }

        return self._format_tool_call("search", arguments)

    def generate_calculate(self) -> str:
        """Generate calculate(expression) invocation."""
        expressions = [
            f"{random.randint(1, 100)} + {random.randint(1, 100)}",
            f"{random.randint(1, 50)} * {random.randint(1, 50)}",
            f"{random.randint(1, 100)} / {random.randint(1, 20)}",
            f"sqrt({random.randint(1, 10000)})",
            f"pow({random.randint(1, 10)}, {random.randint(1, 5)})",
            f"sin({random.uniform(0, 6.28):.2f})",
            f"cos({random.uniform(0, 6.28):.2f})",
            f"tan({random.uniform(0, 3.14):.2f})",
            f"log({random.randint(1, 1000)})",
            f"ln({random.randint(1, 100)})",
            f"({random.randint(1, 100)} + {random.randint(1, 100)}) * {random.randint(1, 10)}",
            f"({random.randint(1, 50)} - {random.randint(1, 50)}) / {random.randint(1, 10)}",
            f"{random.randint(1, 100)} ** {random.randint(1, 5)}",
            f"abs({random.randint(-100, 100)})",
            f"ceil({random.uniform(1, 100):.2f})",
            f"floor({random.uniform(1, 100):.2f})",
            f"round({random.uniform(1, 100):.2f})",
            "2 + 2", "10 * 5", "sqrt(144)", "sin(pi/2)", "cos(0)",
            "tan(pi/4)", "log(100)", "ln(e)", "2^10", "(5 + 3) * 2",
            "(10 - 4) / 2", "abs(-42)", "ceil(3.7)", "floor(3.7)", "round(3.14159)"
        ]
        
        expression = random.choice(expressions)
        return self._format_tool_call("calculate", {"expression": expression})

    def generate_database_query(self) -> str:
        """Generate database_query(sql, timeout) invocation."""
        tables = ["users", "orders", "products", "transactions", "logs", "events",
                  "customers", "payments", "inventory", "sessions", "notifications",
                  "reviews", "categories", "tags", "comments", "posts"]
        columns = ["id", "name", "email", "created_at", "amount", "status",
                   "username", "phone", "address", "city", "country",
                   "price", "quantity", "total", "discount", "tax", "shipping",
                   "title", "description", "content", "author", "published_at"]

        query_types = [
            f"SELECT * FROM {random.choice(tables)} LIMIT {random.randint(10, 100)}",
            f"SELECT {', '.join(random.sample(columns, random.randint(1, 3)))} FROM {random.choice(tables)} LIMIT {random.randint(10, 100)}",
            f"SELECT COUNT(*) FROM {random.choice(tables)}",
            f"SELECT * FROM {random.choice(tables)} WHERE created_at > '2024-01-01' LIMIT {random.randint(10, 100)}",
            f"SELECT * FROM {random.choice(tables)} WHERE status = 'active' LIMIT {random.randint(10, 100)}",
            f"SELECT * FROM {random.choice(tables)} WHERE {random.choice(columns)} IS NOT NULL LIMIT {random.randint(10, 100)}",
            f"SELECT {', '.join(random.sample(columns, random.randint(1, 2)))} FROM {random.choice(tables)} ORDER BY {random.choice(columns)} DESC LIMIT {random.randint(10, 100)}",
            f"SELECT AVG({random.choice(['amount', 'price', 'quantity', 'total'])}) FROM {random.choice(tables)}",
            f"SELECT SUM({random.choice(['amount', 'price', 'quantity', 'total'])}) FROM {random.choice(tables)}",
        ]

        sql = random.choice(query_types)
        timeout = random.randint(10, 60)

        return self._format_tool_call("database_query", {"sql": sql, "timeout": timeout})

    def generate_send_email(self) -> str:
        """Generate send_email(to, subject, body, cc) invocation."""
        to_email = self.faker.email()

        subject_templates = [
            "Meeting reminder for next week", "Project update and status report",
            "Follow-up on our conversation", "Request for information",
            "Thank you for your inquiry", "Action required: Review needed",
            "Weekly status update", "Important announcement",
            "Quarterly review meeting", "Budget approval request",
            "Team collaboration invitation", "Deadline extension notification",
            "New feature release announcement", "Security update required",
            "Performance metrics summary", "Client feedback and response",
            "Training session invitation", "System maintenance schedule",
            "Code review request", "Documentation update notice"
        ]
        subject = random.choice(subject_templates)

        body_templates = [
            "Please review the attached document and provide feedback by end of week.",
            "I wanted to follow up on our previous conversation about the project timeline.",
            "Thank you for your interest. I'll get back to you with more details soon.",
            "This is a reminder about the upcoming meeting scheduled for next Monday.",
            "I hope this email finds you well. I wanted to discuss the recent changes.",
            "Please find the requested information below. Let me know if you need anything else.",
            "I'm writing to confirm the details we discussed earlier today.",
            "Could you please provide an update on the status of this item?",
            "I wanted to reach out regarding the proposal we discussed last week.",
            "Please let me know if you have any questions or concerns about this matter."
        ]
        body = random.choice(body_templates)

        arguments = {
            "to": to_email,
            "subject": subject,
            "body": body
        }

        # Optional CC (30% chance)
        if random.random() < 0.3:
            cc_emails = [self.faker.email() for _ in range(random.randint(1, 2))]
            arguments["cc"] = cc_emails

        return self._format_tool_call("send_email", arguments)

    def generate_web_fetch(self) -> str:
        """Generate web_fetch(url, method) invocation."""
        urls = [
            "https://api.github.com/users", "https://jsonplaceholder.typicode.com/posts",
            "https://api.example.com/data", "https://api.github.com/repos",
            "https://jsonplaceholder.typicode.com/users", "https://api.example.com/endpoint",
            "https://api.github.com/orgs", "https://api.github.com/issues",
            "https://jsonplaceholder.typicode.com/comments", "https://jsonplaceholder.typicode.com/albums",
            "https://api.github.com/pulls", "https://api.github.com/commits",
            "https://api.example.com/v1/users", "https://api.example.com/v1/products",
            "https://api.example.com/v1/orders", "https://httpbin.org/get",
            "https://httpbin.org/post", "https://httpbin.org/json",
            "https://api.openweathermap.org/data/2.5/weather", "https://api.spotify.com/v1/albums"
        ]

        method = "POST" if random.random() < 0.2 else "GET"
        url = random.choice(urls)

        return self._format_tool_call("web_fetch", {"url": url, "method": method})

    def generate_file_read(self) -> str:
        """Generate file_read(path, encoding) invocation."""
        paths = [
            "/home/user/log.txt", "./config.json", "/var/log/app.log",
            "data/input.csv", "/tmp/output.log", "./data/readme.md",
            "../config/settings.ini", "/var/log/nginx/access.log",
            "/var/log/nginx/error.log", "/etc/nginx/nginx.conf",
            "/home/user/documents/report.pdf", "./src/main.py",
            "./tests/test_suite.py", "data/dataset.json",
            "data/training_data.csv", "data/validation_data.csv",
            "logs/application.log", "logs/error.log", "logs/debug.log",
            "/tmp/temp_file.txt", "/tmp/cache/data.bin", "./output/results.json",
            "./output/analysis.csv", "../shared/config.yaml", "../shared/secrets.env"
        ]

        path = random.choice(paths)
        encodings = ["utf-8", "ascii", "latin-1", "utf-16"]
        encoding = random.choice(encodings)

        return self._format_tool_call("file_read", {"path": path, "encoding": encoding})

    def generate_tool_call(self, tool_name: str = None) -> str:
        """
        Generate a random tool invocation.

        Args:
            tool_name: Specific tool to generate, or None for random

        Returns:
            tool_call: Tool invocation string
        """
        if tool_name is None:
            tool_name = random.choice(self.config.tools)

        generator_map = {
            "search": self.generate_search,
            "calculate": self.generate_calculate,
            "database_query": self.generate_database_query,
            "send_email": self.generate_send_email,
            "web_fetch": self.generate_web_fetch,
            "file_read": self.generate_file_read
        }

        if tool_name not in generator_map:
            raise ValueError(f"Unknown tool: {tool_name}")

        return generator_map[tool_name]()

    def generate_dataset(self, num_samples: int) -> List[str]:
        """
        Generate a dataset of tool invocations.

        Args:
            num_samples: Number of samples to generate

        Returns:
            tool_calls: List of tool invocation strings
        """
        tool_calls = []

        # Ensure balanced distribution across tools
        samples_per_tool = num_samples // len(self.config.tools)
        remainder = num_samples % len(self.config.tools)

        for i, tool in enumerate(self.config.tools):
            count = samples_per_tool + (1 if i < remainder else 0)
            for _ in range(count):
                tool_calls.append(self.generate_tool_call(tool))

        # Shuffle
        random.shuffle(tool_calls)

        return tool_calls

    def save_dataset(self, tool_calls: List[str], filepath: str):
        """
        Save dataset to file.

        Args:
            tool_calls: List of tool invocation strings
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            for tool_call in tool_calls:
                f.write(tool_call + '\n')

    def load_dataset(self, filepath: str) -> List[str]:
        """
        Load dataset from file.

        Args:
            filepath: Path to load file from

        Returns:
            tool_calls: List of tool invocation strings
        """
        with open(filepath, 'r') as f:
            tool_calls = [line.strip() for line in f if line.strip()]
        return tool_calls


class NaturalLanguageToolCallGenerator:
    """
    Generates (natural_language_query, tool_call) pairs for LLM training.
    This is used in Phase 2 to train the LLM to predict tool embeddings.
    """
    
    def __init__(self, output_format: OutputFormat = OutputFormat.JSON):
        """
        Args:
            output_format: Format for tool call output
        """
        self.output_format = output_format
        self.faker = Faker()
        Faker.seed(42)
        random.seed(42)
        self.tool_generator = ToolInvocationGenerator(output_format=output_format)
    
    def _format_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Format tool call using the parent generator's method."""
        return self.tool_generator._format_tool_call(tool_name, arguments)
    
    def generate_search_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding search tool call."""
        topics = [
            "machine learning", "quantum computing", "AI papers",
            "Python tutorials", "climate change", "renewable energy",
            "neural networks", "NLP", "deep learning", "data science"
        ]
        
        topic = random.choice(topics)
        max_results = random.randint(5, 50)
        
        # Various ways users might phrase the request
        query_templates = [
            f"Find me information about {topic}, show {max_results} results",
            f"Search for {topic} and return up to {max_results} items",
            f"I need to research {topic}, give me {max_results} results",
            f"Look up {topic} for me, limit to {max_results}",
            f"What can you find about {topic}? Show me {max_results} results",
            f"Can you search for {topic}? I want {max_results} results maximum",
            f"Help me find articles about {topic}, {max_results} results please",
            f"Query {topic} and get {max_results} results",
        ]
        
        nl_query = random.choice(query_templates)
        
        arguments = {"query": topic, "max_results": max_results}
        
        # 30% chance of date filter
        if random.random() < 0.3:
            days_ago = random.randint(7, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            arguments["date_filter"] = {"from_date": date_from, "to_date": date_to}
            
            # Update NL query to mention date
            date_phrases = [
                f" from the last {days_ago} days",
                f" since {date_from}",
                f" between {date_from} and {date_to}",
            ]
            nl_query += random.choice(date_phrases)
        
        return {
            "query": nl_query,
            "tool_call": self._format_tool_call("search", arguments),
            "tool": "search"
        }
    
    def generate_calculate_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding calculate tool call."""
        # Generate various math problems with NL descriptions
        problems = [
            ("What is 15% tip on $47.50?", "47.50 * 0.15"),
            ("Calculate the square root of 256", "sqrt(256)"),
            ("What's 7 times 8?", "7 * 8"),
            ("Divide 144 by 12", "144 / 12"),
            ("What is 2 to the power of 10?", "2 ** 10"),
            ("Calculate sine of pi/4", "sin(pi/4)"),
            ("What's the natural log of 100?", "ln(100)"),
            ("Round 3.14159 to nearest integer", "round(3.14159)"),
            ("What is the absolute value of -42?", "abs(-42)"),
            ("Calculate 25 plus 37", "25 + 37"),
            ("What's 100 minus 67?", "100 - 67"),
            ("Compute the floor of 7.8", "floor(7.8)"),
            ("What is the ceiling of 3.2?", "ceil(3.2)"),
            ("Calculate 15 squared", "15 ** 2"),
            ("What is cosine of 0?", "cos(0)"),
        ]
        
        # Add dynamic problems
        a, b = random.randint(1, 100), random.randint(1, 100)
        dynamic_problems = [
            (f"What is {a} plus {b}?", f"{a} + {b}"),
            (f"Calculate {a} times {b}", f"{a} * {b}"),
            (f"What's {a} divided by {max(1, b//10)}?", f"{a} / {max(1, b//10)}"),
            (f"Compute {a} minus {b}", f"{a} - {b}"),
        ]
        
        all_problems = problems + dynamic_problems
        nl_query, expression = random.choice(all_problems)
        
        return {
            "query": nl_query,
            "tool_call": self._format_tool_call("calculate", {"expression": expression}),
            "tool": "calculate"
        }
    
    def generate_database_query_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding database_query tool call."""
        # Various database query scenarios
        scenarios = [
            {
                "nl": "Get the last 10 orders from California",
                "sql": "SELECT * FROM orders WHERE state = 'CA' ORDER BY created_at DESC LIMIT 10",
            },
            {
                "nl": "Find all users who signed up this month",
                "sql": "SELECT * FROM users WHERE created_at >= DATE_TRUNC('month', CURRENT_DATE) LIMIT 100",
            },
            {
                "nl": "Count how many active products we have",
                "sql": "SELECT COUNT(*) FROM products WHERE status = 'active'",
            },
            {
                "nl": "Get total revenue from last quarter",
                "sql": "SELECT SUM(amount) FROM transactions WHERE created_at >= DATE_TRUNC('quarter', CURRENT_DATE - INTERVAL '3 months')",
            },
            {
                "nl": "List all customers from New York",
                "sql": "SELECT * FROM customers WHERE state = 'NY' LIMIT 50",
            },
            {
                "nl": "Find the top 5 selling products",
                "sql": "SELECT * FROM products ORDER BY sales_count DESC LIMIT 5",
            },
            {
                "nl": "Get all pending orders",
                "sql": "SELECT * FROM orders WHERE status = 'pending' LIMIT 100",
            },
            {
                "nl": "Find users with email containing gmail",
                "sql": "SELECT * FROM users WHERE email LIKE '%gmail%' LIMIT 50",
            },
            {
                "nl": "Get average order value",
                "sql": "SELECT AVG(total) FROM orders",
            },
            {
                "nl": "List recent transactions over $1000",
                "sql": "SELECT * FROM transactions WHERE amount > 1000 ORDER BY created_at DESC LIMIT 20",
            },
        ]
        
        scenario = random.choice(scenarios)
        timeout = random.randint(10, 60)
        
        # Add timeout variation to NL query sometimes
        nl_query = scenario["nl"]
        if random.random() < 0.3:
            nl_query += f" (timeout: {timeout}s)"
        
        return {
            "query": nl_query,
            "tool_call": self._format_tool_call("database_query", {
                "sql": scenario["sql"],
                "timeout": timeout
            }),
            "tool": "database_query"
        }
    
    def generate_send_email_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding send_email tool call."""
        email = self.faker.email()
        
        scenarios = [
            {
                "nl": f"Send an email to {email} about the meeting tomorrow",
                "subject": "Meeting Tomorrow",
                "body": "This is a reminder about our meeting scheduled for tomorrow. Please confirm your attendance."
            },
            {
                "nl": f"Email {email} to follow up on the project",
                "subject": "Project Follow-up",
                "body": "I wanted to follow up on the project status. Please provide an update when you have a chance."
            },
            {
                "nl": f"Send a thank you email to {email}",
                "subject": "Thank You",
                "body": "Thank you for your time and assistance. I really appreciate your help."
            },
            {
                "nl": f"Notify {email} about the deadline extension",
                "subject": "Deadline Extension Notice",
                "body": "The deadline has been extended. Please review the new timeline and adjust accordingly."
            },
            {
                "nl": f"Email {email} requesting feedback on the proposal",
                "subject": "Feedback Request - Proposal",
                "body": "Please review the attached proposal and provide your feedback by end of week."
            },
        ]
        
        scenario = random.choice(scenarios)
        
        arguments = {
            "to": email,
            "subject": scenario["subject"],
            "body": scenario["body"]
        }
        
        # 20% chance of CC
        if random.random() < 0.2:
            cc_email = self.faker.email()
            arguments["cc"] = [cc_email]
        
        return {
            "query": scenario["nl"],
            "tool_call": self._format_tool_call("send_email", arguments),
            "tool": "send_email"
        }
    
    def generate_web_fetch_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding web_fetch tool call."""
        scenarios = [
            ("Fetch the GitHub users API", "https://api.github.com/users", "GET"),
            ("Get posts from JSONPlaceholder", "https://jsonplaceholder.typicode.com/posts", "GET"),
            ("Fetch weather data from OpenWeatherMap", "https://api.openweathermap.org/data/2.5/weather", "GET"),
            ("Get data from the example API", "https://api.example.com/data", "GET"),
            ("Submit data to the webhook endpoint", "https://api.example.com/webhook", "POST"),
            ("Fetch the latest GitHub issues", "https://api.github.com/issues", "GET"),
            ("Get comments from JSONPlaceholder", "https://jsonplaceholder.typicode.com/comments", "GET"),
            ("Send data to the analytics endpoint", "https://api.example.com/analytics", "POST"),
        ]
        
        nl_query, url, method = random.choice(scenarios)
        
        return {
            "query": nl_query,
            "tool_call": self._format_tool_call("web_fetch", {"url": url, "method": method}),
            "tool": "web_fetch"
        }
    
    def generate_file_read_pair(self) -> Dict[str, str]:
        """Generate NL query and corresponding file_read tool call."""
        scenarios = [
            ("Read the config file", "./config.json", "utf-8"),
            ("Open the log file", "/var/log/app.log", "utf-8"),
            ("Read the CSV data file", "data/input.csv", "utf-8"),
            ("Get contents of the readme", "./README.md", "utf-8"),
            ("Read the settings file", "../config/settings.ini", "utf-8"),
            ("Open the nginx access log", "/var/log/nginx/access.log", "utf-8"),
            ("Read the test suite", "./tests/test_suite.py", "utf-8"),
            ("Get the dataset JSON", "data/dataset.json", "utf-8"),
            ("Read the error log with latin-1 encoding", "logs/error.log", "latin-1"),
        ]
        
        nl_query, path, encoding = random.choice(scenarios)
        
        return {
            "query": nl_query,
            "tool_call": self._format_tool_call("file_read", {"path": path, "encoding": encoding}),
            "tool": "file_read"
        }
    
    def generate_pair(self, tool_name: str = None) -> Dict[str, str]:
        """Generate a random (NL query, tool call) pair."""
        generators = {
            "search": self.generate_search_pair,
            "calculate": self.generate_calculate_pair,
            "database_query": self.generate_database_query_pair,
            "send_email": self.generate_send_email_pair,
            "web_fetch": self.generate_web_fetch_pair,
            "file_read": self.generate_file_read_pair,
        }
        
        if tool_name is None:
            tool_name = random.choice(list(generators.keys()))
        
        return generators[tool_name]()
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate a balanced dataset of (NL query, tool call) pairs."""
        tools = list(["search", "calculate", "database_query", "send_email", "web_fetch", "file_read"])
        samples_per_tool = num_samples // len(tools)
        remainder = num_samples % len(tools)
        
        dataset = []
        for i, tool in enumerate(tools):
            count = samples_per_tool + (1 if i < remainder else 0)
            for _ in range(count):
                dataset.append(self.generate_pair(tool))
        
        random.shuffle(dataset)
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, str]], filepath: str):
        """Save dataset to JSONL file."""
        with open(filepath, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def load_dataset(self, filepath: str) -> List[Dict[str, str]]:
        """Load dataset from JSONL file."""
        dataset = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip():
                    dataset.append(json.loads(line))
        return dataset
