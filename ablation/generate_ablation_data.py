"""
Generate test data for ablation studies: conversation context + tool calls.
Supports both Python-style and JSON format outputs.
"""

import json
import random
from typing import List, Dict
from faker import Faker
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum


class OutputFormat(Enum):
    """Output format for tool calls."""
    PYTHON = "python"  # search(query='...', max_results=10)
    JSON = "json"      # {"tool": "search", "arguments": {...}}


class AblationDataGenerator:
    """
    Generate conversation queries that should trigger tool calls.
    Supports both Python-style and JSON output formats.
    """
    
    def __init__(self, output_format: OutputFormat = OutputFormat.JSON):
        """
        Args:
            output_format: Format for tool call output (PYTHON or JSON)
        """
        self.output_format = output_format
        self.faker = Faker()
        Faker.seed(42)
        random.seed(42)
    
    def _format_tool_call(self, tool_name: str, arguments: Dict) -> str:
        """Format a tool call in the configured output format."""
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
    
    def _format_python_value(self, value) -> str:
        """Format a value for Python-style output."""
        if isinstance(value, str):
            escaped = value.replace("'", "\\'").replace("\n", "\\n")
            return f"'{escaped}'"
        elif isinstance(value, bool):
            return str(value)
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, list):
            formatted = [self._format_python_value(v) for v in value]
            return "[" + ", ".join(formatted) + "]"
        elif isinstance(value, dict):
            if "from_date" in value and "to_date" in value:
                return f"DateRange(from_date='{value['from_date']}', to_date='{value['to_date']}')"
            return str(value)
        return str(value)
    
    def generate_search_query(self) -> Dict[str, str]:
        """Generate a query that should trigger search tool."""
        topics = [
            "machine learning", "quantum computing", "recent AI papers",
            "Python tutorials", "climate change", "renewable energy",
            "neural networks", "natural language processing"
        ]
        
        topic = random.choice(topics)
        max_results = random.randint(5, 50)
        
        has_date_filter = random.random() < 0.3
        if has_date_filter:
            days_ago = random.randint(1, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
        
        # Generate query templates
        if has_date_filter:
            queries = [
                f"Find me information about {topic} from {date_from} to {date_to}, return up to {max_results} results",
                f"Search for {topic} between {date_from} and {date_to}, limit to {max_results} results",
                f"I need to look up {topic} from {date_from} to {date_to}, show me {max_results} results",
            ]
            arguments = {
                "query": topic,
                "max_results": max_results,
                "date_filter": {"from_date": date_from, "to_date": date_to}
            }
        else:
            queries = [
                f"Find me information about {topic}, return up to {max_results} results",
                f"Search for {topic}, limit to {max_results} results",
                f"I need to look up {topic}, show me {max_results} results",
                f"Can you search for {topic}? Return {max_results} results",
            ]
            arguments = {"query": topic, "max_results": max_results}
        
        user_query = random.choice(queries)
        tool_call = self._format_tool_call("search", arguments)
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "search"
        }
    
    def generate_calculate_query(self) -> Dict[str, str]:
        """Generate a query that should trigger calculate tool."""
        query_templates = [
            "Calculate {expression}",
            "What is {expression}?",
            "Compute {expression}",
            "Evaluate {expression}",
            "Solve {expression}"
        ]
        
        expressions = [
            "2 + 2", "10 * 5", "sqrt(144)", "sin(pi/2)",
            "log(100)", "2^10", "(5 + 3) * 2"
        ]
        
        query_template = random.choice(query_templates)
        expression = random.choice(expressions)
        user_query = query_template.format(expression=expression)
        
        tool_call = self._format_tool_call("calculate", {"expression": expression})
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "calculate"
        }
    
    def generate_database_query(self) -> Dict[str, str]:
        """Generate a query that should trigger database_query tool."""
        what_to_sql = {
            "all users": "SELECT * FROM users LIMIT {limit}",
            "recent orders": "SELECT * FROM orders WHERE created_at > '2024-01-01' LIMIT {limit}",
            "product information": "SELECT * FROM products LIMIT {limit}",
            "transaction logs": "SELECT * FROM transactions LIMIT {limit}",
            "user count": "SELECT COUNT(*) FROM users",
            "active records": "SELECT * FROM {table} WHERE status = 'active' LIMIT {limit}"
        }
        
        what = random.choice(list(what_to_sql.keys()))
        limit = random.randint(10, 100)
        timeout = random.randint(10, 60)
        
        sql_template = what_to_sql[what]
        
        if "{table}" in sql_template:
            tables = ["users", "orders", "products", "transactions"]
            table = random.choice(tables)
            sql = sql_template.format(table=table, limit=limit)
        else:
            sql = sql_template.format(limit=limit)
        
        queries = [
            f"Query the database for {what} with a timeout of {timeout} seconds",
            f"Get {what} from the database, timeout {timeout}s",
            f"Fetch {what} from the database with {timeout}s timeout",
        ]
        
        user_query = random.choice(queries)
        tool_call = self._format_tool_call("database_query", {"sql": sql, "timeout": timeout})
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "database_query"
        }
    
    def generate_send_email_query(self) -> Dict[str, str]:
        """Generate a query that should trigger send_email tool."""
        body_templates = [
            "Please review the attached document and provide feedback by end of week.",
            "I wanted to follow up on our previous conversation about the project timeline.",
            "Thank you for your interest. I'll get back to you with more details soon.",
            "This is a reminder about the upcoming meeting scheduled for next Monday.",
        ]
        
        recipient = self.faker.email()
        subject = self.faker.sentence(nb_words=random.randint(3, 6)).strip()
        body = random.choice(body_templates)
        
        has_cc = random.random() < 0.3
        if has_cc:
            cc_emails = [self.faker.email() for _ in range(random.randint(1, 2))]
            cc_str = ", ".join(cc_emails)
        
        if has_cc:
            queries = [
                f"Send an email to {recipient} with subject '{subject}' and body '{body}'. CC: {cc_str}",
                f"Email {recipient} with subject '{subject}' saying '{body}'. Also CC {cc_str}",
            ]
            arguments = {"to": recipient, "subject": subject, "body": body, "cc": cc_emails}
        else:
            queries = [
                f"Send an email to {recipient} with subject '{subject}' and body '{body}'",
                f"Email {recipient} with subject '{subject}' saying '{body}'",
            ]
            arguments = {"to": recipient, "subject": subject, "body": body}
        
        user_query = random.choice(queries)
        tool_call = self._format_tool_call("send_email", arguments)
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "send_email"
        }
    
    def generate_web_fetch_query(self) -> Dict[str, str]:
        """Generate a query that should trigger web_fetch tool."""
        get_queries = ["Fetch data from {url}", "Get content from {url}", "Retrieve {url}"]
        post_queries = ["POST data to {url}", "Send data to {url}"]
        
        urls = [
            "https://api.github.com/users",
            "https://jsonplaceholder.typicode.com/posts",
            "https://api.example.com/data"
        ]
        
        if random.random() < 0.2:
            query_template = random.choice(post_queries)
            method = "POST"
        else:
            query_template = random.choice(get_queries)
            method = "GET"
        
        url = random.choice(urls)
        user_query = query_template.format(url=url)
        tool_call = self._format_tool_call("web_fetch", {"url": url, "method": method})
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "web_fetch"
        }
    
    def generate_file_read_query(self) -> Dict[str, str]:
        """Generate a query that should trigger file_read tool."""
        paths = ["/home/user/log.txt", "./config.json", "/var/log/app.log", "data/input.csv"]
        encodings = ["utf-8", "ascii", "latin-1", "utf-16"]
        
        path = random.choice(paths)
        encoding = random.choice(encodings)
        
        queries = [
            f"Read the file at {path} using {encoding} encoding",
            f"Open {path} with {encoding} encoding",
            f"Get contents of {path}, read as {encoding}",
        ]
        
        user_query = random.choice(queries)
        tool_call = self._format_tool_call("file_read", {"path": path, "encoding": encoding})
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "file_read"
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, str]]:
        """Generate balanced dataset across all tools."""
        generators = {
            "search": self.generate_search_query,
            "calculate": self.generate_calculate_query,
            "database_query": self.generate_database_query,
            "send_email": self.generate_send_email_query,
            "web_fetch": self.generate_web_fetch_query,
            "file_read": self.generate_file_read_query
        }
        
        samples_per_tool = num_samples // len(generators)
        remainder = num_samples % len(generators)
        
        dataset = []
        for i, (tool, generator) in enumerate(generators.items()):
            count = samples_per_tool + (1 if i < remainder else 0)
            for _ in range(count):
                dataset.append(generator())
        
        random.shuffle(dataset)
        return dataset
    
    def save_dataset(self, dataset: List[Dict[str, str]], filepath: str, format: str = "jsonl"):
        """Save dataset to file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, 'w') as f:
                for example in dataset:
                    f.write(json.dumps(example) + '\n')
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(dataset, f, indent=2)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ablation test data")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output", type=str, default="./data/ablation/test_data.jsonl")
    parser.add_argument("--format", type=str, choices=["jsonl", "json"], default="jsonl")
    parser.add_argument("--output_format", type=str, choices=["python", "json"], default="json",
                       help="Format for tool calls (python or json)")
    
    args = parser.parse_args()
    
    output_format = OutputFormat.JSON if args.output_format == "json" else OutputFormat.PYTHON
    
    print(f"Generating ablation test data in {args.output_format} format...")
    generator = AblationDataGenerator(output_format=output_format)
    dataset = generator.generate_dataset(args.num_samples)
    
    print(f"Generated {len(dataset)} examples")
    print(f"Tool distribution:")
    from collections import Counter
    tool_counts = Counter(item["tool"] for item in dataset)
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count}")
    
    generator.save_dataset(dataset, args.output, format=args.format)
    print(f"Saved to: {args.output}")
    
    # Print sample
    print(f"\nSample tool call ({args.output_format} format):")
    print(f"  {dataset[0]['ground_truth'][:100]}...")


if __name__ == "__main__":
    main()
