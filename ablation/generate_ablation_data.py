"""
Generate test data for ablation studies: conversation context + tool calls.
"""

import json
import random
from typing import List, Dict
from faker import Faker
from pathlib import Path
from datetime import datetime, timedelta


class AblationDataGenerator:
    """
    Generate conversation queries that should trigger tool calls.
    """
    
    def __init__(self):
        self.faker = Faker()
        Faker.seed(42)
        random.seed(42)
    
    def generate_search_query(self) -> Dict[str, str]:
        """Generate a query that should trigger search tool."""
        # First, determine the parameters deterministically
        topics = [
            "machine learning",
            "quantum computing",
            "recent AI papers",
            "Python tutorials",
            "climate change",
            "renewable energy",
            "neural networks",
            "natural language processing"
        ]
        
        topic = random.choice(topics)
        topic_escaped = topic.replace("'", "\\'")
        max_results = random.randint(5, 50)
        
        # 30% chance of date filter
        has_date_filter = random.random() < 0.3
        if has_date_filter:
            days_ago = random.randint(1, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
        else:
            date_from = None
            date_to = None
        
        # Generate query templates that include the necessary information
        if has_date_filter:
            queries = [
                "Find me information about {topic} from {date_from} to {date_to}, return up to {max_results} results",
                "Search for {topic} between {date_from} and {date_to}, limit to {max_results} results",
                "I need to look up {topic} from {date_from} to {date_to}, show me {max_results} results",
                "Can you search for {topic} from {date_from} to {date_to}? Return {max_results} results",
            ]
            user_query = random.choice(queries).format(
                topic=topic, 
                date_from=date_from, 
                date_to=date_to,
                max_results=max_results
            )
        else:
            queries = [
                "Find me information about {topic}, return up to {max_results} results",
                "Search for {topic}, limit to {max_results} results",
                "I need to look up {topic}, show me {max_results} results",
                "Can you search for {topic}? Return {max_results} results",
                "What can you tell me about {topic}? Show {max_results} results",
                "Look up {topic} for me, return {max_results} results"
            ]
            user_query = random.choice(queries).format(topic=topic, max_results=max_results)
        
        # Generate tool call that matches the query
        if has_date_filter:
            tool_call = f"search(query='{topic_escaped}', max_results={max_results}, date_filter=DateRange(from_date='{date_from}', to_date='{date_to}'))"
        else:
            tool_call = f"search(query='{topic_escaped}', max_results={max_results})"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "search"
        }
    
    def generate_calculate_query(self) -> Dict[str, str]:
        """Generate a query that should trigger calculate tool."""
        queries = [
            "Calculate {expression}",
            "What is {expression}?",
            "Compute {expression}",
            "Evaluate {expression}",
            "Solve {expression}"
        ]
        
        expressions = [
            "2 + 2",
            "10 * 5",
            "sqrt(144)",
            "sin(pi/2)",
            "log(100)",
            "2^10",
            "(5 + 3) * 2"
        ]
        
        query_template = random.choice(queries)
        expression = random.choice(expressions)
        user_query = query_template.format(expression=expression)
        
        # Generate tool call that matches the expression
        expression_escaped = expression.replace("'", "\\'")
        tool_call = f"calculate(expression='{expression_escaped}')"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "calculate"
        }
    
    def generate_database_query(self) -> Dict[str, str]:
        """Generate a query that should trigger database_query tool."""
        # Map what_options to SQL queries - first determine parameters
        what_to_sql = {
            "all users": "SELECT * FROM users LIMIT {limit}",
            "recent orders": "SELECT * FROM orders WHERE created_at > '2024-01-01' LIMIT {limit}",
            "product information": "SELECT * FROM products LIMIT {limit}",
            "transaction logs": "SELECT * FROM transactions LIMIT {limit}",
            "user count": "SELECT COUNT(*) FROM users",
            "active records": "SELECT * FROM {table} WHERE status = 'active' LIMIT {limit}"
        }
        
        what_options = list(what_to_sql.keys())
        what = random.choice(what_options)
        limit = random.randint(10, 100)
        timeout = random.randint(10, 60)
        
        # Generate matching SQL query
        sql_template = what_to_sql[what]
        
        if "{table}" in sql_template:
            tables = ["users", "orders", "products", "transactions"]
            table = random.choice(tables)
            sql = sql_template.format(table=table, limit=limit)
        else:
            sql = sql_template.format(limit=limit)
        
        sql_escaped = sql.replace("'", "\\'")
        
        # Generate query that includes the necessary information
        queries = [
            "Query the database for {what} with a timeout of {timeout} seconds",
            "Get {what} from the database, timeout {timeout}s",
            "Run a SQL query to {what}, limit {timeout} seconds",
            "Fetch {what} from the database with {timeout}s timeout",
            "I need {what} from the database, timeout {timeout} seconds"
        ]
        
        user_query = random.choice(queries).format(what=what, timeout=timeout)
        
        tool_call = f"database_query(sql='{sql_escaped}', timeout={timeout})"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "database_query"
        }
    
    def generate_send_email_query(self) -> Dict[str, str]:
        """Generate a query that should trigger send_email tool."""
        # Use deterministic templates for body instead of random faker text
        body_templates = [
            "Please review the attached document and provide feedback by end of week.",
            "I wanted to follow up on our previous conversation about the project timeline.",
            "Thank you for your interest. I'll get back to you with more details soon.",
            "This is a reminder about the upcoming meeting scheduled for next Monday.",
            "I hope this email finds you well. I wanted to discuss the recent changes.",
            "Please find the requested information below. Let me know if you need anything else.",
            "I'm writing to confirm the details we discussed earlier today.",
            "Could you please provide an update on the status of this item?"
        ]
        
        recipient = self.faker.email()
        subject = self.faker.sentence(nb_words=random.randint(3, 6)).strip()
        body = random.choice(body_templates)
        
        # 30% chance of CC
        has_cc = random.random() < 0.3
        if has_cc:
            cc_emails = [self.faker.email() for _ in range(random.randint(1, 2))]
            cc_str = ", ".join(cc_emails)
        else:
            cc_emails = []
            cc_str = None
        
        # Generate query that includes all necessary information
        if has_cc:
            queries = [
                "Send an email to {recipient} with subject '{subject}' and body '{body}'. CC: {cc}",
                "Email {recipient} with subject '{subject}' saying '{body}'. Also CC {cc}",
                "Send a message to {recipient} regarding '{subject}' with body '{body}'. CC {cc}",
                "I need to email {recipient} about '{subject}'. Body: '{body}'. CC: {cc}"
            ]
            user_query = random.choice(queries).format(
                recipient=recipient, 
                subject=subject, 
                body=body,
                cc=cc_str
            )
        else:
            queries = [
                "Send an email to {recipient} with subject '{subject}' and body '{body}'",
                "Email {recipient} with subject '{subject}' saying '{body}'",
                "Send a message to {recipient} regarding '{subject}' with body '{body}'",
                "I need to email {recipient} about '{subject}'. Body: '{body}'"
            ]
            user_query = random.choice(queries).format(
                recipient=recipient, 
                subject=subject, 
                body=body
            )
        
        # Generate matching tool call
        recipient_escaped = recipient.replace("'", "\\'")
        subject_escaped = subject.replace("'", "\\'").replace("\n", "\\n")
        body_escaped = body.replace("'", "\\'").replace("\n", "\\n")
        
        if has_cc:
            cc_list_str = "[" + ", ".join([f"'{email}'" for email in cc_emails]) + "]"
            tool_call = f"send_email(to='{recipient_escaped}', subject='{subject_escaped}', body='{body_escaped}', cc={cc_list_str})"
        else:
            tool_call = f"send_email(to='{recipient_escaped}', subject='{subject_escaped}', body='{body_escaped}')"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "send_email"
        }
    
    def generate_web_fetch_query(self) -> Dict[str, str]:
        """Generate a query that should trigger web_fetch tool."""
        # Queries that imply GET (default)
        get_queries = [
            "Fetch data from {url}",
            "Get content from {url}",
            "Retrieve {url}",
            "Download {url}",
            "Access {url}"
        ]
        
        # Queries that explicitly mention POST
        post_queries = [
            "POST data to {url}",
            "Send data to {url}",
            "Submit to {url}"
        ]
        
        urls = [
            "https://api.github.com/users",
            "https://jsonplaceholder.typicode.com/posts",
            "https://api.example.com/data"
        ]
        
        # 80% chance of GET (default), 20% chance of POST
        if random.random() < 0.2:
            query_template = random.choice(post_queries)
            method = "POST"
        else:
            query_template = random.choice(get_queries)
            method = "GET"
        
        url = random.choice(urls)
        user_query = query_template.format(url=url)
        
        # Generate matching tool call
        url_escaped = url.replace("'", "\\'")
        tool_call = f"web_fetch(url='{url_escaped}', method='{method}')"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "web_fetch"
        }
    
    def generate_file_read_query(self) -> Dict[str, str]:
        """Generate a query that should trigger file_read tool."""
        paths = [
            "/home/user/log.txt",
            "./config.json",
            "/var/log/app.log",
            "data/input.csv"
        ]
        
        path = random.choice(paths)
        encodings = ["utf-8", "ascii", "latin-1", "utf-16"]
        encoding = random.choice(encodings)
        
        # Generate query that includes encoding information
        queries = [
            "Read the file at {path} using {encoding} encoding",
            "Open {path} with {encoding} encoding",
            "Get contents of {path}, read as {encoding}",
            "Load {path} with encoding {encoding}",
            "Read {path} for me using {encoding} encoding"
        ]
        
        user_query = random.choice(queries).format(path=path, encoding=encoding)
        
        # Generate matching tool call
        path_escaped = path.replace("'", "\\'")
        tool_call = f"file_read(path='{path_escaped}', encoding='{encoding}')"
        
        return {
            "query": user_query,
            "ground_truth": tool_call,
            "tool": "file_read"
        }
    
    def generate_dataset(self, num_samples: int) -> List[Dict[str, str]]:
        """
        Generate balanced dataset across all tools.
        
        Args:
            num_samples: Total number of samples
            
        Returns:
            List of examples with 'query', 'ground_truth', and 'tool' fields
        """
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
        """
        Save dataset to file.
        
        Args:
            dataset: List of examples
            filepath: Path to save
            format: "jsonl" or "json"
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "jsonl":
            with open(path, 'w') as f:
                for example in dataset:
                    f.write(json.dumps(example) + '\n')
        elif format == "json":
            with open(path, 'w') as f:
                json.dump(dataset, f, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ablation test data")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of test samples to generate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/ablation/test_data.jsonl",
        help="Output file path"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["jsonl", "json"],
        default="jsonl",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    print("Generating ablation test data...")
    generator = AblationDataGenerator()
    dataset = generator.generate_dataset(args.num_samples)
    
    print(f"Generated {len(dataset)} examples")
    print(f"Tool distribution:")
    from collections import Counter
    tool_counts = Counter(item["tool"] for item in dataset)
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count}")
    
    generator.save_dataset(dataset, args.output, format=args.format)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
