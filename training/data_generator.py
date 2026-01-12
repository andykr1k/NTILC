"""
Synthetic data generator for tool invocation training data.
"""

import random
import json
from typing import List, Dict, Any
from faker import Faker
from datetime import datetime, timedelta

from .config import DataGeneratorConfig


class ToolInvocationGenerator:
    """
    Generates synthetic tool invocation strings for training.
    
    Supports multiple tool types with realistic parameter distributions.
    """
    
    def __init__(self, config: DataGeneratorConfig = None):
        """
        Args:
            config: Configuration for data generation
        """
        self.config = config or DataGeneratorConfig()
        self.faker = Faker()
        Faker.seed(42)  # For reproducibility
        random.seed(42)
    
    def generate_search(self) -> str:
        """Generate search(query, max_results, date_filter) invocation."""
        query = self.faker.sentence(nb_words=random.randint(3, 10)).strip()
        query = query.replace("'", "\\'")  # Escape quotes
        max_results = random.randint(self.config.min_max_results, self.config.max_max_results)
        
        # Optional date filter (30% chance)
        if random.random() < 0.3:
            days_ago = random.randint(1, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            return f"search(query='{query}', max_results={max_results}, date_filter=DateRange(from_date='{date_from}', to_date='{date_to}'))"
        else:
            return f"search(query='{query}', max_results={max_results})"
    
    def generate_calculate(self) -> str:
        """Generate calculate(expression) invocation."""
        # Generate various mathematical expressions
        expressions = [
            f"{random.randint(1, 1000)} + {random.randint(1, 1000)}",
            f"{random.randint(1, 100)} * {random.randint(1, 100)}",
            f"{random.randint(1, 100)} / {random.randint(1, 100)}",
            f"sqrt({random.randint(1, 10000)})",
            f"pow({random.randint(1, 10)}, {random.randint(1, 5)})",
            f"sin({random.uniform(0, 6.28)})",
            f"cos({random.uniform(0, 6.28)})",
            f"log({random.randint(1, 1000)})",
            f"({random.randint(1, 100)} + {random.randint(1, 100)}) * {random.randint(1, 10)}",
            f"{random.randint(1, 100)} ** {random.randint(1, 5)}"
        ]
        expression = random.choice(expressions)
        return f"calculate(expression='{expression}')"
    
    def generate_database_query(self) -> str:
        """Generate database_query(sql, timeout) invocation."""
        # Generate various SQL queries
        tables = ["users", "orders", "products", "transactions", "logs", "events"]
        columns = ["id", "name", "email", "created_at", "amount", "status"]
        
        query_types = [
            f"SELECT * FROM {random.choice(tables)} LIMIT {random.randint(1, 100)}",
            f"SELECT {', '.join(random.sample(columns, random.randint(1, 3)))} FROM {random.choice(tables)}",
            f"SELECT COUNT(*) FROM {random.choice(tables)} WHERE status = 'active'",
            f"SELECT * FROM {random.choice(tables)} WHERE created_at > '2024-01-01'",
            f"INSERT INTO {random.choice(tables)} (name, email) VALUES ('{self.faker.name()}', '{self.faker.email()}')",
            f"UPDATE {random.choice(tables)} SET status = 'completed' WHERE id = {random.randint(1, 1000)}"
        ]
        
        sql = random.choice(query_types)
        sql = sql.replace("'", "\\'")  # Escape quotes
        timeout = random.randint(1, 60)
        
        return f"database_query(sql='{sql}', timeout={timeout})"
    
    def generate_send_email(self) -> str:
        """Generate send_email(to, subject, body, cc) invocation."""
        to_email = self.faker.email()
        subject = self.faker.sentence(nb_words=random.randint(3, 8)).strip()
        subject = subject.replace("'", "\\'")
        body = self.faker.text(max_nb_chars=random.randint(50, 500)).strip()
        body = body.replace("'", "\\'").replace("\n", "\\n")
        
        # Optional CC (40% chance)
        if random.random() < 0.4:
            cc_emails = [self.faker.email() for _ in range(random.randint(1, 3))]
            cc_str = "[" + ", ".join([f"'{email}'" for email in cc_emails]) + "]"
            return f"send_email(to='{to_email}', subject='{subject}', body='{body}', cc={cc_str})"
        else:
            return f"send_email(to='{to_email}', subject='{subject}', body='{body}')"
    
    def generate_web_fetch(self) -> str:
        """Generate web_fetch(url, method) invocation."""
        methods = ["GET", "POST"]
        method = random.choice(methods)
        
        # Generate various URL patterns
        domains = ["example.com", "api.github.com", "jsonplaceholder.typicode.com", "httpbin.org"]
        paths = ["/users", "/posts", "/data", "/api/v1/endpoint", "/search", "/items"]
        
        url = f"https://{random.choice(domains)}{random.choice(paths)}"
        
        return f"web_fetch(url='{url}', method='{method}')"
    
    def generate_file_read(self) -> str:
        """Generate file_read(path, encoding) invocation."""
        # Generate various file paths
        directories = ["/home/user", "/var/log", "/data", "/tmp", "./data", "../config"]
        filenames = ["log.txt", "config.json", "data.csv", "output.log", "readme.md", "settings.ini"]
        
        path = f"{random.choice(directories)}/{random.choice(filenames)}"
        encodings = ["utf-8", "ascii", "latin-1", "utf-16"]
        encoding = random.choice(encodings)
        
        return f"file_read(path='{path}', encoding='{encoding}')"
    
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
