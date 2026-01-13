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
        # Use realistic topics instead of random faker sentences
        topics = [
            "machine learning",
            "quantum computing",
            "recent AI papers",
            "Python tutorials",
            "climate change",
            "renewable energy",
            "neural networks",
            "natural language processing",
            "deep learning",
            "computer vision",
            "data science",
            "web development",
            "cloud computing",
            "cybersecurity",
            "blockchain technology",
            "software engineering",
            "database design",
            "API development",
            "mobile app development",
            "DevOps practices",
            "containerization",
            "microservices architecture",
            "agile methodology",
            "test-driven development",
            "code review best practices",
            "version control systems",
            "CI/CD pipelines",
            "cloud infrastructure",
            "serverless computing",
            "edge computing",
            "IoT devices",
            "robotics",
            "augmented reality",
            "virtual reality",
            "game development",
            "UI/UX design",
            "frontend frameworks",
            "backend architecture",
            "distributed systems",
            "system design patterns"
        ]

        topic = random.choice(topics)
        topic_escaped = topic.replace("'", "\\'")
        max_results = random.randint(
            self.config.min_max_results, self.config.max_max_results)

        # Optional date filter (30% chance)
        if random.random() < 0.3:
            days_ago = random.randint(1, 365)
            date_from = (datetime.now() - timedelta(days=days_ago)
                         ).strftime("%Y-%m-%d")
            date_to = datetime.now().strftime("%Y-%m-%d")
            return f"search(query='{topic_escaped}', max_results={max_results}, date_filter=DateRange(from_date='{date_from}', to_date='{date_to}'))"
        else:
            return f"search(query='{topic_escaped}', max_results={max_results})"

    def generate_calculate(self) -> str:
        """Generate calculate(expression) invocation."""
        # Use more structured and realistic mathematical expressions
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
            "2 + 2",
            "10 * 5",
            "sqrt(144)",
            "sin(pi/2)",
            "cos(0)",
            "tan(pi/4)",
            "log(100)",
            "ln(e)",
            "2^10",
            "(5 + 3) * 2",
            "(10 - 4) / 2",
            "abs(-42)",
            "ceil(3.7)",
            "floor(3.7)",
            "round(3.14159)"
        ]
        expression = random.choice(expressions)
        expression_escaped = expression.replace("'", "\\'")
        return f"calculate(expression='{expression_escaped}')"

    def generate_database_query(self) -> str:
        """Generate database_query(sql, timeout) invocation."""
        # Use more structured SQL queries (matching ablation generator style)
        tables = ["users", "orders", "products", "transactions", "logs", "events",
                  "customers", "payments", "inventory", "sessions", "notifications",
                  "reviews", "categories", "tags", "comments", "posts"]
        columns = ["id", "name", "email", "created_at", "amount", "status",
                   "username", "password", "phone", "address", "city", "country",
                   "price", "quantity", "total", "discount", "tax", "shipping",
                   "title", "description", "content", "author", "published_at"]

        # Structured query types
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
            f"SELECT * FROM {random.choice(tables)} WHERE {random.choice(columns)} LIKE '%{random.choice(['test', 'demo', 'sample'])}%' LIMIT {random.randint(10, 100)}",
            f"SELECT * FROM {random.choice(tables)} WHERE {random.choice(columns)} IN ({', '.join([str(random.randint(1, 100)) for _ in range(random.randint(2, 5))])}) LIMIT {random.randint(10, 100)}",
            f"SELECT * FROM {random.choice(tables)} WHERE {random.choice(columns)} BETWEEN {random.randint(1, 50)} AND {random.randint(51, 100)} LIMIT {random.randint(10, 100)}"
        ]

        sql = random.choice(query_types)
        sql = sql.replace("'", "\\'")  # Escape quotes
        timeout = random.randint(10, 60)

        return f"database_query(sql='{sql}', timeout={timeout})"

    def generate_send_email(self) -> str:
        """Generate send_email(to, subject, body, cc) invocation."""
        to_email = self.faker.email()

        # Use more realistic subject templates
        subject_templates = [
            "Meeting reminder for next week",
            "Project update and status report",
            "Follow-up on our conversation",
            "Request for information",
            "Thank you for your inquiry",
            "Action required: Review needed",
            "Weekly status update",
            "Important announcement",
            "Quarterly review meeting",
            "Budget approval request",
            "Team collaboration invitation",
            "Deadline extension notification",
            "New feature release announcement",
            "Security update required",
            "Performance metrics summary",
            "Client feedback and response",
            "Training session invitation",
            "System maintenance schedule",
            "Code review request",
            "Documentation update notice",
            "Bug fix deployment",
            "API endpoint changes",
            "Database migration plan",
            "Infrastructure upgrade proposal"
        ]
        subject = random.choice(subject_templates)
        subject_escaped = subject.replace("'", "\\'")

        # Use template-based body text instead of random faker text
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
            "Please let me know if you have any questions or concerns about this matter.",
            "I'm reaching out to schedule a follow-up meeting to discuss the next steps.",
            "Attached please find the quarterly report for your review and approval.",
            "We need to address the issues raised in the last team meeting.",
            "I would appreciate your input on the proposed changes to the workflow.",
            "This email serves as confirmation of our agreement on the project scope.",
            "Please review the attached specifications and let me know your thoughts.",
            "I wanted to inform you about the recent updates to our system architecture.",
            "We are planning a team building event next month and would love your participation.",
            "I need your assistance with reviewing the code changes before deployment.",
            "Thank you for your continued support and collaboration on this project.",
            "I'm writing to inform you about the upcoming deadline for the submission.",
            "Please find below a summary of the key points from our discussion.",
            "I wanted to check in and see if you have any questions about the implementation.",
            "We have identified some areas for improvement and would value your feedback.",
            "This is to notify you about the scheduled maintenance window next weekend."
        ]
        body = random.choice(body_templates)
        body_escaped = body.replace("'", "\\'").replace("\n", "\\n")

        # Optional CC (30% chance, matching ablation generator)
        if random.random() < 0.3:
            cc_emails = [self.faker.email()
                         for _ in range(random.randint(1, 2))]
            cc_str = "[" + \
                ", ".join([f"'{email}'" for email in cc_emails]) + "]"
            return f"send_email(to='{to_email}', subject='{subject_escaped}', body='{body_escaped}', cc={cc_str})"
        else:
            return f"send_email(to='{to_email}', subject='{subject_escaped}', body='{body_escaped}')"

    def generate_web_fetch(self) -> str:
        """Generate web_fetch(url, method) invocation."""
        # Use more realistic URL patterns
        urls = [
            "https://api.github.com/users",
            "https://jsonplaceholder.typicode.com/posts",
            "https://api.example.com/data",
            "https://api.github.com/repos",
            "https://jsonplaceholder.typicode.com/users",
            "https://api.example.com/endpoint",
            "https://api.github.com/orgs",
            "https://api.github.com/issues",
            "https://jsonplaceholder.typicode.com/comments",
            "https://jsonplaceholder.typicode.com/albums",
            "https://api.github.com/pulls",
            "https://api.github.com/commits",
            "https://api.example.com/v1/users",
            "https://api.example.com/v1/products",
            "https://api.example.com/v1/orders",
            "https://httpbin.org/get",
            "https://httpbin.org/post",
            "https://httpbin.org/json",
            "https://api.stripe.com/v1/customers",
            "https://api.stripe.com/v1/charges",
            "https://api.openweathermap.org/data/2.5/weather",
            "https://api.spotify.com/v1/albums",
            "https://api.spotify.com/v1/tracks",
            "https://api.twitter.com/1.1/statuses",
            "https://api.linkedin.com/v2/people",
            "https://graph.facebook.com/v18.0/me",
            "https://api.slack.com/api/users.list",
            "https://api.slack.com/api/channels.list"
        ]

        # 80% chance of GET (default), 20% chance of POST (matching ablation generator)
        if random.random() < 0.2:
            method = "POST"
        else:
            method = "GET"

        url = random.choice(urls)
        url_escaped = url.replace("'", "\\'")

        return f"web_fetch(url='{url_escaped}', method='{method}')"

    def generate_file_read(self) -> str:
        """Generate file_read(path, encoding) invocation."""
        # Use more structured file paths (matching ablation generator)
        paths = [
            "/home/user/log.txt",
            "./config.json",
            "/var/log/app.log",
            "data/input.csv",
            "/tmp/output.log",
            "./data/readme.md",
            "../config/settings.ini",
            "/var/log/nginx/access.log",
            "/var/log/nginx/error.log",
            "/etc/nginx/nginx.conf",
            "/home/user/documents/report.pdf",
            "./src/main.py",
            "./tests/test_suite.py",
            "data/dataset.json",
            "data/training_data.csv",
            "data/validation_data.csv",
            "logs/application.log",
            "logs/error.log",
            "logs/debug.log",
            "/tmp/temp_file.txt",
            "/tmp/cache/data.bin",
            "./output/results.json",
            "./output/analysis.csv",
            "../shared/config.yaml",
            "../shared/secrets.env",
            "/opt/app/config/config.ini",
            "/opt/app/data/database.db",
            "~/documents/notes.txt",
            "~/projects/src/index.js",
            "~/projects/tests/unit.test.js",
            "/usr/local/bin/script.sh",
            "/usr/share/data/sample.json",
            "config/database.yml",
            "config/application.yml",
            "migrations/001_initial.sql",
            "migrations/002_add_users.sql"
        ]

        path = random.choice(paths)
        encodings = ["utf-8", "ascii", "latin-1", "utf-16"]
        encoding = random.choice(encodings)
        path_escaped = path.replace("'", "\\'")

        return f"file_read(path='{path_escaped}', encoding='{encoding}')"

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
