"""
Baseline 1: Naive Prompting
Simply provide tools in the prompt and ask LLM to generate tool calls.
"""

import re
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from multiprocessing import Process, Queue
import multiprocessing

from .tool_schemas import TOOL_SCHEMAS, format_tools_for_prompt, get_tool_examples


def _extract_tool_call(text: str, prompt: str = "") -> str:
    """
    Extract tool call from generated text by properly parsing function call syntax.
    
    Args:
        text: Generated text that may contain tool call and explanatory text
        prompt: Original prompt (for fallback extraction)
        
    Returns:
        Extracted tool call string
    """
    # First, try to find text after "Tool call:"
    if "Tool call:" in text:
        candidate = text.split("Tool call:")[-1].strip()
    else:
        # Fallback: try to extract from end of prompt
        if prompt and prompt in text:
            candidate = text[len(prompt):].strip()
        else:
            candidate = text.strip()
    
    # Remove leading/trailing whitespace and take first line
    candidate = candidate.split("\n")[0].strip()
    
    # Find the start of the function call (tool_name followed by opening paren)
    # Pattern: word( where word is the tool name
    paren_start = -1
    tool_start = -1
    for i in range(len(candidate)):
        if candidate[i] == '(':
            # Check if there's a valid tool name before it
            j = i - 1
            # Skip whitespace between tool name and paren
            while j >= 0 and candidate[j].isspace():
                j -= 1
            # Find the end of the tool name
            tool_end = j + 1
            # Find the start of the tool name
            while j >= 0 and (candidate[j].isalnum() or candidate[j] == '_'):
                j -= 1
            tool_start = j + 1
            if tool_start < tool_end:  # Found a valid tool name
                paren_start = i
                break
    
    if paren_start > 0 and tool_start >= 0:
        # Found opening paren, try to find matching closing paren
        paren_count = 0
        in_string = False
        string_char = None
        escape_next = False
        
        for i in range(paren_start, len(candidate)):
            char = candidate[i]
            
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
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                    if paren_count == 0:
                        # Found complete function call
                        return candidate[tool_start:i+1].strip()
        
        # If we didn't find a closing paren, try to find where explanatory text starts
        # Look for common phrases that indicate the end of the tool call
        explanatory_phrases = [
            " To fulfill",
            " To complete",
            " To meet",
            " Here's",
            " Here is",
            " You can",
            " def ",
            "Human:",
            " To respond",
            " To generate",
        ]
        for phrase in explanatory_phrases:
            idx = candidate.find(phrase, paren_start)
            if idx > paren_start:
                # Try to find closing paren before this phrase
                # If not found, return up to this point (might be incomplete)
                return candidate[tool_start:idx].strip()
    
    # Last resort: return first line, but try to clean it up
    # Remove common explanatory phrases that might be on the same line
    cleaned = candidate
    explanatory_phrases = [
        " To fulfill your request",
        " To complete this task",
        " To meet your requirement",
        " Here's how",
        " Here is",
        " You can use",
        " def ",
        "Human:",
        " To respond",
    ]
    for phrase in explanatory_phrases:
        idx = cleaned.find(phrase)
        if idx > 0:
            cleaned = cleaned[:idx].strip()
            break
    
    return cleaned.strip()


def _naive_worker_process(gpu_id, queries, prompts_list, result_queue, progress_queue, model_name, max_length, temperature, batch_size=8):
    """Worker process that runs on a specific GPU (module-level for pickling)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    # Set CUDA device for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # Load model and tokenizer in this process
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set left padding for decoder-only models
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process queries in batches for better GPU utilization
    predictions = []
    num_batches = (len(prompts_list) + batch_size - 1) // batch_size
    
    for i in range(0, len(prompts_list), batch_size):
        batch_prompts = prompts_list[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Generate for batch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract for each in batch
        batch_predictions = []
        for prompt, gen_ids in zip(batch_prompts, outputs):
            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            # Use improved extraction function
            tool_call = _extract_tool_call(generated_text, prompt)
            batch_predictions.append((generated_text, tool_call))
        
        predictions.extend(batch_predictions)
        
        # Send progress update after each batch
        if progress_queue is not None:
            progress_queue.put(len(batch_predictions))
    
    # Send final results back
    result_queue.put((gpu_id, predictions))


class NaivePromptingBaseline:
    """
    Baseline that uses naive prompting: just list tools in prompt.
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 256,
        temperature: float = 0.7,
        include_examples: bool = True,
        num_gpus: int = 1
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to run on (base device, will use multiple if num_gpus > 1)
            max_length: Maximum generation length
            temperature: Sampling temperature
            include_examples: Whether to include example tool calls in prompt
            num_gpus: Number of GPUs to use (creates separate model instances)
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        self.include_examples = include_examples
        self.num_gpus = num_gpus
        
        # Determine actual number of GPUs to use
        if num_gpus > 1 and torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            self.num_gpus = min(num_gpus, available_gpus)
            if self.num_gpus < num_gpus:
                print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. Using {self.num_gpus} GPUs.")
        else:
            self.num_gpus = 1
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set left padding for decoder-only models
        self.tokenizer.padding_side = 'left'
        
        # Create model instances for each GPU
        self.models = []
        self.devices = []
        
        if self.num_gpus > 1:
            print(f"Creating {self.num_gpus} model instances for multi-GPU inference")
            for gpu_id in range(self.num_gpus):
                device = f"cuda:{gpu_id}"
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model.to(device)
                model.eval()
                self.models.append(model)
                self.devices.append(device)
        else:
            # Single GPU or CPU
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(device)
            model.eval()
            self.models = [model]
            self.devices = [device]
        
        # For backward compatibility
        self.model = self.models[0]
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _build_prompt(self, user_query: str) -> str:
        """
        Build prompt with tool descriptions and user query.
        
        Args:
            user_query: User's request/query
            
        Returns:
            Complete prompt string
        """
        prompt_parts = []
        
        # Add tool descriptions
        prompt_parts.append(format_tools_for_prompt())
        prompt_parts.append("")
        
        # Add examples if requested
        if self.include_examples:
            prompt_parts.append("Examples:")
            examples = get_tool_examples()
            for tool_name, tool_examples in examples.items():
                prompt_parts.append(f"  {tool_examples[0]}")
            prompt_parts.append("")
        
        # Add instruction
        prompt_parts.append("Given the user's request below, generate the appropriate tool call. Output the following format only.")
        prompt_parts.append("Format: tool_name(param1='value1', param2=value2)")
        prompt_parts.append("")
        prompt_parts.append(f"User request: {user_query}")
        prompt_parts.append("Tool call:")
        
        return "\n".join(prompt_parts)
    
    def predict(self, user_query: str) -> str:
        """
        Predict tool call for user query.
        
        Args:
            user_query: User's request/query
            
        Returns:
            Predicted tool call string
        """
        prompt = self._build_prompt(user_query)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract tool call using improved parsing
        tool_call = _extract_tool_call(generated_text, prompt)
        
        return tool_call
    
    
    def predict_batch(self, user_queries: List[str], batch_size: int = None) -> List[tuple]:
        """
        Predict tool calls for multiple queries using multi-GPU if available.
        
        Args:
            user_queries: List of user requests
            batch_size: Batch size per GPU. If None, uses 8 as default
            
        Returns:
            List of tuples (generated_text, tool_call)
        """
        if batch_size is None:
            batch_size = 8
        
        # If single GPU, use simple batching
        if self.num_gpus == 1:
            prompts = [self._build_prompt(query) for query in user_queries]
            predictions = []
            num_batches = (len(user_queries) + batch_size - 1) // batch_size
            
            for i in tqdm(range(0, len(user_queries), batch_size), desc="Predicting tool calls", total=num_batches):
                batch_prompts = prompts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                for prompt, gen_ids in zip(batch_prompts, outputs):
                    generated_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                    # Use improved extraction function
                    tool_call = _extract_tool_call(generated_text, prompt)
                    predictions.append((generated_text, tool_call))
            
            return predictions
        
        # Multi-GPU: Process queries in parallel using multiprocessing
        # Build prompts first (before forking processes)
        prompts = [self._build_prompt(query) for query in user_queries]
        
        # Split queries and prompts into chunks for each GPU
        chunk_size = len(user_queries) // self.num_gpus
        query_chunks = []
        prompt_chunks = []
        for i in range(self.num_gpus):
            start_idx = i * chunk_size
            if i == self.num_gpus - 1:
                end_idx = len(user_queries)
            else:
                end_idx = (i + 1) * chunk_size
            query_chunks.append(user_queries[start_idx:end_idx])
            prompt_chunks.append(prompts[start_idx:end_idx])
        
        # Set multiprocessing start method (required for CUDA)
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            multiprocessing.set_start_method('spawn', force=True)
        
        # Start processes for each GPU
        result_queue = Queue()
        progress_queue = Queue()  # Separate queue for progress updates
        processes = []
        
        for gpu_id, (queries, prompts_list) in enumerate(zip(query_chunks, prompt_chunks)):
            if not queries:
                continue
            p = Process(
                target=_naive_worker_process,
                args=(
                    gpu_id,
                    queries,
                    prompts_list,
                    result_queue,
                    progress_queue,  # Pass progress queue
                    self.model_name,
                    self.max_length,
                    self.temperature,
                    batch_size  # Pass batch_size to worker
                )
            )
            p.start()
            processes.append(p)
        
        # Collect results with real-time progress updates
        all_predictions = [None] * len(user_queries)
        completed = 0
        
        with tqdm(total=len(user_queries), desc="Predicting tool calls") as pbar:
            # Update progress bar as batches complete
            while completed < len(user_queries):
                try:
                    # Check for progress updates (non-blocking)
                    while not progress_queue.empty():
                        batch_count = progress_queue.get_nowait()
                        completed += batch_count
                        pbar.update(batch_count)
                    
                    # Check for final results (non-blocking)
                    if not result_queue.empty():
                        gpu_id, chunk_predictions = result_queue.get_nowait()
                        chunk_start = gpu_id * chunk_size
                        all_predictions[chunk_start:chunk_start + len(chunk_predictions)] = chunk_predictions
                    
                    # Small sleep to avoid busy waiting
                    import time
                    time.sleep(0.1)
                except:
                    # If queue is empty, wait a bit
                    import time
                    time.sleep(0.1)
            
            # Collect any remaining results
            for _ in range(len(processes)):
                if not result_queue.empty():
                    gpu_id, chunk_predictions = result_queue.get_nowait()
                    chunk_start = gpu_id * chunk_size
                    if all_predictions[chunk_start] is None:  # Only if not already set
                        all_predictions[chunk_start:chunk_start + len(chunk_predictions)] = chunk_predictions
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        return all_predictions
    
    def evaluate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Evaluate on test data.
        
        Args:
            test_data: List of dicts with 'query' and 'ground_truth' keys
            
        Returns:
            Dictionary of metrics (includes 'predictions' key for reuse)
        """
        from evaluation.metrics import exact_match_accuracy, tool_accuracy, parameter_accuracy
        
        queries = [item["query"] for item in test_data]
        ground_truth = [item["ground_truth"] for item in test_data]
        
        predictions_with_original = self.predict_batch(queries)
        # Extract just the tool calls for metrics
        predictions = [tool_call for _, tool_call in predictions_with_original]
        original_predictions = [orig for orig, _ in predictions_with_original]
        
        metrics = {
            "exact_match_accuracy": exact_match_accuracy(ground_truth, predictions),
            "tool_accuracy": tool_accuracy(ground_truth, predictions),
            "predictions": predictions,  # Store extracted tool calls for reuse
            "original_predictions": original_predictions  # Store original generated text
        }
        
        param_metrics = parameter_accuracy(ground_truth, predictions)
        metrics.update(param_metrics)
        
        return metrics
