# Input Directory
DATASET="Vigil"
TOOLS_CSV_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tools.csv"
TOOLS_JSON_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tools.json"
TOOL_EMBEDDING_DATASET_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset.jsonl"
TOOL_EMBEDDING_DATASET_SUMMARY_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/tool_embedding_dataset_summary.jsonl"
BENCHMARK_PATH="/scratch4/home/akrik/NTILC/data/${DATASET}/benchmark.json"
OUTPUT_DIR="/scratch4/home/akrik/NTILC/data/${DATASET}/output"

# Create Tool Schema (If Needed)
# python utils/create_tool_schema.py --tools-path=${TOOLS_CSV_PATH} --output-path=${TOOLS_JSON_PATH}

# Create Tool Embedding Dataset (If Needed)
python utils/create_dataset.py --tools-path=${TOOLS_JSON_PATH} --output-path=${TOOL_EMBEDDING_DATASET_PATH} --summary-path=${TOOL_EMBEDDING_DATASET_SUMMARY_PATH}

# Train Tool Embedding Model
# python training/train_embedding_space.py --dataset-path=${TOOL_EMBEDDING_DATASET_PATH} --output-path=${OUTPUT_DIR}

# Evaluate ToolCall15 Agent Pipeline
# python evals/ToolCall15/runEval.py --checkpoint-path="/scratch4/home/akrik/NTILC/data/ToolCall15/output/best.pt" --benchmark-path="/scratch4/home/akrik/NTILC/data/ToolCall15/benchmark.json" --tools-path="/scratch4/home/akrik/NTILC/data/ToolCall15/tools.json" --output-path="/scratch4/home/akrik/NTILC/data/ToolCall15/output/eval/eval_summary.json"
