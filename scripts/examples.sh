# Input Directory
INPUT_DIR="/scratch4/home/akrik/NTILC/data/ToolVerifier"

# Create Tool Schema (If Needed)
python create_tool_schema.py --tools-path=${INPUT_DIR}/tools.json --output-path=${INPUT_DIR}/tool_schema.json

# Create Tool Embedding Dataset (If Needed)
python create_dataset.py --tools-path=${INPUT_DIR}/tool_schema.json --output-path=${INPUT_DIR}/tool_embedding_dataset.jsonl --summary-path=${INPUT_DIR}/tool_embedding_dataset_summary.jsonl

# Train Tool Embedding Model
python train_embedding_space.py --dataset-path=${INPUT_DIR}/tool_embedding_dataset.jsonl --output-path=${INPUT_DIR}/output