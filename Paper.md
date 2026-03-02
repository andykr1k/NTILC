# Flow of the paper
## Architecture
### Per paragraph we want 5-7 sentences
## Introduction - The Flow - 5-7 Paragraphs
### First Paragraph
#### "Something is hot" - background (overall temperature of the current environment)
### Second Paragraph
#### Emphasize previous method - refer to figure 1 (show why the method is bad - 2/3 points)
##### Side by side comparison of previous method and proposed method
#### Should be based on the results
### Third Paragraph
#### Explain why the previous methods is bad
##### Prompt based calling
##### No tool calling stage in between query to answer
##### Decoding step takes half inference time
##### Parsing failures
##### Context rot from large database of tools
### Fourth Paragraph
#### Talk about the proposed method
#### Must solve exactly what was mentioned in the limitation
#### Make this feel intuitive (people should believe in it without proof)
### Fifth Paragraph
#### Talk about the dataset - high level
#### Emphasize the metrics/experiments/results section - high level
#### Substantiate here....(sounds reasonable)
### Sixth Paragraph
#### Our contribution...
## Related Work
### Show methods that correlate to what was
## Method
### Qwen+Lora
### Tool Embeddings
### Argument Generation
### Orchestration
## Dataset
### Scraping
### NL to Command Generation
## Experiments
### Experiement Setting
#### Baseline/Metrics
### Results
#### Show results from different benchmarks
#### Its a big plus to interpret results
##### Plan Block, Dispatch Block and Response Block
#### Talk about the insights learned from this
#### Limitations -> Failure Case
##### New tool injection?
### Ablation Studies
## Conclusion - 1 paragraph
### Emphasize "i told you according to the previous data I gave you"
