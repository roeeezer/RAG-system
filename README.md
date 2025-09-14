# Project Overview

This project implements a modular system for processing and retrieving information from kolzchut data.

## Setup Requirements

### Data Files
- The `kolzchut` folder must contain the extracted data from the zip file
- The `untracked` folder must contain a `gemini_api_keys.txt` file with at least one Gemini API key (you can create one for free with your google account at https://aistudio.google.com/app/apikey)

## Component Architecture

### Data Preprocessing
- `pre_process_data_interface.py`: Handles all preprocessing operations on the input data

### Index Optimization
- Located in the `IndexOptimizer` folder
- Contains optimizer interface and implementations
- Responsible for optimizing the indexing process

### Data Indexing
- `index_data_interface.py`: Manages data indexing operations

### LLM Answer Retrieval
Located in the `LlmAnswerRetriever` folder:
- Contains LLM Answer Retriever interface and implementations
- Handles final answer retrieval
- Includes Gemini class for managing requests to Gemini LLM

### Synonym Expansion
Located in the `SynonymExpanders` folder:
- Contains `SynonymExpander` class
- Utilized by `SynonymEnrichmentOptimizer` for query expansion