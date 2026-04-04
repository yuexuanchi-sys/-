# 初中数学知识图谱系统 (Middle School Math Knowledge Graph)

## Architecture Overview

This project builds a **middle school math knowledge graph** using a **Hybrid Pipeline**:
- **Local BERT** (BERT+BiLSTM+CRF) for entity extraction (fast, offline)
- **Cloud LLM API** (DeepSeek via KGGen) for relation extraction (high accuracy)
- **Neo4j** graph database for storage and querying
- **Flask** web app for interactive visualization

## Core Modules

| File | Role |
|------|------|
| `config.py` | Central configuration (env vars via `.env`) |
| `entity_recognizer_enhanced.py` | BERT+CRF char-level entity recognition |
| `entity_extractor_kggen.py` | KGGen-enhanced entity extraction (ensemble) |
| `kggen_client.py` | Cloud LLM API client (DeepSeek/KGGen) |
| `relation_extractor_kggen.py` | LLM-powered relation extraction |
| `neo4j_manager.py` | Neo4j database CRUD operations (py2neo) |
| `main_kggen_pipeline.py` | Central pipeline orchestrator |
| `knowledge_graph_builder_kggen.py` | Knowledge graph builder |
| `bert_trainer_v2.py` | BERT+BiLSTM+CRF trainer (BIOES labels, augmentation) |
| `data_loader.py` | Math textbook data loading |
| `enhanced_data_processor.py` | Text cleaning, segmentation, NER data prep |
| `app.py` | Flask web app for graph visualization |
| `grade7_builder.py` | Grade 7 specific knowledge graph builder |

## Data Flow

```
Raw Text (textbook .docx files in data/)
    |
    v
[entity_recognizer_enhanced.py]  -- Local BERT+CRF extracts math entities
    |                                (concepts, formulas, theorems, etc.)
    v
[relation_extractor_kggen.py]    -- Cloud LLM identifies relationships
    |  uses kggen_client.py          between extracted entities
    v
[neo4j_manager.py]               -- Stores (entity, relation, entity) triples
    |                                in Neo4j graph database
    v
[app.py]                         -- Flask web UI for visualization
```

## Configuration

All secrets and paths are read from environment variables. Copy `.env.example` to `.env` and fill in your values:
- `KGGEN_API_KEY` - API key for DeepSeek/KGGen
- `NEO4J_PASSWORD` - Neo4j database password
- `BERT_MODEL_PATH` - Path to BERT model (default: bert-base-chinese)
- `DATA_DIR` - Path to textbook data directory

## Development Notes

- Place textbook data files in `data/`
- All core source code lives in the project root
- `src/` contains compatibility re-exports for the original module layout
- Run `pip install -r requirements.txt` for all dependencies
- Use `python run.py web` to start the Flask visualization server
- Use `python main_kggen_pipeline.py --mode hybrid` to run the full pipeline
