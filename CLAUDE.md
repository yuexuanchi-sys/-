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
| `data_loader.py` | Math textbook data loading (.docx) |
| `enhanced_data_processor.py` | Text cleaning, BIOES NER data generation |
| `entity_extractor_kggen.py` | Entity extraction (rules + POS + BERT ensemble) |
| `relation_extractor_kggen.py` | Relation extraction (rules + patterns + LLM) |
| `knowledge_graph_builder_kggen.py` | KG builder with entity quality filtering |
| `kggen_client.py` | Cloud LLM API client (DeepSeek/KGGen) |
| `neo4j_manager.py` | Neo4j database CRUD operations (py2neo) |
| `main_kggen_pipeline.py` | Central pipeline orchestrator |
| `bert_trainer_v2.py` | BERT+BiLSTM+CRF trainer (BIOES labels, augmentation) |
| `evaluation.py` | Paper-grade evaluation (NER metrics, KG analysis) |
| `app.py` | Flask web app for graph visualization |
| `performance_optimizer.py` | Performance monitoring utilities |
| `run.py` | CLI entry point |
| `test_pipeline.py` | Comprehensive test suite (9 tests) |

## Data Flow

```
Raw Text (textbook .docx files in data/)
    |
    v
[entity_extractor_kggen.py]     -- Rules + POS + BERT extracts math entities
    |                                (concepts, formulas, theorems, etc.)
    v
[knowledge_graph_builder_kggen.py] -- Multi-layer entity quality filtering
    |                                  (stopwords, OCR noise, domain validation)
    v
[relation_extractor_kggen.py]    -- Rules + patterns + Cloud LLM for relations
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
- Run `pip install -r requirements.txt` for all dependencies
- Use `python run.py web` to start the Flask visualization server
- Use `python main_kggen_pipeline.py --mode hybrid` to run the full pipeline
- Use `python main_kggen_pipeline.py --mode train` to train BERT NER model
- Use `python main_kggen_pipeline.py --mode evaluate` to generate paper metrics
- Use `python test_pipeline.py` to run the test suite
