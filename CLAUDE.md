# 初中数学知识图谱系统 (Middle School Math Knowledge Graph)

## Architecture Overview

This project builds a **middle school math knowledge graph** using a **Hybrid Pipeline**:
- **Local BERT** for entity extraction (fast, offline)
- **Cloud LLM API** for relation extraction (high accuracy)
- **Neo4j** graph database for storage and querying

## Core Modules

| File | Role |
|------|------|
| `src/entity_recognizer_enhanced.py` | Local BERT-based entity recognition |
| `src/kggen_client.py` | Cloud LLM API client for knowledge graph generation |
| `src/relation_extractor_kggen.py` | LLM-powered relation extraction |
| `src/neo4j_manager.py` | Neo4j database CRUD operations |
| `src/main_unified_pipeline.py` | Central pipeline orchestrator |

## Data Flow

```
Raw Text (textbook content)
    |
    v
[entity_recognizer_enhanced.py]  -- Local BERT extracts math entities
    |                                (concepts, formulas, theorems, etc.)
    v
[relation_extractor_kggen.py]    -- Cloud LLM identifies relationships
    |  uses kggen_client.py          between extracted entities
    v
[neo4j_manager.py]               -- Stores (entity, relation, entity) triples
    |                                in Neo4j graph database
    v
Neo4j Knowledge Graph
```

## Key Design Decisions

1. **Hybrid approach**: BERT handles entity extraction locally (fast, no API cost); cloud LLM handles the harder task of relation extraction (higher accuracy).
2. **JSON as interchange format**: Entities are passed between modules as JSON objects.
3. **Pipeline orchestration**: `main_unified_pipeline.py` coordinates the entire flow.

## Development Notes

- Place test input files in `data/`
- Configuration (API keys, Neo4j credentials) goes in `config/`
- All core source code lives in `src/`

## Next Steps (Pending Implementation)

- [ ] Complete entity_recognizer_enhanced.py with BERT model loading
- [ ] Implement kggen_client.py API client
- [ ] Build relation_extractor_kggen.py
- [ ] Set up neo4j_manager.py with full CRUD + query support
- [ ] Wire everything together in main_unified_pipeline.py
- [ ] Add test script for end-to-end pipeline validation
- [ ] Add visualization (Pyvis/Dash) for graph display
