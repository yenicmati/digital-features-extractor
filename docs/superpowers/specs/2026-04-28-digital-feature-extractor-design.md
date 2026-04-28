# Digital Feature Extractor — Design Spec
**Date**: 2026-04-28  
**Status**: Draft — awaiting user approval

---

## 1. Objectif

Construire un outil CLI Python qui prend en input une codebase (URL GitHub ou chemin local), analyse son code, construit un knowledge graph, et en extrait les **Digital Features** au sens ADEO — des capacités user-visible qui apportent de la valeur, distinctes des fonctions techniques pures.

L'output est triple : un fichier JSON machine-readable, un rapport HTML lisible par les PMs, et un knowledge graph interactif.

Le mapping vers les Business Features et Business Use Cases est hors scope du MVP (prévu v2).

---

## 2. Définition de référence : Digital Feature

Conformément au document `Architecture - Business & Digital feature definition.md` :

> Une **Digital Feature** est une capacité ou fonctionnalité user-visible d'un Digital Product, qui apporte de la valeur à l'utilisateur final, peut être développée et déployée indépendamment, et constitue un élément du changelog.

Une Digital Feature **n'est pas** une fonction technique interne. Elle correspond à un besoin utilisateur, pas à une ligne de code.

---

## 3. Attributs extraits par feature

| Attribut | Type | Description |
|---|---|---|
| `id` | string | Identifiant unique généré (slug) |
| `name` | string | Nom officiel, compréhensible par les PMs |
| `description` | string | But de la fonctionnalité et valeur apportée |
| `status` | enum | `Live`, `To Be Developed`, `Deprecated`, `To Review` |
| `parent_product` | string | Inféré depuis le nom du repo / package root |
| `entry_points` | list[string] | Endpoints HTTP, event handlers, fonctions publiques sources |
| `business_capability_hint` | string | Suggestion LLM de la Business Capability parente (non validée) |
| `confidence_score` | float | Score 0-1 : qualité de l'inférence LLM |

---

## 4. Architecture

### Vue d'ensemble

```
Input (GitHub URL ou chemin local)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  INGESTION                                      │
│  - Clone repo (git clone --depth=1) ou scan     │
│  - Filtre : src/ uniquement, exclure tests/     │
│    node_modules/, vendor/, .git/                │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  GRAPHIFY CORE (OSS — safishamsi/graphify)      │
│  Pass 1 : tree-sitter AST (Java/TS/Python)      │
│           → classes, fonctions, endpoints, deps │
│  Pass 2 : LLM semantic extraction               │
│           → relations sémantiques entre noeuds  │
│  Output  : NetworkX graph                       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  DIGITAL FEATURE EXTRACTOR (couche custom)      │
│  - Leiden clustering sur le NetworkX graph      │
│  - Pour chaque cluster → prompt LLM structuré   │
│    avec la définition Digital Feature ADEO      │
│  - Validation Pydantic des attributs            │
│  - Déduplication et scoring de confiance        │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  OUTPUT GENERATOR                               │
│  - features.json (machine-readable)             │
│  - report.html  (lisible PMs / architectes)     │
│  - graph.html   (visualisation interactive)     │
└─────────────────────────────────────────────────┘
```

### Composants

#### 4.1 Ingestion

- `GithubIngester` : `git clone --depth=1 <url>` dans un répertoire temporaire, avec support des repos privés via token d'environnement `GITHUB_TOKEN`.
- `LocalIngester` : scan récursif du chemin fourni.
- Les deux implémentent l'interface `Ingester` (méthode `get_file_paths() -> list[Path]`).
- Filtres configurables via `.featureextractor.yml` à la racine du projet analysé (inclure/exclure des globs).

#### 4.2 Graphify Core (OSS)

- Intégré comme dépendance Python (`pip install graphify`).
- On utilise son API programmatique, pas son CLI.
- On surcharge uniquement le prompt LLM de la Pass 2 pour orienter l'extraction sémantique vers les Digital Features.
- Output : objet `networkx.Graph` avec attributs par nœud (type, fichier, ligne, relations).

#### 4.3 Digital Feature Extractor

- **Clustering** : Leiden algorithm (via `leidenalg` + `igraph`) sur le NetworkX graph → communautés de nœuds cohérents.
- **Prompt LLM** : pour chaque cluster, on soumet au LLM :
  - La liste des nœuds du cluster (fichiers, fonctions, endpoints)
  - La définition ADEO de Digital Feature
  - Un few-shot exemple (2 exemples de features bien formées)
  - La demande de remplir le JSON schema `DigitalFeature`
- **Modèles supportés** : OpenAI (GPT-4o, GPT-4-turbo), Anthropic (Claude claude-sonnet-4-5), tout modèle compatible OpenAI API.
- **Cache** : résultats LLM mis en cache par hash SHA256 du cluster → rejouer l'analyse sans coût supplémentaire.
- **Confidence scoring** : score basé sur (1) taille du cluster, (2) présence de docstrings/commentaires, (3) présence d'endpoints HTTP explicites.

#### 4.4 Output Generator

- `JsonExporter` : dump Pydantic → `features.json`
- `HtmlReporter` : template Jinja2 → `report.html` (liste features par produit, avec entry_points linkables)
- `GraphVisualizer` : NetworkX → pyvis → `graph.html` (nœuds features = couleur distincte, clusters colorés par communauté)

---

## 5. Structure du projet

```
digitalFeaturesExtractor/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── base.py                 # Ingester ABC
│   │   ├── github_ingester.py
│   │   └── local_ingester.py
│   ├── graph/
│   │   ├── __init__.py
│   │   └── graphify_wrapper.py     # wraps graphify OSS API
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── feature_extractor.py    # clustering + orchestration LLM
│   │   ├── prompts.py              # prompts structurés
│   │   └── models.py               # Pydantic schemas
│   ├── output/
│   │   ├── __init__.py
│   │   ├── json_exporter.py
│   │   ├── html_reporter.py
│   │   └── graph_visualizer.py
│   └── cli.py                      # entrypoint CLI (Click)
├── tests/
│   ├── ingestion/
│   ├── extraction/
│   └── output/
├── docs/
│   └── superpowers/specs/
├── .featureextractor.yml.example   # template config
├── pyproject.toml
└── README.md
```

---

## 6. Interface CLI

```bash
# Analyser un repo GitHub
python -m digitalFeaturesExtractor analyze \
  --source https://github.com/org/repo \
  --llm-provider openai \
  --model gpt-4o \
  --output ./output/

# Analyser un repo local
python -m digitalFeaturesExtractor analyze \
  --source ./path/to/local/repo \
  --llm-provider anthropic \
  --model claude-sonnet-4-5 \
  --output ./output/

# Options
#   --include "src/**"      glob à inclure (défaut: tout)
#   --exclude "tests/**"    glob à exclure
#   --no-cache              désactiver le cache LLM
#   --min-confidence 0.4    filtrer les features sous ce seuil
```

Variables d'environnement : `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GITHUB_TOKEN`.

---

## 7. Gestion des erreurs

| Situation | Comportement |
|---|---|
| Repo privé sans token | Erreur explicite avec instructions pour `GITHUB_TOKEN` |
| Codebase > 100K lignes | Traitement par module/package, pas en monobloc |
| Cluster sans feature user-visible | Feature marquée `status: "To Review"`, `confidence_score < 0.5` |
| Aucun endpoint HTTP détecté | Fallback : classes publiques comme unités d'analyse |
| Langage non supporté par tree-sitter | Warning + skip, ne bloque pas le pipeline |
| Erreur LLM (rate limit, timeout) | Retry avec backoff exponentiel (3 tentatives) |

---

## 8. Hors scope (MVP)

- Mapping Digital Feature → Business Feature (v2)
- Mapping Business Feature → Business Use Case (v2)
- Interface web (UI graphique) — CLI uniquement pour le MVP
- Mode continu / webhook GitHub
- Stockage persistant (base de données)

---

## 9. Dépendances principales

| Dépendance | Usage |
|---|---|
| `graphify` | Core graph building (OSS) |
| `tree-sitter` | AST parsing Java/TS/Python |
| `networkx` | Manipulation du knowledge graph |
| `leidenalg` + `igraph` | Clustering des communautés |
| `openai` / `anthropic` | SDK LLM |
| `pydantic` | Validation des schemas |
| `jinja2` | Templates HTML |
| `pyvis` | Visualisation graph interactive |
| `click` | Interface CLI |
| `gitpython` | Clone de repos Git |
