"""Research Paper Answer Bot -- source package.

Pipeline, in data-flow order:

* ``config``       -- settings from environment / ``.env``.
* ``ingestion``    -- PDF extraction and per-page chunking.
* ``embeddings``   -- open-source and OpenAI embedding backends.
* ``vector_store`` -- Zilliz Cloud Serverless collections, idempotent upsert.
* ``retrieval``    -- dense, hybrid (BM25 + dense) and reranked retrieval.
* ``rag``          -- grounded generation with top-3 citations and abstention.
* ``evaluation``   -- question set, metrics, configuration comparison.
* ``app_support``  -- helpers for the Streamlit app (``app.py``).
"""
