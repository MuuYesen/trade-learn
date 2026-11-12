"""Compatibility example; the complete shared workflow lives in the pipeline module.

Run with --data (authorized local CSV/Parquet) and --split.
No data download, model publication or live trading is performed.
"""

from examples.research.index_enhance_engine_pipeline import EngineResearchIndexEnhance, main

EngineLiveIndexEnhance = EngineResearchIndexEnhance

if __name__ == "__main__":
    main()
