"""Compatibility example; the complete shared workflow lives in the pipeline module.

Run with --data (authorized local CSV/Parquet) and --split.
No data download, model publication or live trading is performed.
"""

from examples.research.index_enhance_lite_pipeline import LiteResearchIndexEnhance, main

LiteLiveIndexEnhance = LiteResearchIndexEnhance

if __name__ == "__main__":
    main()
