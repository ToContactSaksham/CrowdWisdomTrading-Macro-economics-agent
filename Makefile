.PHONY: run offline test clean
run:        ## live run (needs APIFY_TOKEN / TAVILY_API_KEY)
	python run_pipeline.py
offline:    ## reproducible run on synthetic fallbacks
	OFFLINE=1 RUN_DATE=2026-10-07 python run_pipeline.py
test:
	python -m pytest -q tests
clean:
	rm -rf outputs/* data/processed/* data/raw/*fallback* data/raw/trading_logs.csv
