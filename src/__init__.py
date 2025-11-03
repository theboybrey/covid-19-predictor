"""Package init for src.

Keep top-level imports minimal so importing `src` (e.g. `from src import evaluate`) does
not attempt to import the Kaggle client or run downloads. Provide a small entry
function to run the ingestion when the module is executed directly.
"""

def download_and_extract_entry():
    # import lazily so normal imports don't fail
    try:
        from .data_ingest import download_and_extract
    except Exception:
        # allow running from src/ directory as well
        from data_ingest import download_and_extract
    download_and_extract()


if __name__ == "__main__":
    download_and_extract_entry()
    print("🎉 Done.")