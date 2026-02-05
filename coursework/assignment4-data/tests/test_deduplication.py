import logging

from xopen import xopen

from .adapters import run_exact_line_deduplication, run_minhash_deduplication
from .common import FIXTURES_PATH

logger = logging.getLogger(__name__)


def test_exact_line_deduplication(tmp_path):
    documents_with_line_duplicates_paths = list(
        (FIXTURES_PATH / "documents_with_line_duplicates").glob("doc*.txt")
    )
    documents_without_line_duplicates_paths = list(
        (FIXTURES_PATH / "documents_line_deduplicated").glob("doc*.txt")
    )
    # Load deduplicated documents
    deduplicated_documents = []
    for path in documents_without_line_duplicates_paths:
        with open(path) as f:
            deduplicated_documents.append(f.read())

    run_exact_line_deduplication(
        input_files=documents_with_line_duplicates_paths, output_directory=tmp_path
    )
    output_filepaths = list(tmp_path.glob("*"))

    assert len(output_filepaths) == 5
    for filepath in output_filepaths:
        with xopen(filepath) as f:
            output_file_contents = f.read()
            try:
                deduplicated_documents.remove(output_file_contents)
            except ValueError:
                raise ValueError(
                    f"Failed to find output file {filepath} contents {output_file_contents.__repr__()} in "
                    f"expected deduplicated documents {deduplicated_documents}."
                )
    assert len(deduplicated_documents) == 0


def test_minhash_deduplication_exact_duplicates(tmp_path):
    """
    Check that minhash deduplication properly identifies and removes exact duplicates.
    """
    documents_with_line_duplicates_paths = list(
        (FIXTURES_PATH / "documents_with_line_duplicates").glob("doc*.txt")
    )
    # Load deduplicated documents
    deduplicated_documents = []
    for path in documents_with_line_duplicates_paths:
        # NOTE: document 1 and document 2 are exact duplicates, so we only
        # want to keep one of them.
        if path.name == "doc2.txt":
            continue
        with open(path) as f:
            deduplicated_documents.append(f.read())

    run_minhash_deduplication(
        input_files=documents_with_line_duplicates_paths,
        output_directory=tmp_path,
        num_hashes=100,
        num_bands=10,
        ngrams=5,
        jaccard_threshold=0.8,
    )
    output_filepaths = list(tmp_path.glob("*"))
    assert len(output_filepaths) == 4
    for filepath in output_filepaths:
        with xopen(filepath) as f:
            output_file_contents = f.read()
            try:
                deduplicated_documents.remove(output_file_contents)
            except ValueError:
                raise ValueError(
                    f"Failed to find output file {filepath} contents {output_file_contents.__repr__()} in "
                    f"expected deduplicated documents {deduplicated_documents}."
                )
    assert len(deduplicated_documents) == 0


def test_minhash_deduplication_fuzzy_duplicates(tmp_path):
    """
    Check that minhash deduplication properly identifies and removes fuzzy
    duplicates (two documents with the MIT license, but with slightly different
    whitespace and attribution).
    """
    documents_with_fuzzy_duplicates_paths = list(
        (FIXTURES_PATH / "documents_with_fuzzy_duplicates").glob("*.txt")
    )
    # Load deduplicated documents
    deduplicated_documents = []
    kept_duplicated_documents = []
    for path in documents_with_fuzzy_duplicates_paths:
        # rails_mit_license.txt and react_mit_license.txt are fuzzy duplicates, so we only want
        # to keep one of them.
        with open(path) as f:
            if (
                path.name == "rails_mit_license.txt"
                or path.name == "react_mit_license.txt"
            ):
                kept_duplicated_documents.append(f.read())
            else:
                deduplicated_documents.append(f.read())

    run_minhash_deduplication(
        input_files=documents_with_fuzzy_duplicates_paths,
        output_directory=tmp_path,
        num_hashes=500,
        num_bands=50,
        ngrams=5,
        jaccard_threshold=0.8,
    )
    output_filepaths = list(tmp_path.glob("*"))
    assert len(output_filepaths) == 2
    for filepath in output_filepaths:
        with xopen(filepath) as f:
            output_file_contents = f.read()
            if output_file_contents in deduplicated_documents:
                deduplicated_documents.remove(output_file_contents)
            elif output_file_contents in kept_duplicated_documents:
                kept_duplicated_documents.remove(output_file_contents)
            else:
                raise ValueError(
                    f"Failed to find output file {filepath} contents {output_file_contents.__repr__()} in "
                    f"expected deduplicated documents {deduplicated_documents} or "
                    f"kept duplicated documents {kept_duplicated_documents}."
                )
    assert len(deduplicated_documents) == 0
    # One of the kept deduplicated documents should be kept, and the other should be removed.
    assert len(kept_duplicated_documents) == 1
