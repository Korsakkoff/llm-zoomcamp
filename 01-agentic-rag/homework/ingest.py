import requests
from minsearch import Index
from gitsource import GithubRepositoryDataReader
from gitsource import chunk_documents


def load_github_data():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    return documents


def build_index(documents):

    chunks = chunk_documents(documents, size=2000, step=1000)

    index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
    index.fit(chunks)
    return index
