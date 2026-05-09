# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.utils.auth import Secret
from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore


class TestMariaDBDocumentStoreInit:
    """Tests for MariaDBDocumentStore initialization — no DB connection needed."""

    def test_init_default_params(self):
        """Store can be created with default parameters."""
        store = MariaDBDocumentStore(
            user="test_user",      # pass directly to avoid env var requirement
            password="test_pass",
        )
        assert store.host == "localhost"
        assert store.port == 3306
        assert store.table_name == "haystack_documents"
        assert store.embedding_dimension == 768
        assert store.vector_function == "cosine_similarity"
        assert store.recreate_table is False

    def test_init_custom_params(self):
        """Store accepts custom parameters correctly."""
        store = MariaDBDocumentStore(
            host="myhost",
            port=3307,
            database="mydb",
            user=Secret.from_token("myuser"),
            password=Secret.from_token("mypass"),
            table_name="custom_docs",
            embedding_dimension=1536,
            vector_function="l2_distance",
            recreate_table=True,
        )
        assert store.host == "myhost"
        assert store.port == 3307
        assert store.database == "mydb"
        assert store.table_name == "custom_docs"
        assert store.embedding_dimension == 1536
        assert store.vector_function == "l2_distance"
        assert store.recreate_table is True

    def test_invalid_vector_function_raises(self):
        """Invalid vector_function raises ValueError at construction time."""
        with pytest.raises(ValueError, match="vector_function must be one of"):
            MariaDBDocumentStore(
                user="test",
                password="test",
                vector_function="invalid_function",
            )

    def test_connection_is_lazy(self):
        """No DB connection is made during __init__."""
        store = MariaDBDocumentStore(
            user="nonexistent_user",
            password="wrong_password",
        )
        # connection should still be None — no DB call yet
        assert store._connection is None
        assert store._cursor is None
        assert store._table_initialized is False

    def test_to_dict(self, monkeypatch):
        """Store serializes to dict correctly."""
        monkeypatch.setenv("MARIADB_USER", "myuser")
        monkeypatch.setenv("MARIADB_PASSWORD", "mypass")
        store = MariaDBDocumentStore(
            host="localhost",
            port=3306,
            database="haystack",
            user=Secret.from_env_var("MARIADB_USER"),
            password=Secret.from_env_var("MARIADB_PASSWORD"),
            table_name="haystack_documents",
            embedding_dimension=768,
            vector_function="cosine_similarity",
            recreate_table=False,
        )
        d = store.to_dict()
        assert d["type"] == "haystack_integrations.document_stores.mariadb.document_store.MariaDBDocumentStore"
        params = d["init_parameters"]
        assert params["host"] == "localhost"
        assert params["port"] == 3306
        assert params["database"] == "haystack"
        assert params["table_name"] == "haystack_documents"
        assert params["embedding_dimension"] == 768
        assert params["vector_function"] == "cosine_similarity"
        assert params["recreate_table"] is False

    def test_from_dict(self):
        """Store deserializes from dict correctly."""
        data = {
            "type": "haystack_integrations.document_stores.mariadb.document_store.MariaDBDocumentStore",
            "init_parameters": {
                "host": "localhost",
                "port": 3306,
                "database": "haystack",
                "user": {"type": "env_var", "env_vars": ["MARIADB_USER"], "strict": True},
                "password": {"type": "env_var", "env_vars": ["MARIADB_PASSWORD"], "strict": True},
                "table_name": "custom_docs",
                "embedding_dimension": 768,
                "vector_function": "cosine_similarity",
                "recreate_table": False,
            },
        }
        store = MariaDBDocumentStore.from_dict(data)
        assert store.host == "localhost"
        assert store.embedding_dimension == 768


@pytest.mark.integration
class TestMariaDBDocumentStoreIntegration:
    """
    Integration tests — require a running MariaDB 11.7+ instance.

    Run with:
        docker run -d --name mariadb-haystack \
            -e MARIADB_ROOT_PASSWORD=password \
            -e MARIADB_DATABASE=haystack \
            -p 3306:3306 mariadb:11.7

        pytest -m integration tests/
    """

    @pytest.fixture
    def document_store(self):
        """Creates a fresh document store for each test."""
        store = MariaDBDocumentStore(
            host="localhost",
            port=3306,
            database="haystack",
            user="root",
            password="password",
            table_name="test_haystack_documents",
            recreate_table=True,  # start fresh each test
        )
        yield store
        store.close()

    def test_count_documents_empty(self, document_store):
        """Empty store returns count of 0."""
        assert document_store.count_documents() == 0

    def test_delete_documents_empty_list(self, document_store):
        """Deleting empty list does not raise."""
        document_store.delete_documents([])  # should not raise