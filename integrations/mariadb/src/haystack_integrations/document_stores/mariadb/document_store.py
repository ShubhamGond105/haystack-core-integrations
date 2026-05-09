# SPDX-FileCopyrightText: 2023-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any, Literal, Optional

import mariadb

from haystack import default_from_dict, default_to_dict
from haystack.dataclasses.document import Document
from haystack.document_stores.errors import DocumentStoreError, DuplicateDocumentError
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils.auth import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

CREATE_TABLE_STATEMENT = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id VARCHAR(128) PRIMARY KEY,
    embedding VECTOR({embedding_dimension}) COMMENT 'MHNSW(M=16)',
    content LONGTEXT,
    blob_data LONGBLOB,
    blob_meta JSON,
    blob_mime_type VARCHAR(255),
    meta JSON
)
"""

INSERT_STATEMENT = """
INSERT INTO {table_name}
(id, embedding, content, blob_data, blob_meta, blob_mime_type, meta)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

UPDATE_ON_DUPLICATE_STATEMENT = """
ON DUPLICATE KEY UPDATE
    embedding = VALUES(embedding),
    content = VALUES(content),
    blob_data = VALUES(blob_data),
    blob_meta = VALUES(blob_meta),
    blob_mime_type = VALUES(blob_mime_type),
    meta = VALUES(meta)
"""

VALID_VECTOR_FUNCTIONS = ["cosine_similarity", "l2_distance"]

VECTOR_FUNCTION_TO_MARIADB_FUNC = {
    "cosine_similarity": "VEC_DISTANCE_COSINE",
    "l2_distance": "VEC_DISTANCE_EUCLIDEAN",
}


class MariaDBDocumentStore(DocumentStore):
    """
    A Document Store backed by MariaDB 11.7+.

    Uses MariaDB's native VECTOR datatype and MHNSW indexing for vector similarity search.

    Usage example:
    ```python
    from haystack_integrations.document_stores.mariadb import MariaDBDocumentStore

    document_store = MariaDBDocumentStore(
        host="localhost",
        port=3306,
        database="haystack",
        embedding_dimension=768,
    )
    ```
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 3306,
        database: str = "haystack",
        user: Secret = Secret.from_env_var("MARIADB_USER"),
        password: Secret = Secret.from_env_var("MARIADB_PASSWORD"),
        table_name: str = "haystack_documents",
        embedding_dimension: int = 768,
        vector_function: Literal["cosine_similarity", "l2_distance"] = "cosine_similarity",
        recreate_table: bool = False,
    ) -> None:
        """
        Creates a new MariaDBDocumentStore instance.

        Connection to MariaDB is established lazily on first use.
        A table to store Haystack documents will be created if it doesn't exist.

        :param host: The MariaDB host. Defaults to "localhost".
        :param port: The MariaDB port. Defaults to 3306.
        :param database: The name of the database to connect to.
        :param user: The database user. Read from MARIADB_USER env var by default.
        :param password: The database password. Read from MARIADB_PASSWORD env var by default.
        :param table_name: The name of the table to store Haystack documents.
        :param embedding_dimension: The dimension of the embedding vectors.
        :param vector_function: The similarity function for vector search.
            "cosine_similarity" uses VEC_DISTANCE_COSINE (lower = more similar).
            "l2_distance" uses VEC_DISTANCE_EUCLIDEAN (lower = more similar).
        :param recreate_table: Whether to drop and recreate the table on init.
        """
        # Initialize connection attributes first so __del__ is safe
        self._connection: Optional[mariadb.Connection] = None
        self._cursor: Optional[mariadb.Cursor] = None
        self._table_initialized = False

        if vector_function not in VALID_VECTOR_FUNCTIONS:
            msg = f"vector_function must be one of {VALID_VECTOR_FUNCTIONS}, but got {vector_function}"
            raise ValueError(msg)

        self.host = host
        self.port = port
        self.database = database

        # Convert plain strings to Secret objects for consistent handling
        if isinstance(user, str):
            self.user = Secret.from_token(user)
        else:
            self.user = user

        if isinstance(password, str):
            self.password = Secret.from_token(password)
        else:
            self.password = password
        self.table_name = table_name
        self.embedding_dimension = embedding_dimension
        self.vector_function = vector_function
        self.recreate_table = recreate_table

    def to_dict(self) -> dict[str, Any]:
        """Serializes the component to a dictionary."""
        return default_to_dict(
            self,
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user.to_dict(),
            password=self.password.to_dict(),
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=self.recreate_table,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MariaDBDocumentStore":
        """Deserializes the component from a dictionary."""
        deserialize_secrets_inplace(data["init_parameters"], ["user", "password"])
        return default_from_dict(cls, data)

    def _ensure_db_setup(self) -> None:
        """
        Establishes DB connection and initializes the table if not done yet.
        Called lazily before any DB operation.
        """
        # Reuse existing valid connection
        if self._connection is not None and self._cursor is not None:
            try:
                self._connection.ping()
                return
            except mariadb.Error:
                # Connection dropped — close and reconnect
                self.close()

        user = self.user.resolve_value() or ""
        password = self.password.resolve_value() or ""

        try:
            self._connection = mariadb.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=user,
                password=password,
                autocommit=True,
            )
        except mariadb.Error as e:
            msg = (
                f"Failed to connect to MariaDB at {self.host}:{self.port}/{self.database}. "
                "Check your credentials and that MariaDB 11.7+ is running."
            )
            raise DocumentStoreError(msg) from e

        self._cursor = self._connection.cursor(dictionary=True)

        if not self._table_initialized:
            self._initialize_table()

    def _initialize_table(self) -> None:
        """Creates the documents table if it doesn't exist."""
        if self.recreate_table:
            self.delete_table()

        sql = CREATE_TABLE_STATEMENT.format(
            table_name=self.table_name,
            embedding_dimension=self.embedding_dimension,
        )

        try:
            self._cursor.execute(sql)
        except mariadb.Error as e:
            msg = f"Could not create table '{self.table_name}'"
            raise DocumentStoreError(msg) from e

        self._table_initialized = True

    def delete_table(self) -> None:
        """Drops the documents table if it exists."""
        try:
            self._cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            self._table_initialized = False
        except mariadb.Error as e:
            msg = f"Could not delete table '{self.table_name}'"
            raise DocumentStoreError(msg) from e

    def close(self) -> None:
        """Closes the database connection."""
        if self._cursor is not None:
            try:
                self._cursor.close()
            except Exception:  # noqa: BLE001
                pass
            self._cursor = None

        if self._connection is not None:
            try:
                self._connection.close()
            except Exception:  # noqa: BLE001
                pass
            self._connection = None

        self._table_initialized = False

    def __del__(self) -> None:
        self.close()

    def count_documents(self) -> int:
        """Returns how many documents are stored."""
        self._ensure_db_setup()
        self._cursor.execute(f"SELECT COUNT(*) as cnt FROM {self.table_name}")
        result = self._cursor.fetchone()
        return result["cnt"] if result else 0

    def filter_documents(self, filters: dict[str, Any] | None = None) -> list[Document]:
        """Returns documents matching the given filters. Full implementation in next commit."""
        self._ensure_db_setup()
        # TODO: implement filter conversion (next commit)
        self._cursor.execute(f"SELECT * FROM {self.table_name}")
        records = self._cursor.fetchall()
        return _from_mariadb_to_haystack_documents(records)

    def write_documents(self, documents: list[Document], policy: DuplicatePolicy = DuplicatePolicy.FAIL) -> int:
        """Writes documents to the store. Full implementation in next commit."""
        # TODO: implement (next commit)
        raise NotImplementedError("write_documents will be implemented in next commit")

    def delete_documents(self, document_ids: list[str]) -> None:
        """Deletes documents by ID."""
        if not document_ids:
            return
        self._ensure_db_setup()
        placeholders = ", ".join(["?"] * len(document_ids))
        try:
            self._cursor.execute(
                f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})",
                tuple(document_ids),
            )
        except mariadb.Error as e:
            msg = "Could not delete documents from MariaDBDocumentStore"
            raise DocumentStoreError(msg) from e

def _from_mariadb_to_haystack_documents(records: list[dict[str, Any]]) -> list[Document]:
    """Converts raw MariaDB rows to Haystack Document objects."""
    from dataclasses import replace
    from haystack.dataclasses import ByteStream

    haystack_documents = []
    for record in records:
        haystack_dict = dict(record)

        blob_data = haystack_dict.pop("blob_data", None)
        blob_meta = haystack_dict.pop("blob_meta", None)
        blob_mime_type = haystack_dict.pop("blob_mime_type", None)

        # MariaDB returns JSON as string — parse it
        if isinstance(haystack_dict.get("meta"), str):
            haystack_dict["meta"] = json.loads(haystack_dict["meta"])

        # Convert embedding bytes to list of floats if present
        if haystack_dict.get("embedding") is not None:
            emb = haystack_dict["embedding"]
            if hasattr(emb, "tolist"):
                haystack_dict["embedding"] = emb.tolist()
            elif isinstance(emb, (bytes, bytearray)):
                import struct
                n = len(emb) // 4
                haystack_dict["embedding"] = list(struct.unpack(f"{n}f", emb))

        # Remove None meta to avoid Document.from_dict issues
        if haystack_dict.get("meta") is None:
            haystack_dict.pop("meta", None)

        doc = Document.from_dict(haystack_dict)

        if blob_data:
            if isinstance(blob_meta, str):
                blob_meta = json.loads(blob_meta)
            blob = ByteStream(data=blob_data, meta=blob_meta or {}, mime_type=blob_mime_type)
            doc = replace(doc, blob=blob)

        haystack_documents.append(doc)

    return haystack_documents