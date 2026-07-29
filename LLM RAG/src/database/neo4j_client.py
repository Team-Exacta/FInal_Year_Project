from neo4j import GraphDatabase
from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

class Neo4jClient:
    def __init__(self):
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self._driver = None

    def connect(self):
        if not self._driver:
            try:
                self._driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password))
                logger.info("Connected to Neo4j successfully.")
            except Exception as e:
                logger.error(f"Failed to connect to Neo4j: {e}")
                self._driver = None

    def close(self):
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed.")

    def query(self, query: str, parameters: dict = None):
        """Executes a read query and returns the results."""
        self.connect()
        if not self._driver:
            return []
            
        try:
            with self._driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Error executing Neo4j query: {e}")
            return []

    def execute_write(self, transaction_func, *args, **kwargs):
        """Executes a write transaction."""
        self.connect()
        if not self._driver:
            return
            
        try:
            with self._driver.session() as session:
                session.execute_write(transaction_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error executing Neo4j write transaction: {e}")
