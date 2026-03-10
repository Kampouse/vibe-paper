"""SQLite database for Always-On Memory"""
import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Memory:
    id: int
    summary: str
    entities: List[str]
    topics: List[str]
    source: str
    importance: float
    created_at: datetime
    consolidated: bool


@dataclass
class Insight:
    id: int
    memory_ids: List[int]
    connection_type: str
    insight: str
    created_at: datetime


class MemoryDB:
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._setup_schema()
    
    def _setup_schema(self):
        """Create tables if they don't exist"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                summary TEXT NOT NULL,
                entities JSON DEFAULT '[]',
                topics JSON DEFAULT '[]',
                source TEXT,
                importance REAL DEFAULT 0.5,
                raw_content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                consolidated BOOLEAN DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_ids JSON NOT NULL,
                connection_type TEXT,
                insight TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER,
                reward REAL,
                hint TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (memory_id) REFERENCES memories(id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_created 
            ON memories(created_at);
            
            CREATE INDEX IF NOT EXISTS idx_memories_consolidated 
            ON memories(consolidated);
        """)
        self.conn.commit()
    
    def store_memory(
        self,
        summary: str,
        entities: List[str],
        topics: List[str],
        source: str,
        importance: float = 0.5,
        raw_content: str = None
    ) -> int:
        """Store a new memory"""
        cursor = self.conn.execute(
            """INSERT INTO memories 
               (summary, entities, topics, source, importance, raw_content)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (summary, json.dumps(entities), json.dumps(topics), 
             source, importance, raw_content)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def store_insight(
        self,
        memory_ids: List[int],
        connection_type: str,
        insight: str
    ) -> int:
        """Store a consolidation insight"""
        cursor = self.conn.execute(
            """INSERT INTO insights 
               (memory_ids, connection_type, insight)
               VALUES (?, ?, ?)""",
            (json.dumps(memory_ids), connection_type, insight)
        )
        self.conn.commit()
        return cursor.lastrowid
    
    def store_feedback(
        self,
        memory_id: int,
        reward: float,
        hint: str = None
    ):
        """Store feedback for RL training"""
        self.conn.execute(
            """INSERT INTO feedback 
               (memory_id, reward, hint)
               VALUES (?, ?, ?)""",
            (memory_id, reward, hint)
        )
        self.conn.commit()
    
    def get_unconsolidated(self, limit: int = 50) -> List[Dict]:
        """Get memories that haven't been consolidated"""
        rows = self.conn.execute(
            """SELECT * FROM memories 
               WHERE consolidated = 0 
               ORDER BY created_at DESC 
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def mark_consolidated(self, memory_ids: List[int]):
        """Mark memories as consolidated"""
        placeholders = ','.join('?' * len(memory_ids))
        self.conn.execute(
            f"""UPDATE memories 
                SET consolidated = 1 
                WHERE id IN ({placeholders})""",
            memory_ids
        )
        self.conn.commit()
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search (no embeddings)"""
        rows = self.conn.execute(
            """SELECT * FROM memories 
               WHERE summary LIKE ? 
               OR raw_content LIKE ?
               ORDER BY importance DESC, created_at DESC
               LIMIT ?""",
            (f"%{query}%", f"%{query}%", limit)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_recent(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent memories"""
        rows = self.conn.execute(
            """SELECT * FROM memories 
               WHERE created_at >= datetime('now', ?)
               ORDER BY created_at DESC
               LIMIT ?""",
            (f'-{hours} hours', limit)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_all_insights(self, limit: int = 20) -> List[Dict]:
        """Get all consolidation insights"""
        rows = self.conn.execute(
            """SELECT * FROM insights 
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,)
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        stats = {}
        stats['total_memories'] = self.conn.execute(
            "SELECT COUNT(*) FROM memories"
        ).fetchone()[0]
        stats['total_insights'] = self.conn.execute(
            "SELECT COUNT(*) FROM insights"
        ).fetchone()[0]
        stats['unconsolidated'] = self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE consolidated = 0"
        ).fetchone()[0]
        stats['total_feedback'] = self.conn.execute(
            "SELECT COUNT(*) FROM feedback"
        ).fetchone()[0]
        return stats
    
    def _row_to_dict(self, row) -> Dict:
        """Convert sqlite3.Row to dict, parsing JSON fields"""
        d = dict(row)
        # Parse JSON fields
        if 'entities' in d and isinstance(d['entities'], str):
            d['entities'] = json.loads(d['entities'])
        if 'topics' in d and isinstance(d['topics'], str):
            d['topics'] = json.loads(d['topics'])
        if 'memory_ids' in d and isinstance(d['memory_ids'], str):
            d['memory_ids'] = json.loads(d['memory_ids'])
        return d
    
    def close(self):
        """Close database connection"""
        self.conn.close()
