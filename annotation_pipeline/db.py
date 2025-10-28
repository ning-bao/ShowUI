import sqlite3
import os
import time
from pathlib import Path


DB_PATH = os.environ.get('ANNOTATION_DB_PATH', str(Path('data') / 'metadata.db'))


def _ensure_db_dir():
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def get_conn():
    _ensure_db_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            parent_path TEXT,
            name TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            folder_path TEXT,
            has_annotation INTEGER NOT NULL DEFAULT 0,
            size_bytes INTEGER,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            FOREIGN KEY(folder_path) REFERENCES folders(path) ON DELETE SET NULL
        )
        '''
    )
    cur.execute('CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder_path)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_images_has_annotation ON images(has_annotation)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_folders_parent ON folders(parent_path)')
    conn.commit()
    conn.close()


def _normalize_path(p: str) -> str:
    return str(p).replace('\\', '/')


def upsert_folder(path: str):
    path = _normalize_path(path).strip('/')
    if path == '':
        return
    parent = '/'.join(path.split('/')[:-1]) if '/' in path else None
    name = path.split('/')[-1]
    now = time.time()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO folders(path, parent_path, name, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            parent_path=excluded.parent_path,
            name=excluded.name,
            updated_at=excluded.updated_at
        ''',
        (path, parent, name, now, now)
    )
    conn.commit()
    conn.close()


def ensure_folder_chain(path: str):
    path = _normalize_path(path).strip('/')
    if not path:
        return
    parts = path.split('/')
    chain = []
    for i in range(len(parts)):
        chain.append('/'.join(parts[: i + 1]))
    for p in chain:
        upsert_folder(p)


def upsert_image(rel_path: str, has_annotation: bool = False, size_bytes: int | None = None):
    rel_path = _normalize_path(rel_path).strip('/')
    folder_path = '/'.join(rel_path.split('/')[:-1]) if '/' in rel_path else None
    filename = rel_path.split('/')[-1]
    if folder_path:
        ensure_folder_chain(folder_path)
    now = time.time()
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO images(path, filename, folder_path, has_annotation, size_bytes, created_at, updated_at)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            filename=excluded.filename,
            folder_path=excluded.folder_path,
            has_annotation=excluded.has_annotation,
            size_bytes=excluded.size_bytes,
            updated_at=excluded.updated_at
        ''',
        (rel_path, filename, folder_path, 1 if has_annotation else 0, size_bytes, now, now)
    )
    conn.commit()
    conn.close()


def set_has_annotation(rel_path: str, has_annotation: bool):
    rel_path = _normalize_path(rel_path).strip('/')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('UPDATE images SET has_annotation = ?, updated_at = ? WHERE path = ?', (1 if has_annotation else 0, time.time(), rel_path))
    conn.commit()
    conn.close()


def delete_image(rel_path: str):
    rel_path = _normalize_path(rel_path).strip('/')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM images WHERE path = ?', (rel_path,))
    conn.commit()
    conn.close()


def delete_folder(folder_path: str):
    """Delete a folder from the database."""
    folder_path = _normalize_path(folder_path).strip('/')
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('DELETE FROM folders WHERE path = ?', (folder_path,))
    conn.commit()
    conn.close()


def list_images(limit: int | None = None, offset: int | None = None):
    conn = get_conn()
    cur = conn.cursor()
    sql = 'SELECT path as filename, has_annotation FROM images ORDER BY filename'
    if limit is not None:
        sql += ' LIMIT ?'
        if offset is not None:
            sql += ' OFFSET ?'
            cur.execute(sql, (limit, offset))
        else:
            cur.execute(sql, (limit,))
    else:
        cur.execute(sql)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def count_images() -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT COUNT(1) FROM images')
    (cnt,) = cur.fetchone()
    conn.close()
    return int(cnt)


def list_all_folders() -> list[str]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute('SELECT path FROM folders ORDER BY path')
    rows = [r['path'] for r in cur.fetchall()]
    conn.close()
    return rows


