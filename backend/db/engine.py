from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import event, text
from sqlalchemy.pool import NullPool
from config import settings

# ── Connection string ──────────────────────────────────────────────────────────
# NullPool is mandatory for SQLite + aiosqlite.
# SQLAlchemy's default QueuePool may share a connection across threads,
# which corrupts SQLite (connections are not thread-safe).
DATABASE_URL = f"sqlite+aiosqlite:///{settings.db_path}"

# ── Async engine ───────────────────────────────────────────────────────────────
engine = create_async_engine(
    DATABASE_URL,
    poolclass=NullPool,                 # CRITICAL: no connection pooling for SQLite async
    echo=settings.debug_mode,          # Only log SQL in debug mode
    future=True,
)

# ── SQLite pragmas ─────────────────────────────────────────────────────────────
# Use the sync engine event — 'connect' fires on the raw DBAPI connection.
@event.listens_for(engine.sync_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")     # Write-Ahead Logging — safe concurrent reads
    cursor.execute("PRAGMA synchronous=NORMAL")   # Safe and fast (not FULL)
    cursor.execute("PRAGMA foreign_keys=ON")      # Enforce FK constraints & cascade deletes
    cursor.close()

# ── Session factory ────────────────────────────────────────────────────────────
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# ── Per-request session dependency ────────────────────────────────────────────
async def get_db():
    """
    FastAPI dependency that yields a scoped AsyncSession per request.
    Commits on success, rolls back on any exception, always closes.
    Services must call session.flush() to materialise IDs mid-transaction;
    they must NOT call session.commit() — that's exclusively done here.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
