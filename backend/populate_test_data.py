import asyncio
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from models.orm import Base, Document, Block, Link
from config import settings

DATABASE_URL = f"sqlite+aiosqlite:///{settings.db_path}"

async def populate():
    engine = create_async_engine(DATABASE_URL, future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with async_session() as session:
        # Create some test documents
        doc1_id = str(uuid.uuid4())
        doc2_id = str(uuid.uuid4())
        doc3_id = str(uuid.uuid4())
        doc4_id = str(uuid.uuid4())

        docs = [
            Document(id=doc1_id, title="Quantum Computing Basics", content="Introduction to qubits and entanglement.", word_count=50, block_count=1),
            Document(id=doc2_id, title="Cryptography in the Quantum Age", content="How Shor's algorithm affects RSA.", word_count=60, block_count=1),
            Document(id=doc3_id, title="RSA Algorithm", content="The math behind public key encryption.", word_count=40, block_count=1),
            Document(id=doc4_id, title="Future of AI", content="Artificial General Intelligence and beyond.", word_count=70, block_count=1),
        ]
        
        session.add_all(docs)
        await session.flush()

        # Create some links
        links = [
            Link(source_doc_id=doc1_id, target_doc_id=doc2_id, weight=2),
            Link(source_doc_id=doc2_id, target_doc_id=doc3_id, weight=1),
            Link(source_doc_id=doc1_id, target_doc_id=doc3_id, weight=1),
        ]
        
        session.add_all(links)
        await session.commit()
        
    print(f"Populated database with 4 documents and 3 links.")
    print(f"Sample Document ID for traversal: {doc1_id}")

if __name__ == "__main__":
    asyncio.run(populate())
