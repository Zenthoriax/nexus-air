"""Initial tables with indexes

Revision ID: 7653360006f3
Revises: 
Create Date: 2026-03-07 16:38:37.771479

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7653360006f3'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'documents',
        sa.Column('id',           sa.String(length=36),  nullable=False),
        sa.Column('title',        sa.String(length=255), nullable=False),
        sa.Column('content',      sa.Text(),             nullable=False),
        sa.Column('tags',         sa.JSON(),             nullable=True),
        sa.Column('word_count',   sa.Integer(),          nullable=True),
        sa.Column('block_count',  sa.Integer(),          nullable=True),
        sa.Column('content_hash', sa.String(length=64),  nullable=True),
        sa.Column('created_at',   sa.DateTime(timezone=True), nullable=True),
        sa.Column('updated_at',   sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    # Deduplication lookup — every ingest checks content_hash first
    op.create_index('ix_documents_content_hash', 'documents', ['content_hash'])

    op.create_table(
        'tags',
        sa.Column('id',    sa.Integer(),       autoincrement=True, nullable=False),
        sa.Column('name',  sa.String(length=50), nullable=False),
        sa.Column('color', sa.String(length=7),  nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name'),
    )

    op.create_table(
        'blocks',
        sa.Column('id',         sa.String(length=36),  nullable=False),
        sa.Column('doc_id',     sa.String(length=36),  nullable=False),
        sa.Column('content',    sa.Text(),             nullable=False),
        sa.Column('block_type', sa.String(length=50),  nullable=True),
        sa.Column('position',   sa.Integer(),          nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['doc_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    # Embedding and retrieval lookups filter by doc_id
    op.create_index('ix_blocks_doc_id', 'blocks', ['doc_id'])

    op.create_table(
        'links',
        sa.Column('id',            sa.Integer(),        autoincrement=True, nullable=False),
        sa.Column('source_doc_id', sa.String(length=36), nullable=False),
        sa.Column('target_doc_id', sa.String(length=36), nullable=False),
        sa.Column('weight',        sa.Integer(),        nullable=True),
        sa.ForeignKeyConstraint(['source_doc_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_doc_id'], ['documents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    # Graph build queries filter by both source and target
    op.create_index('ix_links_source_doc_id', 'links', ['source_doc_id'])
    op.create_index('ix_links_target_doc_id', 'links', ['target_doc_id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_links_target_doc_id', table_name='links')
    op.drop_index('ix_links_source_doc_id', table_name='links')
    op.drop_table('links')

    op.drop_index('ix_blocks_doc_id', table_name='blocks')
    op.drop_table('blocks')

    op.drop_table('tags')

    op.drop_index('ix_documents_content_hash', table_name='documents')
    op.drop_table('documents')
