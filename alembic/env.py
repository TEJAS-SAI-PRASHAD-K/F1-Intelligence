"""Alembic environment, configured for the raw / features / ml schemas.

Two settings matter here and are annoying to retrofit later:

  include_schemas=True       -- without it, autogenerate ignores our non-public
                                schemas and will try to DROP every table it
                                cannot see.
  version_table_schema="ml"  -- keeps alembic_version out of `public`.

`include_object` additionally stops Alembic from trying to manage objects in
schemas it does not own (e.g. views we create by hand in `features`).
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from app.config import get_settings
from app.db import models  # noqa: F401  -- import registers tables on Base.metadata
from app.db.base import SCHEMA_FEATURES, SCHEMA_ML, SCHEMA_RAW, Base

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

MANAGED_SCHEMAS = {SCHEMA_RAW, SCHEMA_ML, None}


def include_object(obj, name, type_, reflected, compare_to):
    """Only manage objects in schemas we own."""
    if type_ == "table":
        return obj.schema in MANAGED_SCHEMAS
    return True


def _configure(**kwargs):
    context.configure(
        target_metadata=target_metadata,
        include_schemas=True,
        include_object=include_object,
        version_table="alembic_version",
        version_table_schema=SCHEMA_ML,
        compare_type=True,
        compare_server_default=True,
        **kwargs,
    )


def run_migrations_offline() -> None:
    _configure(
        url=config.get_main_option("sqlalchemy.url"),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        _configure(connection=connection)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
