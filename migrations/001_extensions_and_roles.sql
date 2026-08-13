-- 001: extensions and role separation.
--
-- Establishes invariant 2 at the database boundary: the simulation process
-- must be structurally unable to read resolution labels. That is enforced by a
-- Postgres grant, not by code review, because code review is not a mechanism.
--
-- Forward-only. Never edit a migration that has been applied; add a new one.

BEGIN;

-- --------------------------------------------------------------------------
-- Extensions
-- --------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector: halfvec(384) + IVFFlat
CREATE EXTENSION IF NOT EXISTS pgcrypto;    -- gen_random_uuid() for run ids

-- --------------------------------------------------------------------------
-- Roles
--
-- Created NOSUPERUSER and NOBYPASSRLS explicitly. A superuser bypasses every
-- grant below, which would make the M1 grant test pass while the invariant it
-- claims to enforce is absent.
-- --------------------------------------------------------------------------

SELECT format('CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD %L',
              'cascade_sim', :'sim_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cascade_sim')
\gexec

SELECT format('CREATE ROLE %I LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS PASSWORD %L',
              'cascade_eval', :'eval_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'cascade_eval')
\gexec

-- Keep passwords in sync on re-run so a rotated secret does not require a
-- manual step outside the migration path.
SELECT format('ALTER ROLE %I PASSWORD %L', 'cascade_sim', :'sim_password')
\gexec
SELECT format('ALTER ROLE %I PASSWORD %L', 'cascade_eval', :'eval_password')
\gexec

-- --------------------------------------------------------------------------
-- Schema privileges: deny by default
--
-- REVOKE ... FROM cascade_sim alone is a no-op if the role was never granted
-- anything directly; what actually matters is that PUBLIC holds nothing,
-- because every role inherits PUBLIC.
-- --------------------------------------------------------------------------

REVOKE ALL ON SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM PUBLIC;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM PUBLIC;

GRANT USAGE ON SCHEMA public TO cascade_sim, cascade_eval;

-- Future tables created by the admin role inherit no PUBLIC grant either.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON SEQUENCES FROM PUBLIC;
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON FUNCTIONS FROM PUBLIC;

-- --------------------------------------------------------------------------
-- Migration bookkeeping
-- --------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     text        PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now(),
    checksum    text        NOT NULL
);

COMMIT;
