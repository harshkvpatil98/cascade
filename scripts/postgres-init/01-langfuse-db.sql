-- Create the Langfuse database inside the same Postgres instance.
--
-- Runs once, from docker-entrypoint-initdb.d, on an empty data directory.
-- Langfuse v2 needs only Postgres (ADR-0006), so it shares this server rather
-- than adding a second one; it gets its own *database* so that a `make nuke`
-- of study data and a reset of telemetry stay separable, and so nothing
-- Langfuse creates can collide with the study schema.
--
-- Deliberately .sql rather than .sh. The entrypoint execs an init script it
-- considers executable, and a bind mount can report the executable bit while
-- still refusing the exec -- which fails as
-- `bad interpreter: Permission denied` and, because init scripts run before
-- the server accepts external connections, is easy to miss. A .sql file is
-- piped to psql and has no exec bit to get wrong.

SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'langfuse')
\gexec
