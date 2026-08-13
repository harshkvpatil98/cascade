"""Configuration: precedence, immutability, and secrets that stay out of YAML.

``configs/base.yaml`` is the single source of study tunables, so this module
asserts against the *shipped* file rather than a fixture. A drift between what
the config says and what the code assumes then shows up here instead of at M7.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from cascade.config import Settings, load_settings, repo_root


def test_base_config_loads() -> None:
    settings = Settings()
    assert settings.study.name == "cascade-v1"
    assert settings.kernel.steps == 24
    assert settings.ensemble.replicates == 200


def test_the_measurement_contract_constants_match_the_spec() -> None:
    """Guard the numbers the study's claims are computed from.

    Not targets -- these are *inputs*. A silent edit to ``steps`` or
    ``replicates`` changes what 36,000 runs even means.
    """
    settings = Settings()
    assert settings.kernel.steps == 24
    assert settings.ensemble.replicates == 200
    assert settings.ensemble.ablation_replicates == 30
    assert settings.ensemble.bootstrap_b == 10_000
    assert settings.ensemble.sigma_multimodal_threshold == 0.30
    assert settings.kernel.max_step_delta == 0.12
    assert settings.kernel.contest_gamma == 1.6
    assert settings.retrieval.target_p95_ms == 15


def test_pinned_models_are_the_ones_the_spec_names() -> None:
    settings = Settings()
    assert settings.models.agent == "claude-haiku-4-5-20251001"
    assert settings.models.compiler == "claude-sonnet-4-6"
    assert settings.models.embedding == "BAAI/bge-small-en-v1.5"


def test_settings_are_frozen() -> None:
    """Config is immutable within a run, so nothing can retune mid-study."""
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.kernel.steps = 12  # type: ignore[misc]


def test_unknown_keys_are_rejected() -> None:
    """A typo'd override must fail, not be silently ignored."""
    with pytest.raises(ValidationError):
        Settings(not_a_real_field=1)  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "env_var",
    [
        "CASCADE_ENSEMBLE_REPLICATES",  # single underscore where `__` belongs
        "CASCADE_KERNAL__STEPS",  # typo in the section name
        "CASCADE_NOT_A_FIELD",
    ],
)
def test_an_override_that_binds_to_nothing_is_an_error(
    monkeypatch: pytest.MonkeyPatch, env_var: str
) -> None:
    """A root-segment typo is discarded by pydantic-settings without a word.

    ``extra="forbid"`` only sees variables that were harvested, and a variable
    whose root does not name a field is never harvested. Left alone, the study
    runs on the default while the operator believes it was overridden -- and
    nothing in the report would show the difference.
    """
    monkeypatch.setenv(env_var, "50")
    with pytest.raises(ValidationError, match="unrecognised CASCADE_ environment"):
        Settings()


def test_a_typo_inside_a_known_section_is_also_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `extra="forbid"` path, which covers the other half of the space."""
    monkeypatch.setenv("CASCADE_KERNEL__STEPZ", "12")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    "env_var",
    ["CASCADE_CONFIG", "CASCADE_ENV_FILE", "CASCADE_BUDGET__PHASE_CEILING_USD__SIMULATE"],
)
def test_legitimate_variables_are_not_rejected(
    monkeypatch: pytest.MonkeyPatch, env_var: str, tmp_path: Path
) -> None:
    """Loader-steering variables and dict-keyed leaves must still work.

    ``phase_ceiling_usd`` is a dict, so its leaf names are data rather than
    fields; checking only the root segment is what keeps them valid.
    """
    value = str(repo_root() / "configs" / "base.yaml") if env_var == "CASCADE_CONFIG" else "5.0"
    if env_var == "CASCADE_ENV_FILE":
        value = str(tmp_path / "absent.env")
    monkeypatch.setenv(env_var, value)
    settings = Settings()
    if env_var.endswith("SIMULATE"):
        assert settings.phase_ceiling("simulate") == Decimal("5.0")


def test_prices_are_decimal_not_float() -> None:
    """Binary floats would break the meter's 6-dp criterion for unrelated reasons."""
    entry = Settings().price_for("claude-haiku-4-5-20251001")
    assert isinstance(entry.input_per_mtok, Decimal)
    assert entry.input_per_mtok == Decimal("1.00")
    assert entry.output_per_mtok == Decimal("5.00")


def test_yaml_decimals_do_not_inherit_binary_float_error() -> None:
    """``Decimal(0.1)`` is 0.1000000000000000055...; ``Decimal('0.1')`` is not."""
    multipliers = Settings().pricing_multipliers
    assert multipliers.cache_read == Decimal("0.1")
    assert multipliers.batch == Decimal("0.5")
    assert multipliers.cache_write_1h == Decimal("2.0")


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_LLM__MODE", "record")
    assert Settings().llm.mode == "record"


def test_nested_env_override_uses_the_double_underscore_delimiter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CASCADE_KERNEL__ACTIVATION__MAX_ACTIVE_PER_STEP", "3")
    assert Settings().kernel.activation.max_active_per_step == 3


def test_explicit_arguments_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_LLM__MODE", "record")
    base = Settings()
    overridden = base.model_copy(update={"llm": base.llm.model_copy(update={"mode": "live"})})
    assert overridden.llm.mode == "live"


# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------


def test_secrets_are_absent_from_yaml() -> None:
    """The shipped config must contain no credential, in any form.

    Comments are stripped before scanning: base.yaml deliberately *documents*
    which environment variable carries each secret, and that pointer is the
    thing keeping the value itself out of the file.
    """
    text = (repo_root() / "configs" / "base.yaml").read_text(encoding="utf-8")
    settings_only = "\n".join(line.split("#", 1)[0] for line in text.lower().splitlines())
    for forbidden in ("sk-ant", "password", "secret_key", "api_key"):
        assert forbidden not in settings_only, f"{forbidden!r} must not appear in base.yaml"


@pytest.mark.parametrize(
    ("env_var", "attribute"),
    [
        ("CASCADE_ANTHROPIC_API_KEY", "anthropic_api_key"),
        ("CASCADE_DB_ADMIN_PASSWORD", "db_admin_password"),
        ("CASCADE_DB_SIM_PASSWORD", "db_sim_password"),
        ("CASCADE_DB_EVAL_PASSWORD", "db_eval_password"),
        ("CASCADE_LANGFUSE_PUBLIC_KEY", "langfuse_public_key"),
        ("CASCADE_LANGFUSE_SECRET_KEY", "langfuse_secret_key"),
    ],
)
def test_documented_secret_variables_actually_bind(
    monkeypatch: pytest.MonkeyPatch, env_var: str, attribute: str
) -> None:
    """Every secret env var named in the docs must reach its field.

    Secrets are top-level ``Settings`` fields, so they take a *single*
    underscore -- the ``__`` nesting delimiter does not apply. Writing
    ``CASCADE_DB__SIM_PASSWORD`` binds nothing and reports no error: the field
    stays ``None`` and the failure surfaces later as an empty password. This
    test pins the working spelling so the documentation cannot drift from it.
    """
    monkeypatch.setenv(env_var, "value-from-env")
    secret = getattr(Settings(), attribute)
    assert secret is not None, f"{env_var} did not bind to Settings.{attribute}"
    assert secret.get_secret_value() == "value-from-env"


def test_env_var_names_in_base_yaml_and_env_example_agree() -> None:
    """A documented variable that does not bind sends the reader in circles."""
    yaml_text = (repo_root() / "configs" / "base.yaml").read_text(encoding="utf-8")
    example_text = (repo_root() / ".env.example").read_text(encoding="utf-8")
    for env_var in ("CASCADE_DB_SIM_PASSWORD", "CASCADE_LANGFUSE_PUBLIC_KEY"):
        assert env_var in yaml_text
        assert env_var in example_text
    # The non-binding double-underscore spelling must not be documented anywhere.
    assert "CASCADE_DB__" not in yaml_text
    assert "CASCADE_LANGFUSE__PUBLIC_KEY" not in yaml_text


def test_dot_env_supplies_secrets_to_a_plain_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`cascade doctor` must work without going through the Makefile."""
    env_file = tmp_path / "dotenv"
    env_file.write_text("CASCADE_DB_SIM_PASSWORD=from-dot-env\n", encoding="utf-8")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))

    secret = Settings().db_sim_password
    assert secret is not None
    assert secret.get_secret_value() == "from-dot-env"


def test_real_environment_beats_dot_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Documented precedence: env vars outrank the file they default from."""
    env_file = tmp_path / "dotenv"
    env_file.write_text("CASCADE_DB_SIM_PASSWORD=from-dot-env\n", encoding="utf-8")
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))
    monkeypatch.setenv("CASCADE_DB_SIM_PASSWORD", "from-shell")

    secret = Settings().db_sim_password
    assert secret is not None
    assert secret.get_secret_value() == "from-shell"


def test_dot_env_may_carry_compose_only_variables(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`.env` is shared with docker compose and holds two namespaces.

    ``LANGFUSE_ENCRYPTION_KEY`` and friends are consumed by compose, not by
    this model. They must not be mistaken for typo'd settings -- but a genuine
    typo in a ``CASCADE_``-prefixed name still has to fail, which is what the
    second half asserts.
    """
    env_file = tmp_path / "dotenv"
    env_file.write_text(
        "LANGFUSE_ENCRYPTION_KEY=" + "0" * 64 + "\n"
        "LANGFUSE_INIT_USER_PASSWORD=cascade-local-password\n"
        "CASCADE_DB_SIM_PASSWORD=from-dot-env\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CASCADE_ENV_FILE", str(env_file))

    settings = Settings()
    assert settings.db_sim_password is not None
    assert not hasattr(settings, "langfuse_encryption_key")


def test_a_missing_dot_env_is_not_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A fresh clone has no `.env`; config must still load from YAML."""
    monkeypatch.setenv("CASCADE_ENV_FILE", str(tmp_path / "does-not-exist"))
    assert Settings().study.name == "cascade-v1"


def test_secrets_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    settings = Settings()
    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-not-a-real-key"


def test_secrets_do_not_leak_through_repr_or_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    """``dev config`` dumps the whole object; a plaintext key there is a leak."""
    monkeypatch.setenv("CASCADE_ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    settings = Settings()
    assert "sk-ant-not-a-real-key" not in repr(settings)
    assert "sk-ant-not-a-real-key" not in settings.model_dump_json()


def test_database_url_carries_the_role_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CASCADE_DB_SIM_PASSWORD", "simsecret")
    url = Settings().database_url("sim")
    assert url == "postgresql://cascade_sim:simsecret@localhost:5433/cascade"


def test_database_port_is_not_the_default_5432() -> None:
    """5433 so a study cannot silently write into a developer's own instance."""
    assert Settings().database.port == 5433


# ---------------------------------------------------------------------------
# Lookups that must fail loudly
# ---------------------------------------------------------------------------


def test_unpriced_model_raises() -> None:
    with pytest.raises(KeyError, match="no pricing entry"):
        Settings().price_for("gpt-not-in-this-study")


def test_unbudgeted_phase_raises() -> None:
    with pytest.raises(KeyError, match="no budget ceiling"):
        Settings().phase_ceiling("phase-that-does-not-exist")


def test_configured_phase_ceilings_match_the_spec() -> None:
    settings = Settings()
    assert settings.phase_ceiling("compile") == Decimal("40.0")
    assert settings.phase_ceiling("simulate") == Decimal("240.0")
    assert settings.phase_ceiling("baseline") == Decimal("30.0")


# ---------------------------------------------------------------------------
# Paths and caching
# ---------------------------------------------------------------------------


def test_relative_paths_resolve_against_the_repo_not_the_cwd() -> None:
    """A CLI invoked from a subdirectory must read the same config."""
    settings = Settings()
    assert settings.cache_path() == repo_root() / ".cache" / "llm"
    assert settings.checkpoint_path() == repo_root() / ".checkpoints"


def test_absolute_paths_are_left_alone(tmp_path: Path) -> None:
    base = Settings()
    settings = base.model_copy(
        update={"llm": base.llm.model_copy(update={"cache_dir": str(tmp_path)})}
    )
    assert settings.cache_path() == tmp_path


def test_load_settings_is_memoised() -> None:
    """Config is immutable within a run by construction, not by convention."""
    assert load_settings() is load_settings()


def test_overlay_deep_merges_rather_than_replacing() -> None:
    """An overlay setting one flag must not blank the rest of its section."""
    base = load_settings()
    cell = load_settings("C09")
    assert cell.flags.causal_decomposition is False
    assert cell.flags.grounding == "chronofence"
    # Untouched sections survive intact.
    assert cell.kernel == base.kernel
    assert cell.models == base.models
    assert cell.pricing == base.pricing


def test_missing_overlay_raises() -> None:
    with pytest.raises(FileNotFoundError, match="ablation overlay not found"):
        load_settings("C99")
