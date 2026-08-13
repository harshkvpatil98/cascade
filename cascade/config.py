"""Typed configuration for the whole system.

This module preserves the invariant that **no other module reads the process
environment**. Every tunable reaches the rest of the codebase as a validated
Pydantic model loaded here; a bare ``os.environ`` or ``os.getenv`` anywhere
else in ``cascade/`` is a bug and is caught by
``tests/unit/test_no_bare_environ.py``.

Precedence, highest first: explicit constructor arguments, then environment
variables (``CASCADE_`` prefix, ``__`` nesting delimiter), then the repo's
``.env``, then ``configs/base.yaml``. Secrets are environment-only and never
appear in YAML.

``.env`` is read here rather than only by the Makefile so that ``cascade
doctor`` and ``cascade db status`` work from a plain shell. Its location comes
from ``CASCADE_ENV_FILE``, which the test suite points at a nonexistent path:
a test whose result depended on a developer's machine-local ``.env`` would be
exactly the kind of environment coupling the determinism rules exist to
remove.
"""

from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

__all__ = [
    "LLMMode",
    "Settings",
    "load_settings",
    "repo_root",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

# CASCADE_-prefixed variables that steer loading rather than populating a
# field, and so are legitimately absent from the model.
_NON_FIELD_ENV_VARS = frozenset({"CASCADE_CONFIG", "CASCADE_ENV_FILE"})

LLMMode = Literal["record", "replay", "live"]
Grounding = Literal["chronofence", "parametric_only"]
CacheTTL = Literal["5m", "1h"]


def repo_root() -> Path:
    """Return the repository root.

    Preserves the invariant that paths in config are resolved relative to the
    repository, not the caller's working directory, so a CLI invoked from a
    subdirectory reads the same config as one invoked from the root.
    """
    return REPO_ROOT


def _to_decimal(value: Any) -> Decimal:
    """Coerce a YAML scalar to ``Decimal`` without inheriting binary-float error.

    ``Decimal(0.1)`` is 0.1000000000000000055511151231257827..., which would
    make the cost meter's 6-decimal-place assertion fail for reasons that have
    nothing to do with the meter. Routing through ``str`` preserves the
    decimal literal the author wrote.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


Money = Annotated[Decimal, BeforeValidator(_to_decimal)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class StudyConfig(_Model):
    name: str
    salt: str
    manifest_sha256: str | None = None


class ModelsConfig(_Model):
    agent: str
    compiler: str
    embedding: str
    temperature: float
    compiler_temperature: float
    use_batch_api: bool
    agent_max_tokens: int
    compiler_max_tokens: int


class PricingEntry(_Model):
    input_per_mtok: Money
    output_per_mtok: Money


class PricingMultipliers(_Model):
    batch: Money
    cache_read: Money
    cache_write_5m: Money
    cache_write_1h: Money


class PromptCacheConfig(_Model):
    enabled: bool
    ttl: CacheTTL
    min_prefix_tokens: int
    enforce_min_prefix: bool


class ActivationConfig(_Model):
    salience_threshold: float
    forced_interval: int
    max_active_per_step: int


class MemoryConfig(_Model):
    recent_observations: int
    summary_interval: int


class KernelConfig(_Model):
    steps: int
    max_step_delta: float
    contest_gamma: float
    activation: ActivationConfig
    memory: MemoryConfig


class Channel(_Model):
    noise_sigma: float
    lag: int
    quantized: bool


class ApertureConfig(_Model):
    hop1: Channel
    hop2: Channel
    public: Channel


class RetrievalConfig(_Model):
    k_agent: int
    k_compiler: int
    ivfflat_probes: int
    target_p95_ms: float
    partition_granularity: Literal["month", "quarter"]


class EnsembleConfig(_Model):
    replicates: int
    ablation_replicates: int
    bootstrap_b: int
    sigma_multimodal_threshold: float


class BudgetConfig(_Model):
    phase_ceiling_usd: dict[str, Money]
    abort_on_breach: bool


class FlagsConfig(_Model):
    causal_decomposition: bool
    information_asymmetry: bool
    grounding: Grounding


class LLMConfig(_Model):
    mode: LLMMode
    cache_dir: str
    prompt_rev: str
    max_retries: int
    timeout_s: float


class DatabaseConfig(_Model):
    host: str
    port: int
    name: str
    admin_user: str
    sim_user: str
    eval_user: str


class LangfuseConfig(_Model):
    enabled: bool
    host: str


class LedgerConfig(_Model):
    """Scenario-registry constraints (spec §3.1).

    These are *inclusion constraints*, not results. The measurement contract
    forbids writing a target metric into a report path; a constraint that
    defines which questions may enter the set is the opposite -- it has to be
    explicit, versioned and asserted, or the set is not reproducible.
    """

    target_scenarios: int
    yes_rate_min: float
    yes_rate_max: float
    max_domain_share: float
    min_volume: float
    cutoff_fraction: float
    source_cache_dir: str
    curated_dir: str
    polymarket_pages: int
    polymarket_page_size: int
    manifold_pages: int
    manifold_page_size: int
    metaculus_pages: int
    metaculus_page_size: int


class PathsConfig(_Model):
    checkpoints: str
    reports: str
    graphs: str


class _YamlSource(PydanticBaseSettingsSource):
    """Feed ``configs/base.yaml`` into the settings chain as the lowest priority."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        # Whole-document sources implement __call__ instead; this is required
        # by the ABC but never consulted for our usage.
        raise NotImplementedError

    def __call__(self) -> dict[str, Any]:
        if not self._path.is_file():
            raise FileNotFoundError(f"config file not found: {self._path}")
        with self._path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"config file {self._path} must contain a YAML mapping")
        return loaded


class Settings(BaseSettings):
    """Root configuration object. Constructed once, passed explicitly thereafter."""

    model_config = SettingsConfigDict(
        env_prefix="CASCADE_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    study: StudyConfig
    models: ModelsConfig
    pricing: dict[str, PricingEntry]
    pricing_multipliers: PricingMultipliers
    prompt_cache: PromptCacheConfig
    kernel: KernelConfig
    aperture: ApertureConfig
    retrieval: RetrievalConfig
    ensemble: EnsembleConfig
    budget: BudgetConfig
    flags: FlagsConfig
    llm: LLMConfig
    ledger: LedgerConfig
    database: DatabaseConfig
    langfuse: LangfuseConfig
    paths: PathsConfig

    # --- Secrets: environment only, never YAML ------------------------------
    anthropic_api_key: SecretStr | None = Field(default=None)
    # Metaculus rejects unauthenticated API requests (see
    # cascade/ledger/sources/metaculus.py). Absent, the loader reports the
    # source as unavailable rather than contributing silently.
    metaculus_token: SecretStr | None = Field(default=None)
    db_admin_password: SecretStr | None = Field(default=None)
    db_sim_password: SecretStr | None = Field(default=None)
    db_eval_password: SecretStr | None = Field(default=None)
    langfuse_public_key: SecretStr | None = Field(default=None)
    langfuse_secret_key: SecretStr | None = Field(default=None)

    @model_validator(mode="after")
    def _reject_unrecognised_overrides(self) -> Settings:
        """Fail on a ``CASCADE_``-prefixed variable that binds to nothing.

        ``extra="forbid"`` catches a typo *inside* a known section
        (``CASCADE_KERNEL__STEPZ``), but pydantic-settings harvests only the
        variables that match a field, so a typo in the **root** segment is
        silently discarded: ``CASCADE_ENSEMBLE_REPLICATES`` (single underscore
        where the nesting delimiter belongs) sets nothing, reports nothing, and
        leaves the study running on the default while the operator believes it
        is overridden. That is precisely the silent-default failure the
        no-bare-environ rule exists to prevent, so it is an error here.

        Only the root segment is checked, because sections keyed by a dict --
        ``CASCADE_BUDGET__PHASE_CEILING_USD__SIMULATE`` -- have legitimately
        dynamic leaves.
        """
        known = {name.upper() for name in type(self).model_fields}
        offenders = sorted(
            name
            for name in os.environ
            if name.startswith("CASCADE_")
            and name not in _NON_FIELD_ENV_VARS
            and name[len("CASCADE_") :].split("__", 1)[0] not in known
        )
        if offenders:
            raise ValueError(
                f"unrecognised CASCADE_ environment variable(s): {', '.join(offenders)}. "
                f"Known top-level settings: {', '.join(sorted(known))}. "
                "Nested keys use a double underscore, e.g. "
                "CASCADE_ENSEMBLE__REPLICATES. A variable that binds to nothing "
                "would leave the study on its default value without saying so."
            )
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Both paths are resolved at call time, not at class-creation time, so
        # a test can redirect either one without re-importing this module.
        yaml_path = Path(os.environ.get("CASCADE_CONFIG", REPO_ROOT / "configs" / "base.yaml"))
        env_path = Path(os.environ.get("CASCADE_ENV_FILE", REPO_ROOT / ".env"))
        del dotenv_settings  # replaced by the explicitly-located source below
        return (
            init_settings,
            env_settings,
            DotEnvSettingsSource(
                settings_cls,
                env_file=env_path,
                env_file_encoding="utf-8",
                env_prefix="CASCADE_",
                env_nested_delimiter="__",
                case_sensitive=False,
                # `.env` legitimately carries two namespaces: CASCADE_* settings
                # and LANGFUSE_*/compose-only infrastructure values. Only the
                # first belongs to this model. "only_existing" reads it exactly
                # as the environment source does -- prefix applied, unknown keys
                # left alone -- so a compose variable is not mistaken for a
                # typo'd setting. `extra="forbid"` still rejects a real typo in
                # a CASCADE_-prefixed name.
                dotenv_filtering="only_existing",
            ),
            _YamlSource(settings_cls, yaml_path),
        )

    # --- Derived accessors --------------------------------------------------

    def cache_path(self) -> Path:
        return self._resolve(self.llm.cache_dir)

    def checkpoint_path(self) -> Path:
        return self._resolve(self.paths.checkpoints)

    def report_path(self) -> Path:
        return self._resolve(self.paths.reports)

    def graph_path(self) -> Path:
        return self._resolve(self.paths.graphs)

    def source_cache_path(self) -> Path:
        return self._resolve(self.ledger.source_cache_dir)

    def curated_path(self) -> Path:
        return self._resolve(self.ledger.curated_dir)

    def _resolve(self, raw: str) -> Path:
        candidate = Path(raw)
        return candidate if candidate.is_absolute() else REPO_ROOT / candidate

    def price_for(self, model: str) -> PricingEntry:
        """Return the price table entry for ``model``.

        Raises rather than defaulting: an unpriced model silently costing $0
        would corrupt the cost ledger, which blocks the report at M8.
        """
        try:
            return self.pricing[model]
        except KeyError:
            known = ", ".join(sorted(self.pricing))
            raise KeyError(
                f"no pricing entry for model {model!r}; known models: {known}. "
                "Add it to configs/base.yaml -- an unpriced model breaks the cost ledger."
            ) from None

    def phase_ceiling(self, phase: str) -> Decimal:
        try:
            return self.budget.phase_ceiling_usd[phase]
        except KeyError:
            known = ", ".join(sorted(self.budget.phase_ceiling_usd))
            raise KeyError(
                f"no budget ceiling configured for phase {phase!r}; known phases: {known}"
            ) from None

    def database_url(self, role: Literal["admin", "sim", "eval"]) -> str:
        """Build a libpq URL for one of the three roles.

        Kept here rather than in the DB layer so credentials never need an
        environment read outside this module.
        """
        user, password = {
            "admin": (self.database.admin_user, self.db_admin_password),
            "sim": (self.database.sim_user, self.db_sim_password),
            "eval": (self.database.eval_user, self.db_eval_password),
        }[role]
        secret = password.get_secret_value() if password else ""
        return (
            f"postgresql://{user}:{secret}@{self.database.host}:"
            f"{self.database.port}/{self.database.name}"
        )


@lru_cache(maxsize=8)
def _cached_settings(overlay: str | None) -> Settings:
    # Zero-argument construction is correct: the YAML settings source supplies
    # the required fields. The pydantic mypy plugin understands this; without
    # it the call reads as fifteen missing arguments.
    base = Settings()
    if overlay is None:
        return base
    overlay_path = REPO_ROOT / "configs" / "ablations" / f"{overlay}.yaml"
    if not overlay_path.is_file():
        raise FileNotFoundError(f"ablation overlay not found: {overlay_path}")
    with overlay_path.open("r", encoding="utf-8") as handle:
        patch = yaml.safe_load(handle) or {}
    merged = _deep_merge(base.model_dump(mode="python"), patch)
    return Settings(**merged)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Recursively overlay ``patch`` onto ``base`` without mutating either.

    Iterates in sorted key order (invariant 7). Nothing here depends on the
    order today, but an unsorted walk over a mapping is the nondeterminism
    vector the invariant exists to remove, and the static check in
    ``tests/unit/test_invariants.py`` does not special-case config.
    """
    out = dict(base)
    for key, value in sorted(patch.items()):
        current = out.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            out[key] = _deep_merge(current, value)
        else:
            out[key] = value
    return out


def load_settings(overlay: str | None = None) -> Settings:
    """Load configuration, optionally overlaid with an ablation cell.

    ``overlay`` names a file in ``configs/ablations/`` without its extension.
    Results are cached so repeated calls in one process return an identical
    object -- config is immutable within a run by construction.
    """
    return _cached_settings(overlay)
