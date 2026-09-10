"""Immutable evidence from actual stage terminals, independent of output length."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class StageTermination:
    stage_id: int
    request_id: str
    finish_reason: str
    stop_reason: int | str | None = None


@dataclass(frozen=True)
class TerminationEvidence:
    # Do not inherit the OpenAI response's extra='allow' for derived fields.
    __pydantic_config__ = {"extra": "forbid"}

    required_stage_ids: tuple[int, ...]
    stages: tuple[StageTermination, ...]
    complete: bool
    schema_version: int = field(default=1, init=False)
    truncated: bool | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        observed = tuple(sorted(stage.stage_id for stage in self.stages))
        complete = self.complete and bool(self.required_stage_ids) and observed == self.required_stage_ids
        object.__setattr__(self, "complete", complete)
        if any(stage.finish_reason in ("length", "max_tokens") for stage in self.stages):
            object.__setattr__(self, "truncated", True)
        elif complete and all(stage.finish_reason == "stop" for stage in self.stages):
            object.__setattr__(self, "truncated", False)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def finish_reason(self) -> str | None:
        if self.truncated is True:
            return "length"
        return "stop" if self.truncated is False else None
