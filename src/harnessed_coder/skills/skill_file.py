"""Read and validate ``SKILL.md`` files against the Agent Skills format."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from .types import SkillDefinition, SkillMetadata, SkillScope


_NAME_PATTERN = re.compile(r"(?!.*--)[a-z0-9]+(?:-[a-z0-9]+)*")
_STANDARD_FIELDS = {
    "name",
    "description",
    "license",
    "compatibility",
    "metadata",
    "allowed-tools",
}
_MAX_FRONTMATTER_CHARS = 16_384
_MAX_SKILL_BYTES = 256 * 1024


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML 1.2-like loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)
# PyYAML follows YAML 1.1 and treats names such as ``yes`` and ``on`` as
# booleans. Agent Skills permits those lowercase names, so use YAML 1.2 boolean
# spelling while retaining SafeLoader's protection against arbitrary objects.
_UniqueKeyLoader.yaml_implicit_resolvers = {
    key: [
        resolver
        for resolver in resolvers
        if resolver[0] != "tag:yaml.org,2002:bool"
    ]
    for key, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def parse_skill_file(
    skill_file: Path,
    *,
    scope: SkillScope,
    directory: Path,
) -> SkillDefinition:
    """Read and validate one complete ``SKILL.md`` file."""
    frontmatter, body = _read_skill_parts(skill_file)
    fields = _parse_frontmatter(frontmatter, skill_file)
    metadata = _metadata_from_fields(
        fields,
        skill_file=skill_file,
        scope=scope,
        directory=directory,
    )
    return SkillDefinition(metadata=metadata, instructions=body.strip())


def _read_skill_parts(skill_file: Path) -> tuple[str, str]:
    try:
        size = skill_file.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot inspect Skill file {skill_file}: {exc}") from exc
    if size > _MAX_SKILL_BYTES:
        raise ValueError(
            f"Skill file exceeds {_MAX_SKILL_BYTES} bytes: {skill_file}"
        )

    try:
        with skill_file.open("r", encoding="utf-8-sig", newline=None) as stream:
            text = stream.read()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Skill file must be UTF-8: {skill_file}") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read Skill file {skill_file}: {exc}") from exc

    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise ValueError(
            f"Skill file must begin with YAML frontmatter: {skill_file}"
        )

    frontmatter: list[str] = []
    frontmatter_chars = 0
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            return "".join(frontmatter), "".join(lines[index + 1:])
        frontmatter_chars += len(line)
        if frontmatter_chars > _MAX_FRONTMATTER_CHARS:
            raise ValueError(
                f"Skill frontmatter exceeds {_MAX_FRONTMATTER_CHARS} "
                f"characters: {skill_file}"
            )
        frontmatter.append(line)

    raise ValueError(f"Skill frontmatter is not closed with ---: {skill_file}")


def _parse_frontmatter(text: str, skill_file: Path) -> dict[str, object]:
    try:
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter in {skill_file}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Skill frontmatter must be a YAML mapping: {skill_file}"
        )
    fields: dict[str, object] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ValueError(
                f"Skill frontmatter keys must be strings: {skill_file}"
            )
        if key not in _STANDARD_FIELDS:
            raise ValueError(
                f"Unsupported Skill frontmatter field {key!r}; "
                f"put client extensions under metadata: {skill_file}"
            )
        fields[key] = value
    return fields


def _metadata_from_fields(
    fields: dict[str, object],
    *,
    skill_file: Path,
    scope: SkillScope,
    directory: Path,
) -> SkillMetadata:
    name = fields.get("name")
    description = fields.get("description")
    license_value = fields.get("license")
    compatibility = fields.get("compatibility")
    metadata_value = fields.get("metadata")
    allowed_tools = fields.get("allowed-tools")

    if (
        not isinstance(name, str)
        or len(name) > 64
        or not _NAME_PATTERN.fullmatch(name)
    ):
        raise ValueError(
            "Skill name must be 1-64 lowercase letters, digits, or hyphens, "
            f"without leading, trailing, or consecutive hyphens: {skill_file}"
        )
    if name != directory.name:
        raise ValueError(
            f"Skill name {name!r} must match directory {directory.name!r}: "
            f"{skill_file}"
        )
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"Skill description must not be empty: {skill_file}")
    description = description.strip()
    if len(description) > 1024:
        raise ValueError(f"Skill description exceeds 1024 characters: {skill_file}")

    license_text = _optional_non_empty_string(
        license_value,
        field="license",
        skill_file=skill_file,
    )
    compatibility_text = _optional_non_empty_string(
        compatibility,
        field="compatibility",
        skill_file=skill_file,
    )
    if compatibility_text is not None and len(compatibility_text) > 500:
        raise ValueError(
            f"Skill compatibility exceeds 500 characters: {skill_file}"
        )
    allowed_tools_text = _optional_non_empty_string(
        allowed_tools,
        field="allowed-tools",
        skill_file=skill_file,
    )
    metadata_items = _validate_metadata(metadata_value, skill_file)

    return SkillMetadata(
        name=name,
        description=description,
        scope=scope,
        directory=directory,
        skill_file=skill_file,
        license=license_text,
        compatibility=compatibility_text,
        metadata=metadata_items,
        allowed_tools=allowed_tools_text,
    )


def _optional_non_empty_string(
    value: object,
    *,
    field: str,
    skill_file: Path,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Skill {field} must be a non-empty string: {skill_file}"
        )
    return value.strip()


def _validate_metadata(
    value: object,
    skill_file: Path,
) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError(
            f"Skill metadata must be a string-to-string mapping: {skill_file}"
        )
    items: list[tuple[str, str]] = []
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not isinstance(item, str)
        ):
            raise ValueError(
                f"Skill metadata must be a string-to-string mapping: {skill_file}"
            )
        items.append((key, item))
    return tuple(sorted(items))
