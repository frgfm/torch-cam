# Copyright (C) 2024-2025, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.


# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pyyaml>=6.0",
# ]
# ///

import logging
import sys
import tomllib
from pathlib import Path

import yaml

PRECOMMIT_CONFIG = ".pre-commit-config.yaml"
PYPROJECTS = ["./pyproject.toml"]
MINIMUM_UV_VERSION = "0.13.4"
EXPECTED_UV_BUILD_REQUIREMENT = f"uv_build>={MINIMUM_UV_VERSION},<1.0.0"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler(sys.stdout)
log_formatter = logging.Formatter("%(levelname)s:     %(message)s")
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)


def parse_dep_str(dep_str: str) -> dict[str, str]:
    # No version specified
    if all(dep_str.find(char) == -1 for char in ("<", ">", "=")):
        version_idx = len(dep_str)
    else:
        version_idx = min(idx for idx in (dep_str.find(char) for char in ("<", ">", "=")) if idx != -1)
    pkg_idx = dep_str.find("[")
    extra_idx = dep_str.find("]")
    return {
        "pkg": dep_str[: pkg_idx if pkg_idx != -1 else version_idx],
        "version": dep_str[version_idx:],
        "extras": [] if extra_idx == -1 else dep_str[pkg_idx + 1 : extra_idx].split(","),
    }


def parse_workflow_uv_versions(
    minimum_uv_version: tuple[int, ...],
) -> tuple[list[dict[str, str]], list[str]]:
    dependencies = []
    troubles = []
    for workflow_file in Path(".github/workflows").glob("*.yml"):
        with workflow_file.open("r") as f:
            workflow = yaml.safe_load(f)
        uv_steps = [
            step
            for job in (workflow.get("jobs") or {}).values()
            for step in job.get("steps", [])
            if (step.get("uses") or "").startswith("astral-sh/setup-uv@")
        ]
        workflow_env = workflow.get("env") or {}
        uv_version = workflow_env.get("UV_VERSION")
        if uv_steps and not uv_version:
            troubles.append(f"{workflow_file}: setup-uv requires a top-level UV_VERSION")
        if uv_version is not None:
            uv_version = str(uv_version).lstrip("v")
            dependencies.append({"file": str(workflow_file), "version": f"=={uv_version}"})
            try:
                installed_uv_version = tuple(map(int, uv_version.split(".")))
            except ValueError:
                troubles.append(f"{workflow_file}: invalid UV_VERSION {uv_version!r}")
            else:
                if installed_uv_version < minimum_uv_version:
                    troubles.append(f"{workflow_file}: UV_VERSION must be at least {MINIMUM_UV_VERSION}")
        if any(
            (step.get("with") or {}).get("version") != "${{ env.UV_VERSION }}"
            for step in uv_steps
        ):
            troubles.append(f"{workflow_file}: setup-uv version must use ${{{{ env.UV_VERSION }}}}")
    return dependencies, troubles


def main():
    # Retrieve & parse all deps files
    deps_dict = {
        "uv": [],
        "ty": [],
        "pre-commit": [],
    }
    troubles = []
    minimum_uv_version = tuple(map(int, MINIMUM_UV_VERSION.split(".")))
    expected_uv_build_version = EXPECTED_UV_BUILD_REQUIREMENT.removeprefix("uv_build")
    # Parse precommit
    with Path(PRECOMMIT_CONFIG).open("r", encoding="utf-8") as f:
        precommit = yaml.safe_load(f)
    for repo in precommit["repos"]:
        if repo["repo"] == "https://github.com/astral-sh/uv-pre-commit":
            deps_dict["uv"].append({"file": PRECOMMIT_CONFIG, "version": f"=={repo['rev'].lstrip('v')}"})
    # Parse pyproject.toml
    for pyproject_path in PYPROJECTS:
        with Path(pyproject_path).open("rb") as f:
            pyproject = tomllib.load(f)

        # Check the build backend version used by this project.
        build_deps = [parse_dep_str(dep) for dep in pyproject["build-system"]["requires"]]
        uv_build_deps = [dep for dep in build_deps if dep["pkg"] == "uv_build"]
        if len(uv_build_deps) != 1 or uv_build_deps[0]["version"] != expected_uv_build_version:
            troubles.append(f"{pyproject_path}: build-system requires {EXPECTED_UV_BUILD_REQUIREMENT}")

        # Parse dependencies
        core_deps = [parse_dep_str(dep) for dep in pyproject["project"]["dependencies"]]
        core_deps = {dep["pkg"]: dep for dep in core_deps}
        for dep in deps_dict:  # noqa: PLC0206
            if dep in core_deps:
                deps_dict[dep].append({"file": pyproject_path, "version": core_deps[dep]["version"]})

        # Parse optional dependencies
        quality_deps = [parse_dep_str(dep) for dep in pyproject["project"]["optional-dependencies"]["quality"]]
        quality_deps = {dep["pkg"]: dep for dep in quality_deps}
        for dep in deps_dict:  # noqa: PLC0206
            if dep in quality_deps:
                deps_dict[dep].append({"file": pyproject_path, "version": quality_deps[dep]["version"]})

    # Parse github/workflows/...
    workflow_uv_dependencies, workflow_troubles = parse_workflow_uv_versions(minimum_uv_version)
    deps_dict["uv"].extend(workflow_uv_dependencies)
    troubles.extend(workflow_troubles)

    # Assert all deps are in sync
    for dep, versions in deps_dict.items():
        versions_ = {v["version"] for v in versions}
        if len(versions_) > 1:
            inv_dict = {v: set() for v in versions_}
            for version in versions:
                inv_dict[version["version"]].add(version["file"])
            troubles.extend([
                f"\033[31m{dep}\033[0m:",
                "\n".join(f"- '{v}': {', '.join(files)}" for v, files in inv_dict.items()),
            ])

    if len(troubles) > 0:
        raise AssertionError("Some dependencies are out of sync:\n\n" + "\n".join(troubles))
    logger.info("\033[32mAll dependencies are in sync!\033[0m")


if __name__ == "__main__":
    main()
