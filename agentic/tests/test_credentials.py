"""Offline credential regression tests: no model calls, Docker runs or real keys."""

import copy
import io
import json
import os
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.request import HTTPHandler, build_opener
from urllib.response import addinfourl

import yaml

from core import claude_wrapper
from core.credentials import DEFAULT_API_KEY_ENV, recordable_config, resolve_api_key, solver_key_env
from core.evaluate import _run_process_mode, load_config
from core.llm import LLMClient, LLMConfig
from core.meta_harness import Run
from scripts import create_run


class CredentialTests(unittest.TestCase):
    value = "unit-test-credential-value"

    def config(self):
        return {
            "solver": {"model": "test-model", "api_base": "http://localhost:8000/v1",
                       "api_key_env": DEFAULT_API_KEY_ENV},
            "domain": "hotpotqa", "domain_config": {},
            "proposer": {"model": "test-proposer"},
        }

    def test_saved_config_retains_reference_without_reading_environment(self):
        original = self.config()
        before = copy.deepcopy(original)
        with patch.dict(os.environ, {DEFAULT_API_KEY_ENV: self.value}, clear=True):
            for saved in (recordable_config(original), create_run.to_container(original)):
                self.assertEqual(saved["solver"]["api_key_env"], DEFAULT_API_KEY_ENV)
                self.assertNotIn("api_key", saved["solver"])
                self.assertNotIn(self.value, yaml.safe_dump(saved))
        self.assertEqual(original, before)

    def test_legacy_placeholder_normalized_and_inline_secret_rejected_without_echo(self):
        for placeholder in (None, "", "local"):
            cfg = self.config()
            cfg["solver"]["api_key"] = placeholder
            self.assertNotIn("api_key", recordable_config(cfg)["solver"])
        cfg["solver"]["api_key"] = self.value
        for normalize in (recordable_config, create_run.to_container):
            with self.assertRaises(ValueError) as caught:
                normalize(cfg)
            self.assertNotIn(self.value, str(caught.exception))

    def test_default_and_custom_environment_resolution(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_api_key(DEFAULT_API_KEY_ENV), "local")
            with self.assertRaises(ValueError):
                resolve_api_key("CUSTOM_SOLVER_KEY")
        with patch.dict(os.environ, {DEFAULT_API_KEY_ENV: "", "CUSTOM_SOLVER_KEY": self.value}, clear=True):
            self.assertEqual(resolve_api_key(DEFAULT_API_KEY_ENV), "local")
            self.assertEqual(resolve_api_key("CUSTOM_SOLVER_KEY"), self.value)

    def test_invalid_environment_references_rejected(self):
        for name in ("", None, 123, "KEY=value", "--privileged", "A\nB"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                solver_key_env({"api_key_env": name})

    def test_key_is_resolved_only_when_constructing_sdk_client(self):
        cfg = LLMConfig.from_dict({"api_key_env": "CUSTOM_SOLVER_KEY"})
        with patch.dict(os.environ, {"CUSTOM_SOLVER_KEY": self.value}, clear=True), patch("core.llm.OpenAI") as sdk:
            client = LLMClient(cfg)
        self.assertEqual(sdk.call_args.kwargs["api_key"], self.value)
        self.assertIs(client.cfg, cfg)
        self.assertNotIn(self.value, repr(cfg))
        self.assertNotIn(self.value, json.dumps(asdict(cfg)))

    def test_preflight_rejects_redirect_without_forwarding_credential(self):
        # Run only the health-check Python block with an in-memory HTTP transport.
        # No shell commands, network connections, Docker or model calls occur.
        script = (Path(__file__).resolve().parents[1] / "scripts/preflight.sh").read_text()
        block = script.split("<<'PY'", 1)[1].split("\n", 1)[1].split("\nPY\n", 1)[0]
        requests = []

        class RedirectTransport(HTTPHandler):
            def http_open(self, request):
                requests.append(request)
                response = addinfourl(io.BytesIO(b""), {"Location": "http://other-origin.invalid/models"},
                                     request.full_url, 302)
                response.msg = "Found"
                return response

        def local_opener(*handlers):
            return build_opener(*handlers, RedirectTransport())

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "config.yaml"
            source.write_text(yaml.safe_dump(self.config()))
            output = io.StringIO()
            with patch("sys.argv", ["-", str(Path(__file__).resolve().parents[1]), str(source)]), \
                    patch.dict(os.environ, {DEFAULT_API_KEY_ENV: self.value}, clear=True), \
                    patch("urllib.request.build_opener", side_effect=local_opener), redirect_stdout(output):
                with self.assertRaises(SystemExit) as caught:
                    exec(compile(block, "preflight-health-check", "exec"), {})
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0].get_header("Authorization"), "Bearer " + self.value)
            self.assertEqual(output.getvalue(), "")

    def test_create_run_writes_host_and_container_configs_without_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("configs", "core", "domains/hotpotqa"):
                (root / name).mkdir(parents=True)
            (root / "domains/hotpotqa/SKILL.md").write_text("Test fixture only.\n")
            (root / "configs/common.yaml").write_text(yaml.safe_dump(self.config()))
            run_config = root / "configs/test.yaml"
            run_config.write_text(yaml.safe_dump({"run_id": "test-run", "baselines": []}))
            for split in ("train", "val", "selection", "test"):
                path = root / "data" / ("splits" if split in ("train", "val") else "splits_test") / "hotpotqa" / f"{split}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
            output = io.StringIO()
            with patch.object(create_run, "REPO", root), patch("sys.argv", ["create_run.py", str(run_config)]), \
                    patch.dict(os.environ, {DEFAULT_API_KEY_ENV: self.value}, clear=True), \
                    patch.object(create_run.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")), \
                    redirect_stdout(output):
                self.assertEqual(create_run.main(), 0)
            for name in ("resolved_config.yaml", "checkout/config.container.yaml"):
                text = (root / "runs/test-run" / name).read_text()
                self.assertNotIn(self.value, text)
                self.assertNotIn("api_key", yaml.safe_load(text)["solver"])
            self.assertNotIn(self.value, output.getvalue())

    def test_evaluator_load_and_process_snapshot_keep_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = self.config()
            cfg["paths"] = {"checkout": str(root)}
            source = root / "config.yaml"
            source.write_text(yaml.safe_dump(cfg))
            with patch.dict(os.environ, {DEFAULT_API_KEY_ENV: self.value}, clear=True), patch("core.evaluate.subprocess.Popen") as popen:
                loaded = load_config(source)
                _run_process_mode(loaded, root / "unused.py", "val", root, None, [], 1, lambda _: None)
                popen.assert_not_called()
            saved = (root / "effective_config.yaml").read_text()
            self.assertNotIn(self.value, saved)
            self.assertNotIn("_config_path", yaml.safe_load(saved))
            self.assertEqual(yaml.safe_load(saved)["solver"]["api_key_env"], DEFAULT_API_KEY_ENV)
            cfg["solver"]["api_key"] = self.value
            source.write_text(yaml.safe_dump(cfg))
            with self.assertRaises(ValueError):
                load_config(source)
            with self.assertRaises(ValueError):
                _run_process_mode(cfg, root / "unused.py", "val", root, None, [], 1, lambda _: None)

    def test_docker_command_inherits_name_without_key_value(self):
        with tempfile.TemporaryDirectory() as directory:
            container = {"image": "test-image", "workspace_root": directory, "home_dir": directory,
                         "oauth_token": True, "pass_env": ["CUSTOM_SOLVER_KEY"]}
            with patch.dict(os.environ, {"CUSTOM_SOLVER_KEY": self.value}, clear=True), \
                    patch.object(claude_wrapper, "_own_writable_paths"):
                command = claude_wrapper.wrap_in_container(["test-command"], container)
            self.assertIn("CUSTOM_SOLVER_KEY", command)
            self.assertEqual(command[command.index("CUSTOM_SOLVER_KEY") - 1], "-e")
            self.assertNotIn(self.value, " ".join(command))
            container["pass_env"] = ["CUSTOM_SOLVER_KEY=" + self.value]
            with self.assertRaises(ValueError), patch.object(claude_wrapper, "_own_writable_paths"):
                claude_wrapper.wrap_in_container(["test-command"], container)

    def test_proposer_passes_custom_reference_only_when_solver_access_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = object.__new__(Run)
            run.cfg = self.config()
            run.cfg.update(run_id="test-run", paths={"proposer_home": str(root / "home")})
            run.cfg["solver"]["api_key_env"] = "CUSTOM_SOLVER_KEY"
            run.cfg["proposer"]["image"] = "test-image"
            run.checkout = run.logs = root
            run.pending_path = root / "pending.json"
            result = SimpleNamespace(exit_code=0, show=lambda: None)
            for enabled in (True, False):
                run.cfg["proposer"]["allow_solver_access"] = enabled
                with patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "unit-test-oauth", "CUSTOM_SOLVER_KEY": self.value}, clear=True), \
                        patch.object(run, "task_prompt", return_value="test prompt"), \
                        patch.object(claude_wrapper, "run", return_value=result) as launch:
                    run.propose(1)
                container = launch.call_args.kwargs["container"]
                self.assertEqual(container["pass_env"], ["CUSTOM_SOLVER_KEY"] if enabled else [])
                self.assertNotIn(self.value, json.dumps(container))


if __name__ == "__main__":
    unittest.main()
