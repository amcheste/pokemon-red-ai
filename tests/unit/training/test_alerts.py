"""
Unit tests for ``pokemon_red_ai.training.alerts``.

Covers:
* The :class:`Alert` dataclass
* All four built-in channels (Log, Desktop, Slack, Email) — using mocks
  so no real notifications, HTTP calls, or SMTP sessions happen
* Config loading (YAML + JSON, error paths)
* :class:`TrainingAlertCallback` trigger logic, deduplication, plateau
  detection, checkpoint alerts, and crash notification
"""

from __future__ import annotations

import json
import logging
from typing import List
from unittest.mock import MagicMock, Mock, patch

import pytest

from pokemon_red_ai.training.alerts import (
    Alert,
    AlertChannel,
    DesktopChannel,
    EmailChannel,
    LogChannel,
    SlackChannel,
    TrainingAlertCallback,
    channels_from_config,
    load_alert_config,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class RecordingChannel(AlertChannel):
    """Channel that records every alert it receives — used by tests."""

    def __init__(self):
        self.alerts: List[Alert] = []

    def send(self, alert: Alert) -> bool:
        self.alerts.append(alert)
        return True


class FailingChannel(AlertChannel):
    """Channel whose send() always raises — exercises error swallowing."""

    def send(self, alert: Alert) -> bool:
        raise RuntimeError("simulated channel failure")


def _make_callback(
    *,
    channels=None,
    plateau_episodes=3,
    plateau_min_episodes=2,
    checkpoint_alert_freq=0,
    cooldown_seconds=0.0,
    clock=None,
    **kwargs,
) -> TrainingAlertCallback:
    """Build a callback wired to a model+env for use in tests."""
    cb = TrainingAlertCallback(
        channels=channels or [],
        plateau_episodes=plateau_episodes,
        plateau_min_episodes=plateau_min_episodes,
        checkpoint_alert_freq=checkpoint_alert_freq,
        cooldown_seconds=cooldown_seconds,
        clock=clock,
        verbose=0,
        **kwargs,
    )
    cb.model = Mock()
    cb.num_timesteps = 0
    cb.n_calls = 0
    cb.locals = {}
    cb.globals = {}
    return cb


def _step(cb: TrainingAlertCallback, *, info: dict, done: bool = False, step: int = 0) -> None:
    """Advance the callback by one fake env step with given info dict."""
    cb.num_timesteps = step
    cb.locals = {"dones": [done], "infos": [info]}
    cb._on_step()


# ──────────────────────────────────────────────────────────────────────
# Alert dataclass + LogChannel
# ──────────────────────────────────────────────────────────────────────


class TestAlert:
    def test_default_severity(self):
        a = Alert(title="hi", message="there")
        assert a.severity == "info"
        assert a.metadata == {}

    def test_custom_metadata(self):
        a = Alert(title="t", message="m", metadata={"k": 1})
        assert a.metadata == {"k": 1}


class TestLogChannel:
    def test_logs_at_correct_level(self, caplog):
        ch = LogChannel()
        with caplog.at_level(logging.INFO, logger="pokemon_red_ai.training.alerts"):
            assert ch.send(Alert("title", "msg", severity="info")) is True
        assert any("INFO" in rec.getMessage() and "title" in rec.getMessage() for rec in caplog.records)

    def test_warning_level(self, caplog):
        ch = LogChannel()
        with caplog.at_level(logging.WARNING, logger="pokemon_red_ai.training.alerts"):
            ch.send(Alert("warn", "wmsg", severity="warning"))
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_critical_maps_to_error(self, caplog):
        ch = LogChannel()
        with caplog.at_level(logging.ERROR, logger="pokemon_red_ai.training.alerts"):
            ch.send(Alert("crit", "msg", severity="critical"))
        assert any(rec.levelno == logging.ERROR for rec in caplog.records)


# ──────────────────────────────────────────────────────────────────────
# DesktopChannel — exercise both osascript and plyer paths via mocks
# ──────────────────────────────────────────────────────────────────────


class TestDesktopChannel:
    def test_macos_osascript_success(self):
        ch = DesktopChannel(app_name="App")
        with patch("pokemon_red_ai.training.alerts.platform.system", return_value="Darwin"), \
             patch("pokemon_red_ai.training.alerts.shutil.which", return_value="/usr/bin/osascript"), \
             patch("pokemon_red_ai.training.alerts.subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)
            assert ch.send(Alert("hello", "world")) is True
        assert mock_run.called
        # Verify the osascript command was constructed
        args, _ = mock_run.call_args
        cmd = args[0]
        assert cmd[0] == "osascript"
        assert "-e" in cmd
        assert any("hello" in arg for arg in cmd)
        assert any("world" in arg for arg in cmd)

    def test_macos_osascript_escapes_quotes(self):
        ch = DesktopChannel(app_name='App"name')
        with patch("pokemon_red_ai.training.alerts.platform.system", return_value="Darwin"), \
             patch("pokemon_red_ai.training.alerts.shutil.which", return_value="/usr/bin/osascript"), \
             patch("pokemon_red_ai.training.alerts.subprocess.run") as mock_run:
            ch.send(Alert('with "quote"', 'body "quote"'))
        cmd = mock_run.call_args[0][0]
        # No unescaped double quotes inside the quoted strings
        script_text = " ".join(cmd)
        # AppleScript escapes look like \" in the python string
        assert '\\"' in script_text

    def test_macos_failure_returns_false(self):
        import subprocess as _sp
        ch = DesktopChannel()
        with patch("pokemon_red_ai.training.alerts.platform.system", return_value="Darwin"), \
             patch("pokemon_red_ai.training.alerts.shutil.which", return_value="/usr/bin/osascript"), \
             patch(
                 "pokemon_red_ai.training.alerts.subprocess.run",
                 side_effect=_sp.CalledProcessError(1, ["osascript"]),
             ):
            assert ch.send(Alert("x", "y")) is False

    def test_non_macos_with_plyer(self):
        ch = DesktopChannel()
        fake_notification = Mock()
        fake_module = Mock(notification=fake_notification)
        with patch("pokemon_red_ai.training.alerts.platform.system", return_value="Linux"), \
             patch.dict("sys.modules", {"plyer": fake_module}):
            assert ch.send(Alert("t", "m")) is True
        assert fake_notification.notify.called

    def test_non_macos_without_plyer_warns_once(self, caplog):
        ch = DesktopChannel()
        with patch("pokemon_red_ai.training.alerts.platform.system", return_value="Linux"):
            # Force ImportError on plyer
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "plyer":
                    raise ImportError("plyer not installed")
                return real_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=fake_import), \
                 caplog.at_level(logging.WARNING, logger="pokemon_red_ai.training.alerts"):
                assert ch.send(Alert("t", "m")) is False
                assert ch.send(Alert("t2", "m2")) is False  # second call shouldn't re-warn
        warnings = [r for r in caplog.records if "no native notifier" in r.getMessage()]
        assert len(warnings) == 1


# ──────────────────────────────────────────────────────────────────────
# SlackChannel
# ──────────────────────────────────────────────────────────────────────


class TestSlackChannel:
    def test_unconfigured_returns_false(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        ch = SlackChannel()
        assert ch.is_configured() is False
        assert ch.send(Alert("t", "m")) is False

    def test_uses_env_var(self, monkeypatch):
        monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/XXX")
        ch = SlackChannel()
        assert ch.is_configured() is True

    def test_explicit_url_wins(self):
        ch = SlackChannel(webhook_url="https://example.com/wh")
        assert ch.webhook_url == "https://example.com/wh"

    def test_send_posts_payload(self):
        ch = SlackChannel(webhook_url="https://example.com/wh")
        fake_requests = Mock()
        fake_requests.post = Mock(return_value=Mock(status_code=200))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            assert ch.send(Alert("title", "body", severity="critical")) is True
        args, kwargs = fake_requests.post.call_args
        assert args[0] == "https://example.com/wh"
        payload = kwargs["json"]
        assert "title" in payload["text"]
        assert "body" in payload["text"]
        # critical severity emoji
        assert ":rotating_light:" in payload["text"]

    def test_http_error_returns_false(self):
        ch = SlackChannel(webhook_url="https://example.com/wh")
        fake_requests = Mock()
        fake_requests.post = Mock(return_value=Mock(status_code=500, text="server error"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            assert ch.send(Alert("t", "m")) is False

    def test_network_exception_returns_false(self):
        ch = SlackChannel(webhook_url="https://example.com/wh")
        fake_requests = Mock()
        fake_requests.post = Mock(side_effect=ConnectionError("network"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            assert ch.send(Alert("t", "m")) is False

    def test_missing_requests_returns_false(self):
        ch = SlackChannel(webhook_url="https://example.com/wh")
        # Force ImportError when the channel tries to import requests
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "requests":
                raise ImportError("requests not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            assert ch.send(Alert("t", "m")) is False


# ──────────────────────────────────────────────────────────────────────
# EmailChannel
# ──────────────────────────────────────────────────────────────────────


class TestEmailChannel:
    def test_unconfigured_returns_false(self, monkeypatch):
        for var in ["SMTP_HOST", "ALERT_EMAIL_FROM", "ALERT_EMAIL_TO"]:
            monkeypatch.delenv(var, raising=False)
        ch = EmailChannel()
        assert ch.is_configured() is False
        assert ch.send(Alert("t", "m")) is False

    def test_send_via_smtp(self, monkeypatch):
        monkeypatch.delenv("SMTP_HOST", raising=False)
        ch = EmailChannel(
            host="smtp.example.com",
            port=587,
            username="u",
            password="p",
            sender="from@example.com",
            recipients=["to@example.com"],
            use_tls=True,
        )
        assert ch.is_configured() is True

        fake_smtp = MagicMock()
        with patch("pokemon_red_ai.training.alerts.smtplib.SMTP", return_value=fake_smtp):
            assert ch.send(Alert("subj", "body", severity="critical")) is True

        # The MagicMock is used as a context manager, so the SMTP instance is
        # accessed via __enter__()
        instance = fake_smtp.__enter__.return_value
        assert instance.starttls.called
        assert instance.login.called
        assert instance.send_message.called

    def test_smtp_failure_returns_false(self):
        ch = EmailChannel(
            host="smtp.example.com",
            sender="from@example.com",
            recipients=["to@example.com"],
        )
        with patch(
            "pokemon_red_ai.training.alerts.smtplib.SMTP",
            side_effect=OSError("connection refused"),
        ):
            assert ch.send(Alert("t", "m")) is False


# ──────────────────────────────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────────────────────────────


class TestLoadAlertConfig:
    def test_loads_json(self, tmp_path):
        path = tmp_path / "alerts.json"
        path.write_text(json.dumps({"channels": {"desktop": {"enabled": True}}}))
        cfg = load_alert_config(str(path))
        assert cfg["channels"]["desktop"]["enabled"] is True

    def test_loads_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        path = tmp_path / "alerts.yaml"
        path.write_text("channels:\n  desktop:\n    enabled: true\n")
        cfg = load_alert_config(str(path))
        assert cfg["channels"]["desktop"]["enabled"] is True

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_alert_config(str(tmp_path / "nope.json"))

    def test_malformed_json_raises_value_error(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError):
            load_alert_config(str(path))


class TestChannelsFromConfig:
    def test_empty_config_yields_no_channels(self):
        assert channels_from_config({}) == []

    def test_desktop_enabled(self):
        chans = channels_from_config(
            {"channels": {"desktop": {"enabled": True, "app_name": "Foo"}}}
        )
        assert len(chans) == 1
        assert isinstance(chans[0], DesktopChannel)
        assert chans[0].app_name == "Foo"

    def test_slack_with_webhook(self, monkeypatch):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        chans = channels_from_config(
            {
                "channels": {
                    "slack": {
                        "enabled": True,
                        "webhook_url": "https://example.com/wh",
                    }
                }
            }
        )
        assert len(chans) == 1
        assert isinstance(chans[0], SlackChannel)

    def test_slack_without_webhook_skipped(self, monkeypatch, caplog):
        monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
        with caplog.at_level(logging.WARNING, logger="pokemon_red_ai.training.alerts"):
            chans = channels_from_config(
                {"channels": {"slack": {"enabled": True}}}
            )
        assert chans == []
        assert any("webhook_url" in r.getMessage() for r in caplog.records)

    def test_email_with_full_config(self):
        chans = channels_from_config(
            {
                "channels": {
                    "email": {
                        "enabled": True,
                        "host": "smtp.example.com",
                        "sender": "from@example.com",
                        "recipients": ["to@example.com"],
                    }
                }
            }
        )
        assert len(chans) == 1
        assert isinstance(chans[0], EmailChannel)

    def test_multiple_channels(self):
        chans = channels_from_config(
            {
                "channels": {
                    "desktop": {"enabled": True},
                    "slack": {"enabled": True, "webhook_url": "https://x.example/wh"},
                }
            }
        )
        types = {type(c).__name__ for c in chans}
        assert types == {"DesktopChannel", "SlackChannel"}


# ──────────────────────────────────────────────────────────────────────
# TrainingAlertCallback — initialisation
# ──────────────────────────────────────────────────────────────────────


class TestCallbackInit:
    def test_log_channel_always_added(self):
        cb = _make_callback(channels=[])
        assert any(isinstance(c, LogChannel) for c in cb.channels)

    def test_extra_channels_appended(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        assert rec in cb.channels
        assert any(isinstance(c, LogChannel) for c in cb.channels)

    def test_clamps_negative_values(self):
        cb = _make_callback(
            channels=[],
            plateau_episodes=-5,
            plateau_min_episodes=-1,
            checkpoint_alert_freq=-100,
            cooldown_seconds=-2.0,
        )
        assert cb.plateau_episodes >= 1
        assert cb.plateau_min_episodes >= 0
        assert cb.checkpoint_alert_freq >= 0
        assert cb.cooldown_seconds >= 0


# ──────────────────────────────────────────────────────────────────────
# Trigger: first badge / new max badge
# ──────────────────────────────────────────────────────────────────────


class TestBadgeTrigger:
    def test_first_badge_alert(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])

        # No badges yet — no alert
        _step(cb, info={"badges_earned": 0}, step=10)
        assert rec.alerts == []

        # First badge!
        _step(cb, info={"badges_earned": 1}, step=20)
        assert len(rec.alerts) == 1
        assert "First badge" in rec.alerts[0].title

    def test_subsequent_badges_alert_individually(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"badges_earned": 1}, step=10)
        _step(cb, info={"badges_earned": 2}, step=20)
        _step(cb, info={"badges_earned": 3}, step=30)
        assert len(rec.alerts) == 3
        # Different titles — first is "First badge", subsequent are "#N"
        titles = [a.title for a in rec.alerts]
        assert titles[0].startswith("First badge")
        assert "#2" in titles[1]
        assert "#3" in titles[2]

    def test_no_alert_when_badge_count_unchanged(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"badges_earned": 1}, step=10)
        rec.alerts.clear()
        _step(cb, info={"badges_earned": 1}, step=20)
        _step(cb, info={"badges_earned": 1}, step=30)
        assert rec.alerts == []

    def test_disabled_via_flag(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], notify_first_badge=False)
        _step(cb, info={"badges_earned": 1}, step=10)
        assert rec.alerts == []


# ──────────────────────────────────────────────────────────────────────
# Trigger: new map
# ──────────────────────────────────────────────────────────────────────


class TestNewMapTrigger:
    def test_first_map_does_not_alert(self):
        """The very first map seen is just the spawn point — no alert."""
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"unique_maps_list": [0]}, step=5)
        # Should track but not alert
        assert rec.alerts == []
        assert 0 in cb._seen_maps

    def test_second_map_alerts(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"unique_maps_list": [0]}, step=5)
        _step(cb, info={"unique_maps_list": [0, 1]}, step=10)
        assert len(rec.alerts) == 1
        assert "1" in rec.alerts[0].title

    def test_repeated_maps_do_not_realert(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"unique_maps_list": [0]}, step=5)
        _step(cb, info={"unique_maps_list": [0, 1]}, step=10)
        rec.alerts.clear()
        _step(cb, info={"unique_maps_list": [0, 1]}, step=15)
        _step(cb, info={"unique_maps_list": [0, 1]}, step=20)
        assert rec.alerts == []

    def test_handles_non_int_map_ids(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(cb, info={"unique_maps_list": [0, "garbage", 2]}, step=10)
        # Only valid ints kept; first valid map (0) is skipped → 2 alerts? No,
        # only 2 valid ints total — the non-first one (2) alerts.
        assert len(rec.alerts) == 1
        assert "2" in rec.alerts[0].title


# ──────────────────────────────────────────────────────────────────────
# Trigger: new event flag
# ──────────────────────────────────────────────────────────────────────


class TestNewFlagTrigger:
    def test_each_unique_flag_alerts_once(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        _step(
            cb,
            info={"event_progress": {"triggered_names": ["EVENT_GOT_STARTER"]}},
            step=10,
        )
        _step(
            cb,
            info={
                "event_progress": {
                    "triggered_names": ["EVENT_GOT_STARTER", "EVENT_GOT_POKEDEX"]
                }
            },
            step=20,
        )
        # Only the new flag triggers an alert each time
        assert len(rec.alerts) == 2
        assert "EVENT_GOT_STARTER" in rec.alerts[0].title
        assert "EVENT_GOT_POKEDEX" in rec.alerts[1].title

    def test_repeated_flag_does_not_realert(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        info = {"event_progress": {"triggered_names": ["EVENT_X"]}}
        _step(cb, info=info, step=10)
        rec.alerts.clear()
        _step(cb, info=info, step=20)
        _step(cb, info=info, step=30)
        assert rec.alerts == []

    def test_disabled(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], notify_new_flag=False)
        _step(
            cb,
            info={"event_progress": {"triggered_names": ["EVENT_X"]}},
            step=10,
        )
        assert rec.alerts == []


# ──────────────────────────────────────────────────────────────────────
# Trigger: reward plateau
# ──────────────────────────────────────────────────────────────────────


class TestPlateauTrigger:
    def test_plateau_fires_after_threshold(self):
        rec = RecordingChannel()
        cb = _make_callback(
            channels=[rec],
            plateau_episodes=3,
            plateau_min_episodes=2,
        )

        # Ep 1 — best=10, no alert
        _step(cb, info={"episode": {"r": 10.0}}, done=True, step=100)
        # Ep 2 — best still 10, +1 stale
        _step(cb, info={"episode": {"r": 5.0}}, done=True, step=200)
        # Ep 3 — +2 stale
        _step(cb, info={"episode": {"r": 5.0}}, done=True, step=300)
        # Ep 4 — +3 stale, threshold met → alert
        _step(cb, info={"episode": {"r": 5.0}}, done=True, step=400)

        assert any("plateau" in a.title.lower() for a in rec.alerts)

    def test_plateau_does_not_fire_below_min_episodes(self):
        rec = RecordingChannel()
        cb = _make_callback(
            channels=[rec],
            plateau_episodes=1,  # plateau as soon as 1 stale
            plateau_min_episodes=10,  # but warmup of 10
        )
        # 5 episodes, none improving, but warmup not met
        for i in range(5):
            _step(
                cb,
                info={"episode": {"r": 0.0 if i == 0 else -1.0}},
                done=True,
                step=100 * (i + 1),
            )
        assert not any("plateau" in a.title.lower() for a in rec.alerts)

    def test_plateau_resets_on_improvement(self):
        rec = RecordingChannel()
        cb = _make_callback(
            channels=[rec],
            plateau_episodes=3,
            plateau_min_episodes=1,
        )
        # 4 stale → plateau alert
        _step(cb, info={"episode": {"r": 10.0}}, done=True, step=100)
        for i in range(4):
            _step(
                cb,
                info={"episode": {"r": 5.0}},
                done=True,
                step=200 + 100 * i,
            )
        plateau_alerts_before = sum(
            1 for a in rec.alerts if "plateau" in a.title.lower()
        )
        assert plateau_alerts_before >= 1

        # New best — should reset
        _step(cb, info={"episode": {"r": 100.0}}, done=True, step=1000)
        rec.alerts.clear()

        # Now stale again — should re-fire after threshold
        for i in range(4):
            _step(
                cb,
                info={"episode": {"r": 5.0}},
                done=True,
                step=1100 + 100 * i,
            )
        assert any("plateau" in a.title.lower() for a in rec.alerts)


# ──────────────────────────────────────────────────────────────────────
# Trigger: checkpoint
# ──────────────────────────────────────────────────────────────────────


class TestCheckpointTrigger:
    def test_fires_at_threshold(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], checkpoint_alert_freq=100)
        _step(cb, info={}, step=50)
        assert not any("checkpoint" in a.title.lower() for a in rec.alerts)
        _step(cb, info={}, step=150)
        assert any("checkpoint" in a.title.lower() for a in rec.alerts)

    def test_disabled_when_freq_zero(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], checkpoint_alert_freq=0)
        _step(cb, info={}, step=1_000_000)
        assert not any("checkpoint" in a.title.lower() for a in rec.alerts)

    def test_subsequent_checkpoints_each_alert(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], checkpoint_alert_freq=100)
        for s in [100, 200, 300]:
            _step(cb, info={}, step=s)
        n_ckpt = sum(1 for a in rec.alerts if "checkpoint" in a.title.lower())
        assert n_ckpt == 3


# ──────────────────────────────────────────────────────────────────────
# Cooldown / dedup
# ──────────────────────────────────────────────────────────────────────


class FakeClock:
    """Test clock — call it to read, ``advance(seconds)`` to move forward."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float):
        self.t += seconds


class TestCooldown:
    def test_same_key_within_cooldown_suppressed(self):
        rec = RecordingChannel()
        clock = FakeClock()
        cb = _make_callback(
            channels=[rec],
            cooldown_seconds=60.0,
            checkpoint_alert_freq=100,
            clock=clock,
        )
        _step(cb, info={}, step=100)
        first_n = len(rec.alerts)
        # Advance only 10s — still within cooldown
        clock.advance(10.0)
        # Force a same-keyed alert via the dispatcher
        cb._dispatch(Alert("dup", "msg"), cooldown_key=f"checkpoint_100")
        assert len(rec.alerts) == first_n  # suppressed

    def test_different_keys_independent(self):
        rec = RecordingChannel()
        clock = FakeClock()
        cb = _make_callback(channels=[rec], cooldown_seconds=60.0, clock=clock)
        cb._dispatch(Alert("a", "m"), cooldown_key="key_a")
        cb._dispatch(Alert("b", "m"), cooldown_key="key_b")
        assert len(rec.alerts) == 2

    def test_cooldown_expires(self):
        rec = RecordingChannel()
        clock = FakeClock()
        cb = _make_callback(channels=[rec], cooldown_seconds=10.0, clock=clock)
        cb._dispatch(Alert("x", "m"), cooldown_key="k")
        clock.advance(15.0)
        cb._dispatch(Alert("x", "m"), cooldown_key="k")
        assert len(rec.alerts) == 2

    def test_none_key_never_deduplicated(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], cooldown_seconds=1000.0)
        for _ in range(3):
            cb._dispatch(Alert("crash", "m", severity="critical"), cooldown_key=None)
        assert len(rec.alerts) == 3


# ──────────────────────────────────────────────────────────────────────
# Multi-channel & failure handling
# ──────────────────────────────────────────────────────────────────────


class TestDispatch:
    def test_alerts_go_to_all_channels(self):
        rec_a, rec_b = RecordingChannel(), RecordingChannel()
        cb = _make_callback(channels=[rec_a, rec_b])
        _step(cb, info={"badges_earned": 1}, step=10)
        assert len(rec_a.alerts) == 1
        assert len(rec_b.alerts) == 1

    def test_failing_channel_does_not_break_others(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[FailingChannel(), rec])
        _step(cb, info={"badges_earned": 1}, step=10)
        # Recording channel still received the alert
        assert len(rec.alerts) == 1


# ──────────────────────────────────────────────────────────────────────
# Crash notification
# ──────────────────────────────────────────────────────────────────────


class TestNotifyCrash:
    def test_sends_critical_alert(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec])
        try:
            raise ValueError("nan loss")
        except ValueError as e:
            cb.notify_crash(e)
        assert len(rec.alerts) == 1
        assert rec.alerts[0].severity == "critical"
        assert "nan loss" in rec.alerts[0].message
        assert "ValueError" in rec.alerts[0].message

    def test_crash_alerts_not_deduplicated(self):
        rec = RecordingChannel()
        cb = _make_callback(channels=[rec], cooldown_seconds=10_000.0)
        for _ in range(3):
            cb.notify_crash(RuntimeError("boom"))
        assert len(rec.alerts) == 3


# ──────────────────────────────────────────────────────────────────────
# Robustness
# ──────────────────────────────────────────────────────────────────────


class TestRobustness:
    def test_empty_info_dict_does_not_crash(self):
        cb = _make_callback(channels=[RecordingChannel()])
        _step(cb, info={}, step=10)
        # No alerts, no exception

    def test_missing_dones_or_infos(self):
        cb = _make_callback(channels=[RecordingChannel()])
        cb.locals = {}  # neither dones nor infos
        cb.num_timesteps = 100
        # Should return True without crashing
        assert cb._on_step() is True

    def test_works_alongside_other_callbacks(self):
        """TrainingAlertCallback must be combinable in a SB3 CallbackList."""
        from stable_baselines3.common.callbacks import CallbackList

        cb = _make_callback(channels=[RecordingChannel()])
        list_cb = CallbackList([cb])
        # Just verify construction doesn't blow up
        assert list_cb.callbacks == [cb]
