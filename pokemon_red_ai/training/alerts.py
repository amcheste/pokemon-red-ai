"""
Training alerting system for Pokemon Red RL.

Provides :class:`TrainingAlertCallback` — an SB3 callback that fires
notifications when interesting events happen during a training run:

* **Game milestones** — first badge, new max badge count, new map
  discovered, new event flag triggered
* **Reward plateaus** — no improvement in best episode reward for N
  episodes (configurable)
* **Checkpoint saves** — fires every ``checkpoint_alert_freq`` timesteps
* **Training crashes** — call :meth:`TrainingAlertCallback.notify_crash`
  from your training script's ``except`` block

Alerts go through one or more :class:`AlertChannel` implementations.
Three are bundled:

* :class:`LogChannel` — always-on fallback that logs at INFO level.
  Useful for tests and headless runs.
* :class:`DesktopChannel` — macOS-native notifications via ``osascript``
  (no extra dependency).  Falls back to ``plyer`` on other platforms if
  installed; otherwise no-ops with a warning.
* :class:`SlackChannel` — POSTs to a Slack incoming-webhook URL.

Channel configuration can be supplied as constructor args or loaded from
a YAML file via :func:`load_alert_config`.

All alert sends are best-effort: failures are caught and logged at WARN
so a flaky webhook never crashes a 10M-step training run.
"""

from __future__ import annotations

import abc
import json
import logging
import os
import platform
import shutil
import smtplib
import subprocess
from dataclasses import dataclass, field
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Sequence

from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Alert payload
# ──────────────────────────────────────────────────────────────────────


@dataclass
class Alert:
    """A single alert event delivered to one or more channels."""

    title: str
    message: str
    severity: str = "info"  # "info" | "warning" | "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────
# Channels
# ──────────────────────────────────────────────────────────────────────


class AlertChannel(abc.ABC):
    """Base class for an alert delivery channel."""

    @abc.abstractmethod
    def send(self, alert: Alert) -> bool:
        """Deliver ``alert``.  Return True on success, False on failure.

        Implementations must not raise — they should catch their own
        exceptions and return False.
        """


class LogChannel(AlertChannel):
    """Logs alerts via the standard :mod:`logging` module.

    Always available and used as the default fallback.  ``severity`` maps
    to a log level (``info``/``warning``/``critical`` → INFO/WARNING/ERROR).
    """

    _LEVELS = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.ERROR,
    }

    def __init__(self, log: Optional[logging.Logger] = None):
        self._logger = log or logger

    def send(self, alert: Alert) -> bool:
        level = self._LEVELS.get(alert.severity, logging.INFO)
        try:
            self._logger.log(
                level,
                "ALERT [%s] %s — %s",
                alert.severity.upper(),
                alert.title,
                alert.message,
            )
            return True
        except Exception:  # pragma: no cover — logging never raises
            return False


class DesktopChannel(AlertChannel):
    """Cross-platform desktop notifications.

    On macOS uses ``osascript`` (no extra dependency).  On other
    platforms tries to import :mod:`plyer`.  If neither is available,
    :meth:`send` returns False and logs a warning once.
    """

    def __init__(self, app_name: str = "Pokemon Red RL"):
        self.app_name = app_name
        self._warned_unavailable = False

    @staticmethod
    def _osascript_available() -> bool:
        return platform.system() == "Darwin" and shutil.which("osascript") is not None

    def send(self, alert: Alert) -> bool:
        if self._osascript_available():
            return self._send_macos(alert)
        return self._send_plyer(alert)

    def _send_macos(self, alert: Alert) -> bool:
        # Escape double quotes for AppleScript string literals
        title = alert.title.replace('"', '\\"')
        body = alert.message.replace('"', '\\"')
        app = self.app_name.replace('"', '\\"')
        script = (
            f'display notification "{body}" '
            f'with title "{app}" subtitle "{title}"'
        )
        try:
            subprocess.run(
                ["osascript", "-e", script],
                check=True,
                capture_output=True,
                timeout=5,
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
            logger.warning(f"DesktopChannel (osascript) failed: {exc}")
            return False

    def _send_plyer(self, alert: Alert) -> bool:
        try:
            from plyer import notification  # type: ignore
        except ImportError:
            if not self._warned_unavailable:
                logger.warning(
                    "DesktopChannel: no native notifier available "
                    "(macOS osascript missing and plyer not installed). "
                    "Install plyer or run on macOS for desktop alerts."
                )
                self._warned_unavailable = True
            return False

        try:
            notification.notify(
                title=f"{self.app_name}: {alert.title}",
                message=alert.message,
                app_name=self.app_name,
                timeout=5,
            )
            return True
        except Exception as exc:
            logger.warning(f"DesktopChannel (plyer) failed: {exc}")
            return False


class SlackChannel(AlertChannel):
    """Posts alerts to a Slack incoming webhook.

    The webhook URL can come from:

    * the ``webhook_url`` constructor arg, or
    * the ``SLACK_WEBHOOK_URL`` environment variable.

    Severity emoji mapping: info → :information_source:, warning → :warning:,
    critical → :rotating_light:.
    """

    _EMOJI = {
        "info": ":information_source:",
        "warning": ":warning:",
        "critical": ":rotating_light:",
    }

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        username: str = "Pokemon Red RL",
        timeout: float = 5.0,
    ):
        self.webhook_url = webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.username = username
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.webhook_url)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        try:
            import requests  # type: ignore
        except ImportError:
            logger.warning(
                "SlackChannel: 'requests' is not installed. "
                "Install with: pip install requests"
            )
            return False

        emoji = self._EMOJI.get(alert.severity, ":bell:")
        text = f"{emoji} *{alert.title}*\n{alert.message}"
        payload = {"text": text, "username": self.username}

        try:
            response = requests.post(
                self.webhook_url,  # type: ignore[arg-type]
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                logger.warning(
                    f"SlackChannel returned HTTP {response.status_code}: "
                    f"{response.text[:200]}"
                )
                return False
            return True
        except Exception as exc:
            logger.warning(f"SlackChannel failed: {exc}")
            return False


class EmailChannel(AlertChannel):
    """Sends alerts via SMTP.

    Useful for unattended overnight runs.  Configure via constructor args
    or environment variables ``SMTP_HOST``, ``SMTP_PORT``, ``SMTP_USER``,
    ``SMTP_PASSWORD``, ``ALERT_EMAIL_FROM``, ``ALERT_EMAIL_TO``.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        sender: Optional[str] = None,
        recipients: Optional[Sequence[str]] = None,
        use_tls: bool = True,
        timeout: float = 10.0,
    ):
        self.host = host or os.environ.get("SMTP_HOST")
        port_str = os.environ.get("SMTP_PORT", "")
        self.port = port if port is not None else (int(port_str) if port_str else 587)
        self.username = username or os.environ.get("SMTP_USER")
        self.password = password or os.environ.get("SMTP_PASSWORD")
        self.sender = sender or os.environ.get("ALERT_EMAIL_FROM") or self.username
        env_to = os.environ.get("ALERT_EMAIL_TO", "")
        self.recipients = list(recipients) if recipients else (
            [r.strip() for r in env_to.split(",") if r.strip()]
        )
        self.use_tls = use_tls
        self.timeout = timeout

    def is_configured(self) -> bool:
        return bool(self.host and self.sender and self.recipients)

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            return False

        msg = EmailMessage()
        msg["Subject"] = f"[{alert.severity.upper()}] {alert.title}"
        msg["From"] = self.sender  # type: ignore[assignment]
        msg["To"] = ", ".join(self.recipients)
        msg.set_content(alert.message)

        try:
            with smtplib.SMTP(self.host, self.port, timeout=self.timeout) as s:  # type: ignore[arg-type]
                if self.use_tls:
                    s.starttls()
                if self.username and self.password:
                    s.login(self.username, self.password)
                s.send_message(msg)
            return True
        except Exception as exc:
            logger.warning(f"EmailChannel failed: {exc}")
            return False


# ──────────────────────────────────────────────────────────────────────
# Config loader
# ──────────────────────────────────────────────────────────────────────


def load_alert_config(path: str) -> Dict[str, Any]:
    """Load alert config from a YAML or JSON file.

    Schema (all keys optional)::

        channels:
          desktop:
            enabled: true
            app_name: "Pokemon Red RL"
          slack:
            enabled: true
            webhook_url: "https://hooks.slack.com/..."
          email:
            enabled: false
            host: smtp.gmail.com
            port: 587
            sender: alerts@example.com
            recipients: [me@example.com]
        triggers:
          plateau_episodes: 50
          plateau_min_episodes: 20
          checkpoint_alert_freq: 100000
          cooldown_seconds: 60

    Returns the parsed dict.  Raises :class:`FileNotFoundError` if the
    file is missing and :class:`ValueError` if it can't be parsed.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Alert config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()

    try:
        if path.endswith((".yaml", ".yml")):
            import yaml  # type: ignore
            return yaml.safe_load(text) or {}
        return json.loads(text)
    except Exception as exc:
        raise ValueError(f"Failed to parse alert config {path}: {exc}") from exc


def channels_from_config(config: Dict[str, Any]) -> List[AlertChannel]:
    """Build a list of :class:`AlertChannel` instances from a config dict.

    Returns an empty list if no channels are enabled.  ``LogChannel`` is
    not added by default — :class:`TrainingAlertCallback` adds it
    automatically as a fallback.
    """
    channels_cfg = (config or {}).get("channels", {}) or {}
    out: List[AlertChannel] = []

    desktop = channels_cfg.get("desktop") or {}
    if desktop.get("enabled"):
        out.append(DesktopChannel(app_name=desktop.get("app_name", "Pokemon Red RL")))

    slack = channels_cfg.get("slack") or {}
    if slack.get("enabled"):
        ch = SlackChannel(
            webhook_url=slack.get("webhook_url"),
            username=slack.get("username", "Pokemon Red RL"),
        )
        if ch.is_configured():
            out.append(ch)
        else:
            logger.warning(
                "Slack channel enabled but webhook_url not set "
                "(neither config nor SLACK_WEBHOOK_URL env var)."
            )

    email = channels_cfg.get("email") or {}
    if email.get("enabled"):
        ch_email = EmailChannel(
            host=email.get("host"),
            port=email.get("port"),
            username=email.get("username"),
            password=email.get("password"),
            sender=email.get("sender"),
            recipients=email.get("recipients"),
            use_tls=bool(email.get("use_tls", True)),
        )
        if ch_email.is_configured():
            out.append(ch_email)
        else:
            logger.warning("Email channel enabled but SMTP config incomplete.")

    return out


# ──────────────────────────────────────────────────────────────────────
# Callback
# ──────────────────────────────────────────────────────────────────────


class TrainingAlertCallback(BaseCallback):
    """SB3 callback that emits alerts on training milestones.

    Triggers:

    * **First badge** — fires once when ``info["badges_earned"] > 0``.
    * **New max badge** — fires whenever the badge count exceeds the
      previous maximum (so 1st, 2nd, 3rd, …  badge each ping).
    * **New map** — fires whenever a previously-unseen map appears in
      ``info["unique_maps_list"]``.
    * **New event flag** — fires whenever a new flag name appears in
      ``info["event_progress"]["triggered_names"]``.
    * **Reward plateau** — fires once when the best episode reward has
      not improved for ``plateau_episodes`` consecutive episodes (after
      a warmup of ``plateau_min_episodes``).  Re-armed when reward
      improves.
    * **Checkpoint** — fires every ``checkpoint_alert_freq`` timesteps.
      Set to 0 to disable.

    Per-trigger cooldown (``cooldown_seconds``) prevents spam if the
    same kind of event fires repeatedly in a short window.

    The callback is idempotent and safe to register alongside
    :class:`MonitoringCallback` and :class:`TrainingCallback`.
    """

    def __init__(
        self,
        channels: Optional[Sequence[AlertChannel]] = None,
        plateau_episodes: int = 50,
        plateau_min_episodes: int = 20,
        checkpoint_alert_freq: int = 100_000,
        cooldown_seconds: float = 60.0,
        notify_first_badge: bool = True,
        notify_new_map: bool = True,
        notify_new_flag: bool = True,
        notify_plateau: bool = True,
        notify_checkpoint: bool = True,
        verbose: int = 1,
        clock: Optional[Any] = None,
    ):
        """
        Args:
            channels: List of :class:`AlertChannel` instances.  If empty
                or None, only :class:`LogChannel` is used.
            plateau_episodes: Plateau alert fires after this many
                episodes without best-reward improvement.
            plateau_min_episodes: Minimum episode count before plateau
                alerts can fire (warmup).
            checkpoint_alert_freq: Timesteps between checkpoint alerts.
                Set to 0 to disable.
            cooldown_seconds: Per-trigger-type cooldown to prevent spam.
            notify_*: Toggle individual trigger types.
            verbose: 0 silent, 1 info, 2 debug.
            clock: Optional ``time.monotonic``-like callable.  Lets tests
                inject a fake clock.
        """
        super().__init__(verbose)

        # Always include LogChannel as a fallback so verbose messages go
        # somewhere even if no other channel is configured.
        self.channels: List[AlertChannel] = [LogChannel()]
        if channels:
            self.channels.extend(channels)

        self.plateau_episodes = max(1, int(plateau_episodes))
        self.plateau_min_episodes = max(0, int(plateau_min_episodes))
        self.checkpoint_alert_freq = max(0, int(checkpoint_alert_freq))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))

        self.notify_first_badge = bool(notify_first_badge)
        self.notify_new_map = bool(notify_new_map)
        self.notify_new_flag = bool(notify_new_flag)
        self.notify_plateau = bool(notify_plateau)
        self.notify_checkpoint = bool(notify_checkpoint)

        # Test-injectable clock; default to monotonic
        if clock is None:
            import time
            self._clock = time.monotonic
        else:
            self._clock = clock

        # State
        self._first_badge_fired: bool = False
        self._max_badges: int = 0
        self._seen_maps: set = set()
        self._seen_flags: set = set()
        self._best_reward: float = -float("inf")
        self._episodes_since_improvement: int = 0
        self._episode_count: int = 0
        self._plateau_fired: bool = False
        self._next_checkpoint_step: int = self.checkpoint_alert_freq
        self._last_send: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public hooks
    # ------------------------------------------------------------------

    def notify_crash(self, exc: BaseException) -> None:
        """Send a critical-severity alert about a training crash.

        Call from your training script's ``except`` block — exception
        propagation does not flow through SB3 callback methods, so this
        helper exists for the script to invoke directly.
        """
        message = f"{type(exc).__name__}: {exc}"
        self._dispatch(
            Alert(
                title="Training crashed",
                message=message,
                severity="critical",
                metadata={"exception_type": type(exc).__name__},
            ),
            cooldown_key=None,  # crashes never deduplicated
        )

    # ------------------------------------------------------------------
    # SB3 hooks
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                self._handle_step_info(info, bool(done))

        self._maybe_alert_checkpoint()
        return True

    # ------------------------------------------------------------------
    # Internal: per-step / per-episode handlers
    # ------------------------------------------------------------------

    def _handle_step_info(self, info: Dict[str, Any], done: bool) -> None:
        # Badge milestones — checked every step, not just at episode end,
        # because the player can earn a badge mid-episode.
        if self.notify_first_badge:
            badges = int(info.get("badges_earned", 0) or 0)
            if badges > self._max_badges:
                if not self._first_badge_fired and badges >= 1:
                    self._first_badge_fired = True
                    self._dispatch(
                        Alert(
                            title="First badge earned!",
                            message=(
                                f"The agent just earned its first badge "
                                f"at step {self.num_timesteps:,}."
                            ),
                            severity="info",
                            metadata={"badges": badges, "step": self.num_timesteps},
                        ),
                        cooldown_key="first_badge",
                    )
                else:
                    self._dispatch(
                        Alert(
                            title=f"Badge #{badges} earned",
                            message=(
                                f"Total badges: {badges}. "
                                f"Step {self.num_timesteps:,}."
                            ),
                            severity="info",
                            metadata={"badges": badges, "step": self.num_timesteps},
                        ),
                        cooldown_key=f"badge_{badges}",
                    )
                self._max_badges = badges

        # New map discovery
        if self.notify_new_map:
            maps_list = info.get("unique_maps_list") or []
            for m in maps_list:
                try:
                    m_int = int(m)
                except (TypeError, ValueError):
                    continue
                if m_int not in self._seen_maps:
                    self._seen_maps.add(m_int)
                    # Only alert after we have at least one map — first
                    # map at episode 1 is just the spawn point.
                    if len(self._seen_maps) > 1:
                        self._dispatch(
                            Alert(
                                title=f"New map discovered: {m_int}",
                                message=(
                                    f"Total unique maps: {len(self._seen_maps)}. "
                                    f"Step {self.num_timesteps:,}."
                                ),
                                severity="info",
                                metadata={
                                    "map_id": m_int,
                                    "total_maps": len(self._seen_maps),
                                    "step": self.num_timesteps,
                                },
                            ),
                            cooldown_key=f"new_map_{m_int}",
                        )

        # New event flag triggered
        if self.notify_new_flag:
            event_progress = info.get("event_progress") or {}
            triggered = event_progress.get("triggered_names") or []
            for name in triggered:
                if not isinstance(name, str):
                    continue
                if name not in self._seen_flags:
                    self._seen_flags.add(name)
                    self._dispatch(
                        Alert(
                            title=f"Event flag: {name}",
                            message=(
                                f"Total flags triggered: {len(self._seen_flags)}. "
                                f"Step {self.num_timesteps:,}."
                            ),
                            severity="info",
                            metadata={
                                "flag": name,
                                "total_flags": len(self._seen_flags),
                                "step": self.num_timesteps,
                            },
                        ),
                        cooldown_key=f"flag_{name}",
                    )

        # Episode-end bookkeeping for plateau detection
        if done:
            self._episode_count += 1
            ep_info = info.get("episode") or {}
            try:
                reward = float(ep_info.get("r", 0.0))
            except (TypeError, ValueError):
                reward = 0.0

            if reward > self._best_reward:
                self._best_reward = reward
                self._episodes_since_improvement = 0
                self._plateau_fired = False
            else:
                self._episodes_since_improvement += 1
                self._maybe_alert_plateau()

    def _maybe_alert_plateau(self) -> None:
        if not self.notify_plateau or self._plateau_fired:
            return
        if self._episode_count < self.plateau_min_episodes:
            return
        if self._episodes_since_improvement < self.plateau_episodes:
            return

        self._plateau_fired = True
        self._dispatch(
            Alert(
                title="Reward plateau detected",
                message=(
                    f"No improvement in best reward "
                    f"({self._best_reward:.2f}) for "
                    f"{self._episodes_since_improvement} episodes."
                ),
                severity="warning",
                metadata={
                    "best_reward": self._best_reward,
                    "stale_episodes": self._episodes_since_improvement,
                },
            ),
            cooldown_key="plateau",
        )

    def _maybe_alert_checkpoint(self) -> None:
        if not self.notify_checkpoint or self.checkpoint_alert_freq <= 0:
            return
        if self.num_timesteps < self._next_checkpoint_step:
            return

        step = int(self.num_timesteps)
        self._next_checkpoint_step = step + self.checkpoint_alert_freq
        self._dispatch(
            Alert(
                title=f"Checkpoint @ {step:,} steps",
                message=(
                    f"Training reached {step:,} steps. "
                    f"Best reward so far: {self._best_reward:.2f}."
                ),
                severity="info",
                metadata={
                    "step": step,
                    "best_reward": self._best_reward,
                },
            ),
            cooldown_key=f"checkpoint_{step}",
        )

    # ------------------------------------------------------------------
    # Internal: dispatching with cooldown
    # ------------------------------------------------------------------

    def _dispatch(self, alert: Alert, cooldown_key: Optional[str]) -> None:
        """Send ``alert`` to all channels, honoring per-key cooldown."""
        if cooldown_key is not None and self.cooldown_seconds > 0:
            now = self._clock()
            last = self._last_send.get(cooldown_key, -float("inf"))
            if now - last < self.cooldown_seconds:
                if self.verbose >= 2:
                    logger.debug(
                        f"Alert '{cooldown_key}' suppressed by cooldown "
                        f"({now - last:.1f}s < {self.cooldown_seconds:.1f}s)"
                    )
                return
            self._last_send[cooldown_key] = now

        for channel in self.channels:
            try:
                channel.send(alert)
            except Exception as exc:  # pragma: no cover — channels catch their own
                logger.warning(
                    f"Alert channel {type(channel).__name__} raised: {exc}"
                )
