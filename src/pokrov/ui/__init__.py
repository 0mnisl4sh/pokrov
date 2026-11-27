"""Offline Web Dashboard and Live Playground (/ui) for Pokrov."""

from pokrov.ui.routes import (
    PLAYGROUND_PRESETS,
    PlaygroundRequest,
    SwitchProfileRequest,
    router,
)

__all__ = [
    "PLAYGROUND_PRESETS",
    "PlaygroundRequest",
    "SwitchProfileRequest",
    "router",
]
