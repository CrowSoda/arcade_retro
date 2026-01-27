"""
WebSocket routing to appropriate handlers.

Extracted from server.py using strangler fig pattern.
"""

import logging
from collections.abc import Callable
from urllib.parse import urlparse

logger = logging.getLogger("g20.server")


class WebSocketRouter:
    """Routes WebSocket connections to appropriate handlers based on path."""

    def __init__(self):
        self._routes: dict[str, Callable] = {}

    def route(self, path: str):
        """Decorator to register a handler for a path."""

        def decorator(handler: Callable):
            self._routes[path] = handler
            logger.debug(f"Registered WS handler: {path}")
            return handler

        return decorator

    def add_route(self, path: str, handler: Callable):
        """Programmatically add a route."""
        self._routes[path] = handler

    async def handle(self, websocket, path: str):
        """Route incoming connection to appropriate handler."""
        # Normalize path
        parsed = urlparse(path)
        clean_path = parsed.path.rstrip("/")

        handler = self._routes.get(clean_path)
        if handler is None:
            # Try prefix matching
            for route_path, route_handler in self._routes.items():
                if clean_path.startswith(route_path):
                    handler = route_handler
                    break

        if handler is None:
            logger.warning(f"No handler for path: {clean_path}")
            await websocket.close(1008, f"Unknown path: {clean_path}")
            return

        logger.info(f"Routing {clean_path} to {handler.__name__}")
        await handler(websocket)


# Global router instance
ws_router = WebSocketRouter()
