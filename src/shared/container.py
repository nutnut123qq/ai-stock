"""Dependency Injection container for the AI Service."""
from typing import Type, TypeVar, Callable, Any, Dict, Optional
from functools import lru_cache
from src.shared.logging import get_logger

T = TypeVar("T")
logger = get_logger(__name__)


class Container:
    """Simple dependency injection container."""
    
    def __init__(self):
        """Initialize the container."""
        self._singletons: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._transients: Dict[str, Callable[[], Any]] = {}
    
    def register_singleton(
        self,
        key: str,
        instance: Any,
        override: bool = False
    ) -> None:
        """Register a singleton instance."""
        if key in self._singletons and not override:
            logger.warning(f"Singleton {key} already registered. Use override=True to replace.")
            return
        
        self._singletons[key] = instance
        logger.debug(f"Registered singleton: {key}")
    
    def register_factory(
        self,
        key: str,
        factory: Callable[[], Any],
        override: bool = False
    ) -> None:
        """Register a factory function (creates new instance each time)."""
        if key in self._transients and not override:
            logger.warning(f"Factory {key} already registered. Use override=True to replace.")
            return
        
        self._transients[key] = factory
        logger.debug(f"Registered factory: {key}")
    
    def register_singleton_factory(
        self,
        key: str,
        factory: Callable[[], Any],
        override: bool = False
    ) -> None:
        """Register a singleton factory (creates instance once, reuses it)."""
        if key in self._factories and not override:
            logger.warning(f"Singleton factory {key} already registered. Use override=True to replace.")
            return
        
        self._factories[key] = factory
        logger.debug(f"Registered singleton factory: {key}")
    
    def resolve(self, key: str) -> Any:
        """Resolve a dependency by key."""
        # Check singletons
        if key in self._singletons:
            return self._singletons[key]
        
        # Check singleton factories
        if key in self._factories:
            if key not in self._singletons:
                self._singletons[key] = self._factories[key]()
            return self._singletons[key]
        
        # Check transient factories
        if key in self._transients:
            return self._transients[key]()
        
        raise ValueError(f"Dependency {key} not registered in container")
    
    def get(self, key: str, default: Any = None) -> Optional[Any]:
        """Get a dependency or return default if not found."""
        try:
            return self.resolve(key)
        except ValueError:
            return default
    
    def has(self, key: str) -> bool:
        """Check if a dependency is registered."""
        return key in self._singletons or key in self._factories or key in self._transients
    
    def clear(self) -> None:
        """Clear all registered dependencies (useful for testing)."""
        self._singletons.clear()
        self._factories.clear()
        self._transients.clear()
        logger.debug("Container cleared")


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container
