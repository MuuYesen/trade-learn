"""Neutral parameter inheritance shared by strategy components."""
from typing import Any

class Params:
    """Backtrader-style parameter storage object."""

    def __init__(self, defaults: Any = (), **kwargs):
        self._keys: list[str] = []
        if isinstance(defaults, dict):
            for name, val in defaults.items():
                if name not in self._keys:
                    self._keys.append(name)
                setattr(self, name, val)
        elif isinstance(defaults, (list, tuple)):
            for name, val in defaults:
                if isinstance(name, (list, tuple)):  # tuple of (name, val)
                    if name[0] not in self._keys:
                        self._keys.append(name[0])
                    setattr(self, name[0], name[1])
                else:
                    if name not in self._keys:
                        self._keys.append(name)
                    setattr(self, name, val)
        for name, val in kwargs.items():
            if name not in self._keys:
                self._keys.append(name)
            setattr(self, name, val)

    def __setattr__(self, name: str, value: Any) -> None:
        if not name.startswith("_") and hasattr(self, "_keys") and name not in self._keys:
            self._keys.append(name)
        super().__setattr__(name, value)

    def __getitem__(self, index: int) -> Any:
        return getattr(self, self._keys[index])

    def asdict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self._keys}


def collect_param_defaults(cls: type) -> tuple[list[str], list[tuple[str, Any]]]:
    names: list[str] = []
    defaults: list[tuple[str, Any]] = []
    for base_cls in reversed(cls.mro()):
        param_defaults = getattr(base_cls, "params", [])
        if isinstance(param_defaults, dict):
            names.extend(param_defaults.keys())
            defaults.extend(param_defaults.items())
        elif isinstance(param_defaults, (list, tuple)):
            for item in param_defaults:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    names.append(item[0])
                    defaults.append((item[0], item[1]))
    return names, defaults


def split_param_kwargs(cls: type, kwargs: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    param_names, _defaults = collect_param_defaults(cls)
    param_name_set = set(param_names)
    param_kwargs = {}
    other_kwargs = {}
    for key, value in kwargs.items():
        if key in param_name_set:
            param_kwargs[key] = value
        else:
            other_kwargs[key] = value
    return param_kwargs, other_kwargs
