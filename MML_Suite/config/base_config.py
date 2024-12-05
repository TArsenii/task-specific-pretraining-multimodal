import json
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Type, TypeVar

import yaml

T = TypeVar("T", bound="BaseConfig")


@dataclass(kw_only=True)
class BaseConfig:
    """Base configuration class with essential utility methods."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary, excluding None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create a config instance from a dictionary.

        Only includes keys that are defined in the class annotations.
        """
        valid_fields = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to attributes."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting of attributes."""
        setattr(self, key, value)

    @classmethod
    def from_yaml(cls: Type[T], path: str) -> T:
        """Create a config instance from a YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls: Type[T], path: str) -> T:
        """Create a config instance from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str) -> None:
        """Save the config to a YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    def to_json(self, path: str) -> None:
        """Save the config to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_tabular(self) -> str:
        raise NotImplementedError("Method to_tabular not implemented")

    def get(self, key: str, default: Any = None) -> Any:
        """Get an attribute value, returning a default if not found."""
        return getattr(self, key, default)

    @classmethod
    def get_field_names(cls) -> List[str]:
        """Get a list of all field names in the config."""
        return [f.name for f in fields(cls)]

    def __str__(self) -> str:
        """Return a formatted string representation of the config."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
