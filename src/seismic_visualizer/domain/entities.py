"""Entity and EntityRegistry — horizon/fault identity and visibility management.

See CLAUDE.md §4.3.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from seismic_visualizer.domain.exceptions import EntityLimitReachedError, EntityNotFoundError


class EntityKind(enum.Enum):
    """Geological entity type — determines which label cube and labeler to use."""

    HORIZON = "horizon"
    FAULT = "fault"


@dataclass
class Entity:
    """A single labeled geological entity (horizon or fault).

    Args:
        id: Unique integer identifier in range [1, EntityRegistry.MAX_ID].
        name: Human-readable label (e.g. "Horizon 1").
        kind: Whether this entity is a horizon or a fault.
        visible: Whether this entity is rendered in the views.
    """

    id: int
    name: str
    kind: EntityKind
    visible: bool = field(default=True)


class EntityRegistry:
    """Tracks a collection of entities and their display state.

    Each registry holds either horizons or faults — never both.
    Entity IDs are assigned automatically from the lowest free slot in [1, MAX_ID].

    Raises:
        EntityLimitReachedError: When MAX_ID entities already exist and a new one
            is requested.
        EntityNotFoundError: When get/remove/set_visibility is called with an ID
            that does not exist in this registry.
    """

    MAX_ID: int = 200

    def __init__(self) -> None:
        self._entities: dict[int, Entity] = {}


    def all(self) -> list[Entity]:
        """Return all entities sorted by ID (ascending)."""
        return sorted(self._entities.values(), key=lambda e: e.id)

    def get(self, entity_id: int) -> Entity:
        """Return the entity with the given ID.

        Raises:
            EntityNotFoundError: If no entity with that ID exists.
        """
        try:
            return self._entities[entity_id]
        except KeyError:
            raise EntityNotFoundError(f"Entity {entity_id} not found") from None

    def __len__(self) -> int:
        return len(self._entities)


    def create_new(self, kind: EntityKind, name: str | None = None) -> Entity:
        """Create and register a new entity with the next free ID.

        Args:
            kind: Whether this entity is a horizon or a fault.
            name: Optional display name; defaults to "Entity <id>" if omitted.

        Returns:
            The newly created Entity.

        Raises:
            EntityLimitReachedError: If MAX_ID entities already exist.
        """
        next_id = self._next_free_id()
        entity = Entity(id=next_id, name=name or f"{kind.value.capitalize()} {next_id}", kind=kind)
        self._entities[next_id] = entity
        return entity

    def restore(self, entity_id: int, kind: EntityKind, name: str | None = None) -> Entity:
        """Register an entity with a pre-existing ID (used when loading from disk).

        Args:
            entity_id: The ID to use (must be in [1, MAX_ID]).
            kind: Horizon or fault.
            name: Display name; defaults to "Entity <id>" if omitted.

        Returns:
            The restored Entity.
        """
        entity = Entity(id=entity_id, name=name or f"Entity {entity_id}", kind=kind)
        self._entities[entity_id] = entity
        return entity

    def remove(self, entity_id: int) -> None:
        """Remove the entity with the given ID.

        Raises:
            EntityNotFoundError: If no entity with that ID exists.
        """
        if entity_id not in self._entities:
            raise EntityNotFoundError(f"Entity {entity_id} not found")
        del self._entities[entity_id]

    def rename(self, entity_id: int, name: str) -> None:
        """Rename an entity.

        Raises:
            EntityNotFoundError: If no entity with that ID exists.
        """
        self.get(entity_id).name = name

    def set_visibility(self, entity_id: int, visible: bool) -> None:
        """Toggle the visibility flag of an entity.

        Raises:
            EntityNotFoundError: If no entity with that ID exists.
        """
        self.get(entity_id).visible = visible


    def _next_free_id(self) -> int:
        for candidate in range(1, self.MAX_ID + 1):
            if candidate not in self._entities:
                return candidate
        raise EntityLimitReachedError(
            f"Maximum of {self.MAX_ID} entities reached"
        )
