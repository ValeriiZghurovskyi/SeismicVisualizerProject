"""Tests for domain.entities — Entity dataclass and EntityRegistry."""

import pytest

from seismic_visualizer.domain.entities import Entity, EntityKind, EntityRegistry
from seismic_visualizer.domain.exceptions import (
    EntityLimitReachedError,
    EntityNotFoundError,
)


# ---------------------------------------------------------------------------
# EntityKind enum
# ---------------------------------------------------------------------------


class TestEntityKind:
    def test_horizon_value(self) -> None:
        assert EntityKind.HORIZON.value == "horizon"

    def test_fault_value(self) -> None:
        assert EntityKind.FAULT.value == "fault"

    def test_distinct_values(self) -> None:
        assert EntityKind.HORIZON != EntityKind.FAULT


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------


class TestEntity:
    def test_fields_stored(self) -> None:
        e = Entity(id=1, name="Horizon 1", kind=EntityKind.HORIZON)
        assert e.id == 1
        assert e.name == "Horizon 1"
        assert e.kind == EntityKind.HORIZON
        assert e.visible is True

    def test_kind_fault(self) -> None:
        e = Entity(id=2, name="F-1", kind=EntityKind.FAULT)
        assert e.kind == EntityKind.FAULT

    def test_visible_default_true(self) -> None:
        e = Entity(id=5, name="X", kind=EntityKind.HORIZON)
        assert e.visible is True

    def test_visible_can_be_set_false(self) -> None:
        e = Entity(id=2, name="Y", kind=EntityKind.HORIZON, visible=False)
        assert e.visible is False

    def test_visible_is_mutable(self) -> None:
        e = Entity(id=3, name="Z", kind=EntityKind.HORIZON)
        e.visible = False
        assert e.visible is False


# ---------------------------------------------------------------------------
# EntityRegistry — creation
# ---------------------------------------------------------------------------


class TestCreateNew:
    def test_first_entity_gets_id_1(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        assert e.id == 1

    def test_second_entity_gets_id_2(self) -> None:
        reg = EntityRegistry()
        reg.create_new(kind=EntityKind.HORIZON)
        e = reg.create_new(kind=EntityKind.HORIZON)
        assert e.id == 2

    def test_auto_name_includes_id(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        assert "1" in e.name

    def test_custom_name_preserved(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON, name="Top Cretaceous")
        assert e.name == "Top Cretaceous"

    def test_entity_stored_in_registry(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        assert reg.get(e.id) is e

    def test_created_entity_visible_by_default(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        assert e.visible is True

    def test_created_entity_has_kind(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.FAULT)
        assert e.kind == EntityKind.FAULT

    def test_ids_fill_gaps(self) -> None:
        reg = EntityRegistry()
        e1 = reg.create_new(kind=EntityKind.HORIZON)
        e2 = reg.create_new(kind=EntityKind.HORIZON)
        reg.remove(e1.id)
        e3 = reg.create_new(kind=EntityKind.HORIZON)
        assert e3.id == e1.id  # gap at 1 is reused
        assert e2.id == 2

    def test_len_increases_on_create(self) -> None:
        reg = EntityRegistry()
        assert len(reg) == 0
        reg.create_new(kind=EntityKind.HORIZON)
        assert len(reg) == 1
        reg.create_new(kind=EntityKind.HORIZON)
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# EntityRegistry — limit
# ---------------------------------------------------------------------------


class TestEntityLimit:
    def test_max_id_constant(self) -> None:
        assert EntityRegistry.MAX_ID == 200

    def test_create_exactly_max_succeeds(self) -> None:
        reg = EntityRegistry()
        for _ in range(EntityRegistry.MAX_ID):
            reg.create_new(kind=EntityKind.HORIZON)
        assert len(reg) == EntityRegistry.MAX_ID

    def test_exceed_limit_raises(self) -> None:
        reg = EntityRegistry()
        for _ in range(EntityRegistry.MAX_ID):
            reg.create_new(kind=EntityKind.HORIZON)
        with pytest.raises(EntityLimitReachedError):
            reg.create_new(kind=EntityKind.HORIZON)

    def test_limit_error_is_seismic_error(self) -> None:
        from seismic_visualizer.domain.exceptions import SeismicError

        assert issubclass(EntityLimitReachedError, SeismicError)


# ---------------------------------------------------------------------------
# EntityRegistry — get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_existing(self) -> None:
        reg = EntityRegistry()
        created = reg.create_new(kind=EntityKind.HORIZON, name="A")
        fetched = reg.get(created.id)
        assert fetched is created

    def test_get_missing_raises(self) -> None:
        reg = EntityRegistry()
        with pytest.raises(EntityNotFoundError):
            reg.get(99)

    def test_get_after_remove_raises(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        reg.remove(e.id)
        with pytest.raises(EntityNotFoundError):
            reg.get(e.id)


# ---------------------------------------------------------------------------
# EntityRegistry — remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        reg.remove(e.id)
        assert len(reg) == 0

    def test_remove_missing_raises(self) -> None:
        reg = EntityRegistry()
        with pytest.raises(EntityNotFoundError):
            reg.remove(42)

    def test_remove_does_not_affect_others(self) -> None:
        reg = EntityRegistry()
        e1 = reg.create_new(kind=EntityKind.HORIZON)
        e2 = reg.create_new(kind=EntityKind.HORIZON)
        reg.remove(e1.id)
        assert reg.get(e2.id) is e2


# ---------------------------------------------------------------------------
# EntityRegistry — set_visibility
# ---------------------------------------------------------------------------


class TestSetVisibility:
    def test_hide_entity(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        reg.set_visibility(e.id, False)
        assert reg.get(e.id).visible is False

    def test_show_entity(self) -> None:
        reg = EntityRegistry()
        e = reg.create_new(kind=EntityKind.HORIZON)
        reg.set_visibility(e.id, False)
        reg.set_visibility(e.id, True)
        assert reg.get(e.id).visible is True

    def test_set_visibility_missing_raises(self) -> None:
        reg = EntityRegistry()
        with pytest.raises(EntityNotFoundError):
            reg.set_visibility(7, True)


# ---------------------------------------------------------------------------
# EntityRegistry — all()
# ---------------------------------------------------------------------------


class TestAll:
    def test_empty_registry_returns_empty_list(self) -> None:
        reg = EntityRegistry()
        assert reg.all() == []

    def test_all_returns_all_entities(self) -> None:
        reg = EntityRegistry()
        e1 = reg.create_new(kind=EntityKind.HORIZON)
        e2 = reg.create_new(kind=EntityKind.HORIZON)
        result = reg.all()
        assert len(result) == 2
        assert e1 in result
        assert e2 in result

    def test_all_sorted_by_id_ascending(self) -> None:
        reg = EntityRegistry()
        reg.create_new(kind=EntityKind.HORIZON)
        reg.create_new(kind=EntityKind.HORIZON)
        reg.create_new(kind=EntityKind.HORIZON)
        ids = [e.id for e in reg.all()]
        assert ids == sorted(ids)

    def test_all_sorted_after_gap_fill(self) -> None:
        reg = EntityRegistry()
        e1 = reg.create_new(kind=EntityKind.HORIZON)
        e2 = reg.create_new(kind=EntityKind.HORIZON)
        e3 = reg.create_new(kind=EntityKind.HORIZON)
        reg.remove(e2.id)
        e4 = reg.create_new(kind=EntityKind.HORIZON)  # fills gap at 2
        ids = [e.id for e in reg.all()]
        assert ids == sorted(ids)
        assert set(ids) == {e1.id, e3.id, e4.id}
