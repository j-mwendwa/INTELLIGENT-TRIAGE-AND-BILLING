"""
tests/unit/test_memory.py — Unit tests for EntityMemory and ConversationSummaryMemory.
"""

from __future__ import annotations

from unittest.mock import patch


class TestEntityMemory:
    def test_remember_and_retrieve(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-001")
            mem.remember("customer_name", "Alice Smith")
            assert mem.all()["customer_name"] == "Alice Smith"

    def test_update_multiple_facts(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-002")
            mem.update({"plan": "Enterprise", "account_id": "ACC-999"})
            facts = mem.all()
            assert facts["plan"] == "Enterprise"
            assert facts["account_id"] == "ACC-999"

    def test_persists_to_disk(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-003")
            mem.remember("key", "value")

            # Create new instance for same thread — should load from disk
            mem2 = EntityMemory("test-thread-003")
            assert mem2.all()["key"] == "value"

    def test_clear_removes_facts(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-004")
            mem.remember("name", "Bob")
            mem.clear()
            assert mem.all() == {}

    def test_delete_removes_file(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-005")
            mem.remember("x", "y")
            file_path = tmp_path / "test-thread-005.json"
            assert file_path.exists()
            mem.delete()
            assert not file_path.exists()

    def test_contains_operator(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-006")
            mem.remember("region", "EU")
            assert "region" in mem
            assert "missing_key" not in mem

    def test_len_operator(self, tmp_path):
        with patch("src.memory.entity_memory.MEMORY_DIR", tmp_path):
            from src.memory.entity_memory import EntityMemory

            mem = EntityMemory("test-thread-007")
            assert len(mem) == 0
            mem.remember("a", "1")
            mem.remember("b", "2")
            assert len(mem) == 2


class TestContextAssembler:
    def test_assembles_all_sections(self):
        from src.core.context_assembler import ContextAssembler
        from src.core.token_counter import TokenCounter

        assembler = ContextAssembler(token_counter=TokenCounter())
        result = assembler.build(
            system_prompt="You are a helpful assistant.",
            entity_memory={"name": "Alice", "account": "ACC-001"},
            conversation_summary="Customer asked about billing.",
            retrieved_docs=["Doc 1: Refund policy...", "Doc 2: Invoice info..."],
            target_tokens=8000,
        )

        assert "You are a helpful assistant." in result
        assert "Alice" in result
        assert "Customer asked about billing." in result
        assert "Refund policy" in result

    def test_trims_docs_when_over_budget(self):
        from src.core.context_assembler import ContextAssembler
        from src.core.token_counter import TokenCounter

        assembler = ContextAssembler(token_counter=TokenCounter())
        many_docs = ["Long document content " * 200] * 20  # very long docs

        result = assembler.build(
            system_prompt="System.",
            entity_memory={},
            conversation_summary=None,
            retrieved_docs=many_docs,
            target_tokens=500,  # very tight budget
        )

        # Should still have system prompt
        assert "System." in result
