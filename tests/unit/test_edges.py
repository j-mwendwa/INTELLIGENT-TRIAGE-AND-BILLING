"""
tests/unit/test_edges.py — Unit tests for LangGraph edge routing functions.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from src.graph.edges import (
    billing_should_continue,
    route_after_input_check,
    route_after_tools,
    route_supervisor,
)


class TestInputCheckRouting:
    def test_routes_pass_to_supervisor(self, clean_state):
        state = {**clean_state, "input_security": {"decision": "PASS", "reason": "OK"}}
        assert route_after_input_check(state) == "supervisor"

    def test_routes_block_to_rejection(self, clean_state):
        state = {**clean_state, "input_security": {"decision": "BLOCK", "reason": "Injection"}}
        assert route_after_input_check(state) == "rejection"

    def test_routes_missing_security_to_supervisor(self, clean_state):
        # No input_security set — should pass through to supervisor
        state = {**clean_state, "input_security": None}
        assert route_after_input_check(state) == "supervisor"


class TestSupervisorRouting:
    @pytest.mark.parametrize(
        "intent,expected",
        [
            ("billing", "billing_agent"),
            ("technical", "technical_agent"),
            ("compliance", "compliance_agent"),
            ("general", "general_agent"),
            ("escalate", "escalation"),
            ("unknown", "general_agent"),
        ],
    )
    def test_routes_by_intent(self, clean_state, intent, expected):
        state = {**clean_state, "intent": intent}
        assert route_supervisor(state) == expected


class TestShouldContinue:
    def test_continues_when_tool_calls_present(self, clean_state):
        msg = AIMessage(
            content="",
            tool_calls=[{"name": "billing_search", "args": {"question": "invoice"}, "id": "123"}],
        )
        state = {**clean_state, "intent": "billing", "messages": [msg], "iteration": 1}
        assert billing_should_continue(state) == "tools"

    def test_extracts_answer_when_no_tool_calls(self, clean_state):
        msg = AIMessage(content="Here is your invoice breakdown...")
        state = {**clean_state, "intent": "billing", "messages": [msg], "iteration": 1}
        assert billing_should_continue(state) == "extract_answer"

    def test_extracts_answer_when_iteration_limit_reached(self, clean_state):
        msg = AIMessage(
            content="", tool_calls=[{"name": "billing_search", "args": {}, "id": "456"}]
        )
        state = {**clean_state, "messages": [msg], "iteration": 8, "max_iterations": 8}
        assert billing_should_continue(state) == "extract_answer"

    def test_extracts_answer_when_no_messages(self, clean_state):
        assert billing_should_continue(clean_state) == "extract_answer"


class TestRouteAfterTools:
    @pytest.mark.parametrize(
        "intent,expected",
        [
            ("billing", "billing_agent"),
            ("technical", "technical_agent"),
            ("compliance", "compliance_agent"),
            ("general", "general_agent"),
            ("unknown", "general_agent"),
        ],
    )
    def test_returns_to_correct_agent(self, clean_state, intent, expected):
        state = {**clean_state, "intent": intent}
        assert route_after_tools(state) == expected
