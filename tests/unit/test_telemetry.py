"""Unit tests for telemetry, context propagation, and logging."""

import asyncio
import json
import logging
import time
import unittest
from packages.telemetry.context import get_request_id, set_request_id
from packages.telemetry.logging import StructuredJsonFormatter
from packages.telemetry.timer import Timer


class TelemetryTests(unittest.TestCase):
    def test_request_id_context_propagation(self) -> None:
        set_request_id("req_main_thread")
        self.assertEqual(get_request_id(), "req_main_thread")

        async def subtask() -> str:
            return get_request_id()

        ret = asyncio.run(subtask())
        self.assertEqual(ret, "req_main_thread")

    def test_structured_json_formatter(self) -> None:
        set_request_id("req_format_test")
        formatter = StructuredJsonFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Processing image",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        self.assertEqual(parsed["level"], "INFO")
        self.assertEqual(parsed["message"], "Processing image")
        self.assertEqual(parsed["request_id"], "req_format_test")
        self.assertIn("timestamp", parsed)

    def test_timer_measurement(self) -> None:
        with Timer() as timer:
            time.sleep(0.015)
        self.assertGreaterEqual(timer.elapsed_ms, 10.0)


if __name__ == "__main__":
    unittest.main()
