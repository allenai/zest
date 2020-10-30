"""Tests for zest.utils."""

import logging
import unittest

from zest import utils


class ConfigureLoggingTestCase(unittest.TestCase):
    """Test zest.utils.configure_logging."""

    def test_attaches_log_handler(self):
        n_handlers_before = len(logging.root.handlers)

        handler = utils.configure_logging()

        self.assertEqual(len(logging.root.handlers), n_handlers_before + 1)
        self.assertIn(handler, logging.root.handlers)

        logging.root.removeHandler(handler)

    def test_verbose_true_sets_log_level_to_debug(self):
        handler = utils.configure_logging(verbose=True)

        self.assertEqual(handler.level, logging.DEBUG)

        logging.root.removeHandler(handler)

    def test_verbose_false_sets_log_level_to_info(self):
        handler = utils.configure_logging(verbose=False)

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)

    def test_verbose_defaults_to_false(self):
        handler = utils.configure_logging()

        self.assertEqual(handler.level, logging.INFO)

        logging.root.removeHandler(handler)
