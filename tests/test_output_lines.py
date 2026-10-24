import importlib
import unittest
from unittest.mock import Mock, patch

gputil = importlib.import_module('GPUtil.GPUtil')


class OutputLineTests(unittest.TestCase):
    def test_csv_line_endings_and_final_row(self):
        rows = [
            '0, GPU-fictional-a, 10, 1000, 100, 900, synthetic, Card A, A, Disabled, Disabled, 30',
            '1, GPU-fictional-b, 20, 2000, 400, 1600, synthetic, Card B, B, Disabled, Disabled, 40',
        ]
        for separator in ('\n', '\r\n'):
            for trailing in ('', separator):
                with self.subTest(separator=separator, trailing=trailing):
                    process = Mock(returncode=0)
                    process.communicate.return_value = ((separator.join(rows) + trailing).encode(), None)
                    with patch.object(gputil.platform, 'system', return_value='Linux'), patch.object(gputil, 'Popen', return_value=process):
                        gpus = gputil.getGPUs()
                    self.assertEqual([gpu.id for gpu in gpus], [0, 1])
                    self.assertEqual([gpu.uuid for gpu in gpus], ['GPU-fictional-a', 'GPU-fictional-b'])
                    self.assertEqual([gpu.load for gpu in gpus], [0.1, 0.2])

    def test_empty_output(self):
        process = Mock(returncode=0)
        process.communicate.return_value = (b'', None)
        with patch.object(gputil.platform, 'system', return_value='Linux'), patch.object(gputil, 'Popen', return_value=process):
            self.assertEqual(gputil.getGPUs(), [])
