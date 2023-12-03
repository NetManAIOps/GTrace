import re
import time
import unittest
from dataclasses import dataclass

import pytest

from mltk.parsing import ProgramOutputReceiver, ProgramOutputParser, ProgramInfo
from mltk.typing_ import *


class ProgramOutputReceiverTestCase(unittest.TestCase):

    def test_line_breaks(self):
        class MyParser(ProgramOutputParser):
            def parse_line(self, line: bytes):
                logs.append(line)

        logs = []

        ##############################
        # test receiver with timeout #
        ##############################
        receiver = ProgramOutputReceiver([MyParser()],
                                         max_line_length=64,
                                         read_line_timeout=0.2)

        # test getting EOF with nothing
        logs.clear()
        receiver.start()
        receiver.stop()
        self.assertListEqual(logs, [])

        # test ordinary routines with timeout
        receiver.start()
        with pytest.raises(RuntimeError,
                           match='Background worker has already started'):
            receiver.start()

        try:
            # test split line at line breaks
            receiver.put_output(b'12')
            receiver.put_output(b'34')
            receiver.put_output(b'5\n')
            receiver.put_output(b'\r')
            receiver.put_output(b'67\b\b\b\b89\n\r\n1011\r')
            receiver.put_output(b'12\r13')
            receiver.put_output(b'\r')
            time.sleep(0.05)

            self.assertListEqual(logs, [
                b'12345', b'67', b'89', b'1011', b'12', b'13',
            ])

            # test too long line
            logs.clear()
            long_line = b''.join([str(i).encode('utf-8') for i in range(64)])
            long_line = long_line[:65]
            receiver.put_output(long_line)
            time.sleep(0.05)
            self.assertListEqual(logs, [long_line])

            # test read timeout
            logs.clear()
            receiver.put_output(b'no ')
            receiver.put_output(b'break')
            time.sleep(0.05)
            self.assertListEqual(logs, [])
            time.sleep(0.2)
            self.assertListEqual(logs, [b'no break'])

        finally:
            receiver.stop()
            receiver.stop()  # ensure double stop will not cause error

        # test getting EOF during waiting for the line break
        logs.clear()
        receiver.start()
        try:
            receiver.put_output(b'final')
        finally:
            receiver.stop()
        self.assertListEqual(logs, [b'final'])

        #################################
        # test receiver without timeout #
        #################################
        receiver = ProgramOutputReceiver([MyParser()], read_line_timeout=None)

        # test getting EOF with nothing
        logs.clear()
        receiver.start()
        receiver.stop()
        self.assertListEqual(logs, [])

        # test ordinary routines
        logs.clear()
        receiver.start()
        try:
            receiver.put_output(b'a\bc\n')
            time.sleep(0.05)
            self.assertListEqual(logs, [b'a', b'c'])
            logs.clear()
            receiver.put_output(b'final')
        finally:
            receiver.stop()
        self.assertListEqual(logs, [b'final'])

    def test_on_program_info(self):
        @dataclass
        class MyInfo(ProgramInfo):
            tag: str
            value: bytes

        class MyParser(ProgramOutputParser):

            def __init__(self, tag: str, pattern: PatternType):
                self.tag = tag
                self.pattern = pattern

            def parse_line(self, line: bytes):
                for v in self.pattern.findall(line):
                    yield MyInfo(tag=self.tag, value=v)

        class ErrorParser(ProgramOutputParser):

            def parse_line(self, line: bytes):
                raise RuntimeError()

        # test successful parsing
        receiver = ProgramOutputReceiver([
            ErrorParser(),  # this error should be ignored
            MyParser('A', re.compile(rb'1.3')),
            MyParser('B', re.compile(rb'a.c|1.3')),
        ])
        logs = []
        receiver.parsers[1].max_line_length = 20
        receiver.parsers[2].first_n_bytes_only = 52
        receiver.on_program_info.do(logs.append)

        def my_error_handler():
            raise RuntimeError()  # should be ignored
        receiver.on_program_info.do(my_error_handler)

        receiver.start()
        try:
            receiver.put_output(b'  103  aac  \n')
            receiver.put_output(b'  113  123  abc  acc  \n')  # ignored by parsers[1] due to max_line_length
            receiver.put_output(b'  adc  \n')
            receiver.put_output(b'nothing\n')
            receiver.put_output(b'aec\n')  # ignored by parsers[2] due to `first_n_bytes_only`
            time.sleep(0.05)
            self.assertEqual(logs, [
                MyInfo('A', b'103'),
                MyInfo('B', b'113'),
                MyInfo('B', b'123'),
                MyInfo('B', b'abc'),
                MyInfo('B', b'acc'),
                MyInfo('B', b'adc'),
            ])

        finally:
            receiver.stop()
