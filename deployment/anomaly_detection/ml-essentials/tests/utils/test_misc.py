import os
import re
import unittest
from tempfile import TemporaryDirectory
from threading import Thread

import pytest

from mltk.utils import *


class IterFilesTestCase(unittest.TestCase):

    def test_iter_files(self):
        names = ['a/1.txt', 'a/2.txt', 'a/b/1.txt', 'a/b/2.txt',
                 'b/1.txt', 'b/2.txt', 'c.txt']

        with TemporaryDirectory() as tempdir:
            for name in names:
                f_path = os.path.join(tempdir, name)
                f_dir = os.path.split(f_path)[0]
                os.makedirs(f_dir, exist_ok=True)
                with open(f_path, 'wb') as f:
                    f.write(b'')

            self.assertListEqual(names, sorted(iter_files(tempdir)))
            self.assertListEqual(names, sorted(iter_files(tempdir + '/a/../')))


class InheritanceDictTestCase(unittest.TestCase):

    def test_base(self):
        class GrandPa(object): pass
        class Parent(GrandPa): pass
        class Child(Parent): pass
        class Uncle(GrandPa): pass
        class NotExist(object): pass

        d = InheritanceDict()
        d[Child] = 1
        d[GrandPa] = 2
        d[Uncle] = 3

        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Uncle], 3)

        with pytest.raises(KeyError):
            _ = d[NotExist]

        d[GrandPa] = 22
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[Parent], 22)

        with pytest.raises(KeyError):
            _ = d[NotExist]

    def test_cached(self):
        class GrandPa(object): pass
        class Parent(GrandPa): pass
        class Child(Parent): pass
        class Uncle(GrandPa): pass
        class NotExist(object): pass

        d = CachedInheritanceDict()
        d[Child] = 1
        d[GrandPa] = 2
        d[Uncle] = 3

        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[GrandPa], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Parent], 2)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Child], 1)
        self.assertEqual(d[Uncle], 3)
        self.assertEqual(d[Uncle], 3)

        with pytest.raises(KeyError):
            _ = d[NotExist]
        with pytest.raises(KeyError):
            _ = d[NotExist]

        d[GrandPa] = 22
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[GrandPa], 22)
        self.assertEqual(d[Parent], 22)
        self.assertEqual(d[Parent], 22)

        with pytest.raises(KeyError):
            _ = d[NotExist]
        with pytest.raises(KeyError):
            _ = d[NotExist]


class DeepCopyTestCase(unittest.TestCase):

    def test_deep_copy(self):
        # test regex
        pattern = re.compile(r'xyz')
        self.assertIs(deep_copy(pattern), pattern)

        # test list of regex
        v = [pattern, pattern]
        o = deep_copy(v)
        self.assertIsNot(v, o)
        self.assertEqual(v, o)
        self.assertIs(v[0], o[0])
        self.assertIs(o[1], o[0])


class ContextStackTestCase(unittest.TestCase):

    def test_thread_local_and_initial_factory(self):
        stack: ContextStack[dict] = ContextStack(dict)
        thread_top = [None] * 10

        def thread_job(i):
            thread_top[i] = stack.top()

        threads = [
            Thread(target=thread_job, args=(i,))
            for i, _ in enumerate(thread_top)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        for i, top in enumerate(thread_top):
            for j, top2 in enumerate(thread_top):
                if i != j:
                    self.assertIsNot(top, top2)

    def test_push_and_pop(self):
        stack: ContextStack[object] = ContextStack()
        self.assertIsNone(stack.top())

        # push the first layer
        first_layer = object()
        stack.push(first_layer)
        self.assertIs(stack.top(), first_layer)

        # push the second layer
        second_layer = object()
        stack.push(second_layer)
        self.assertIs(stack.top(), second_layer)

        # pop the second layer
        stack.pop()
        self.assertIs(stack.top(), first_layer)

        # pop the first layer
        stack.pop()
        self.assertIsNone(stack.top())
