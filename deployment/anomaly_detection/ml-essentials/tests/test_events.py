import unittest

from mock import Mock

from mltk.events import EventHost, Event


class EventTestCase(unittest.TestCase):

    def test_Event(self):
        f1 = Mock()
        f2 = Mock()
        events = EventHost()

        # test add event via getitem
        self.assertListEqual(list(events), [])
        self.assertNotIn('ev', events)

        ev = events['ev']
        self.assertIsInstance(ev, Event)
        self.assertEqual(repr(ev), 'Event(ev)')
        self.assertListEqual(list(events), ['ev'])
        self.assertIn('ev', events)

        # test add callback
        self.assertEqual(ev.do(f1), f1)
        self.assertEqual(ev.do(f2), f2)

        # test fire
        ev.fire(123, second=456)
        for f in [f1, f2]:
            self.assertEqual(f.call_count, 1)
            self.assertEqual(f.call_args, ((123,), {'second': 456}))
            f.reset_mock()

        events.fire('ev', 123, second=456)
        for f in [f1, f2]:
            self.assertEqual(f.call_count, 1)
            self.assertEqual(f.call_args, ((123,), {'second': 456}))
            f.reset_mock()

        # test remove callback
        ev.cancel_do(f1)

    def test_decorator(self):
        f = Mock()
        events = EventHost()
        ev = events['ev']

        # test call via `__call__`
        @ev.do
        def method(*args, **kwargs):
            return f(*args, **kwargs)

        ev.fire(123, second=456)
        self.assertEqual(f.call_count, 1)
        self.assertEqual(f.call_args, ((123,), {'second': 456}))

    def test_EventHost(self):
        f1 = Mock()
        f2 = Mock()
        f3 = Mock()

        events1 = EventHost()
        events2 = EventHost()

        self.assertListEqual(list(events1), [])
        self.assertNotIn('ev', events1)

        events1.on('ev', f1)
        events1.on('ev', f2)
        events2.on('ev', f3)

        self.assertListEqual(list(events1), ['ev'])
        self.assertIn('ev', events1)

        # test connect host
        events1.connect(events2)
        events1.fire('ev', 123, second=456)
        for f in [f1, f2, f3]:
            self.assertEqual(f.call_count, 1)
            self.assertEqual(f.call_args, ((123,), {'second': 456}))
            f.reset_mock()

        events2.fire('ev', 123, second=456)
        for f in [f1, f2]:
            self.assertEqual(f.call_count, 0)
        self.assertEqual(f3.call_count, 1)
        self.assertEqual(f3.call_args, ((123,), {'second': 456}))
        f3.reset_mock()

        # test disconnect host
        events1.disconnect(events2)
        events1.fire('ev', 123, second=456)
        for f in [f1, f2]:
            self.assertEqual(f.call_count, 1)
            self.assertEqual(f.call_args, ((123,), {'second': 456}))
            f.reset_mock()
        self.assertEqual(f3.call_count, 0)

        # test off a callback
        events1.off('ev', f1)
        events1.fire('ev', 123, second=456)
        self.assertEqual(f1.call_count, 0)
        self.assertEqual(f2.call_count, 1)
        self.assertEqual(f2.call_args, ((123,), {'second': 456}))

    def test_connect_object(self):
        class MyObject(object):
            watcher = []

            def __init__(self):
                self._events = EventHost()
                self.another_event = self._events['another_event']
                self.another_event.do(lambda *args, **kwargs:
                                      self.watcher.append((args, kwargs)))

            def updated(self, *args, **kwargs):
                self.watcher.append((args, kwargs))

        events = EventHost()
        obj = MyObject()
        events.connect(obj)

        # test another event object
        events.fire('another_event', 123, second=456)
        self.assertEqual(obj.watcher, [
            ((123,), {'second': 456})
        ])
        obj.watcher.clear()

        # test exist method
        events.fire('updated', 123, second=456)
        self.assertEqual(obj.watcher, [
            ((123,), {'second': 456})
        ])
        obj.watcher.clear()

        # test non-exist method
        events.fire('not_exist', 123, second=456)
        self.assertEqual(obj.watcher, [])

    def test_fire_order(self):
        v = []
        t = []

        def init():
            v[:] = [1]
            t[:] = []

        def cb1():
            t.append(v[-1])
            v[-1] += 1

        def cb2():
            t.append(-v[-1])
            v[-1] += 1

        def cb3():
            t.append(v[-1] * 10)
            v[-1] += 1

        def cb4():
            t.append(-v[-1] * 10)
            v[-1] += 1

        events1 = EventHost()
        events2 = EventHost()

        events1.on('ev', cb1)
        ev = events1['ev']
        ev.do(cb2)

        events2.on('ev', cb3)
        events2['ev'].do(cb4)
        events1.connect(events2)

        # test order of `EventHost.fire`
        init()
        events1.fire('ev')
        self.assertListEqual(t, [1, -2, 30, -40])

        # test order of `EventHost.reverse_fire`
        init()
        events1.reverse_fire('ev')
        self.assertListEqual(t, [-10, 20, -3, 4])

        # test order of `Event.fire`
        init()
        ev.fire()
        self.assertListEqual(t, [1, -2, 30, -40])

        # test order of `Event.reverse_fire`
        init()
        ev.reverse_fire()
        self.assertListEqual(t, [-10, 20, -3, 4])

    def test_no_side_effect(self):
        events = EventHost()
        events2 = EventHost()

        # disconnect a non-connected host will bring no side effect
        events.disconnect(events2)

        # fire a non-exist event will bring no side effect
        events.fire('not-exist')
        self.assertEqual(events._events, {})

        events.reverse_fire('not-exist')
        self.assertEqual(events._events, {})

        # un-register from a non-exist event will bring no side effect
        events.off('not-exist', print)
        self.assertEqual(events._events, {})

        # fire an event with no callback will bring no side effect
        events['empty-event'].fire()
        self.assertEqual(list(events._events.keys()), ['empty-event'])
        self.assertEqual(events['empty-event']._callbacks, [])

        # un-register from an empty event will bring no side effect
        events['empty-event'].cancel_do(print)
        self.assertEqual(events['empty-event']._callbacks, [])
