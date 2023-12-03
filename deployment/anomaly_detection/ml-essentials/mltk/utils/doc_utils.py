__all__ = ['DocInherit']


class DocInherit(type):
    """
    Meta-class for automatically inherit docstrings from base classes.

    Usage::

        class Parent(object, metaclass=DocInherit):

            \"""Docstring of the parent class.\"""

            def some_method(self):
                \"""Docstring of the method.\"""
                ...

        class Child(Parent):
            # inherits the docstring of :meth:`Parent`

            def some_method(self):
                # inherits the docstring of :meth:`Parent.some_method`
                ...
    """

    def __new__(kclass, name, bases, dct):
        def iter_mro():
            for base in bases:
                for mro in base.__mro__:
                    yield mro

        # inherit the class docstring
        if not dct.get('__doc__', None):
            for cls in iter_mro():
                cls_doc = getattr(cls, '__doc__', None)
                if cls_doc:
                    dct['__doc__'] = cls_doc
                    break

        # inherit the method docstrings
        for key in dct:
            attr = dct[key]
            if attr is not None and not getattr(attr, '__doc__', None):
                for cls in iter_mro():
                    cls_attr = getattr(cls, key, None)
                    if cls_attr:
                        cls_doc = getattr(cls_attr, '__doc__', None)
                        if cls_doc:
                            attr.__doc__ = cls_doc
                            break

        return super(DocInherit, kclass). \
            __new__(kclass, name, bases, dct)
