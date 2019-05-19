'''
Created on Nov 5, 2017

@author: daniel
'''

from astropy.io import fits as pyfits
import numpy as np
from collections import OrderedDict

# function that gets match and prints elements structure of that file.

INDENT = '|--'


def elementinfo(value):
    etype = type(value)
    if isinstance(value, dict):
        return 'nkeys = {}'.format(len(value.keys()))
    elif isinstance(value, np.recarray):
        return 'recarray = {}'.format(len(value.dtype.names))
    elif isinstance(value, np.ndarray):
        return 'shape = {}'.format(value.shape)
    elif isinstance(value, pyfits.PrimaryHDU):
        return 'shape = {}'.format("()")
    elif isinstance(value, pyfits.BinTableHDU):
        return 'shape = {}'.format(value.thing)
    elif isinstance(value, pyfits.FITS_rec):
        return 'FITS_rec = {}'.format(len(value.dtype.names))
    else:
        return ''


def indentation(indent):
    i = indent - 1
    prefix = ""
    if i > 0:
        prefix = "|   " * i
    prefix += "|-- "
    return prefix


def elementtype(value, indent=0):
    if isinstance(value, dict):
        return dictelement(value, indent=indent)
    elif isinstance(value, np.recarray):
        return recarrayelement(value, indent=indent)
    elif isinstance(value, np.ndarray):
        return ndarrayelement(value, indent=indent)
    elif isinstance(value, pyfits.PrimaryHDU):
        return PrimaryHDUelement(value, indent=indent)
    elif isinstance(value, pyfits.BinTableHDU):
        return BinTableHDUelement(value, indent=indent)
    elif isinstance(value, pyfits.FITS_rec):
        return recarrayelement(value, indent=indent)
    else:
        # return ["{i} {v}".format(i=INDENT*indent, v=type(value).__name__)]
        return ["{i} {v}".format(i=indentation(indent), v=type(value).__name__)]


def dictelement(obj, indent):
    lines = []
    for k, v in obj.items():
        # lines += ["{i} {v} {info}".format(i=INDENT*indent, v=k, info=elementinfo(v))]
        lines += ["{i}{v} {info}".format(i=indentation(indent), v=k, info=elementinfo(v))]
        lines += elementtype(v, indent=indent+1)
    return lines


def recarrayelement(obj, indent):
    lines = []
    for name in obj.dtype.names:
        value = obj[name]
        lines += ["{i}{v} {info}".format(i=indentation(indent), v=name, info=elementinfo(value))]
        # lines += ["{i} {v} {info}".format(i=INDENT*indent, v=name, info=elementinfo(value))]
        if isinstance(value, np.ndarray):
            value = value[tuple([0] * value.ndim)]
        lines += elementtype(value, indent=indent+1)
    return lines


def ndarrayelement(obj, indent):
    index = [0] * obj.ndim
    firstelement = obj.item(*index)
    lines = []
    lines += ['{i}{info}'.format(i=indentation(indent), info=elementinfo(obj))]
    # lines += ['{i} {info}'.format(i=INDENT*indent, info=elementinfo(obj))]
    lines += elementtype(firstelement, indent=indent)
    return lines


def PrimaryHDUelement(obj, indent):
    '''
    https://pythonhosted.org/pyfits/api/hdus.html?highlight=primaryhdu#pyfits.PrimaryHDU
    '''
    return ""


def BinTableHDUelement(obj, indent):
    '''
    https://pythonhosted.org/pyfits/api/tables.html?highlight=bintablehdu#pyfits.BinTableHDU
    '''
    lines = []
    indents = indentation(indent)
    lines += ['{i} name: {name}'.format(i=indents, name=obj.name),
              '{i} header: {header}'.format(i=indents, header=obj.header)]
    try:
        lines += ['{i} uint: {uint}'.format(i=indents, uint=obj.uint)]
    except Exception:
        pass
    lines += elementtype(obj.data, indent=indent+1)
    return lines


class Node(object):
    def __init__(self):
        self.parent = None
        self.children = OrderedDict()

    def add_child(self, name, child):
        assert isinstance(child, Node), "child must be instance of Node"
        self.children[name] = child
        child.parent = self

    def isleaf(self):
        return len(self.children) == 0

    def isroot(self):
        return self.parent is None


class TypeNode(Node):

    CONNECTOR = '.'

    def __init__(self):
        super(TypeNode, self).__init__()
        self.path = ''

    def add_child(self, name, child):
        super(TypeNode, self).add_child(name, child)
        ppath = self.path
        connector = TypeNode.CONNECTOR if ppath else ''
        name = name if name else '.'
        child.set_path(ppath + connector + name)

    def set_path(self, path): 
        if path == self.path:
            return

        self.path = path
        connector = TypeNode.CONNECTOR if path else ''
        for name, child in self.children.items():
            child.set_path("{}{}{}".format(path, connector, name))   


class RecarrayTypeNode(TypeNode):
    def __init__(self, obj):
        super(RecarrayTypeNode, self).__init__()
        self.shape = obj.shape

        for name in obj.dtype.names:
            element = obj[name]
            node = elementtree(element)
            self.add_child(name, node)

    def __repr__(self, indent = 0):
        prefix = '|   ' * indent
        items = ["\n{}|-- {} {}".format(prefix, n, o.__repr__(indent + 1))
                 for n, o in self.children.items()]
        return ''.join(items)

    def repr_flat(self,): 
        # connector = TypeNode.CONNECTOR if self.path else ''
        items = ["\n{} {}".format(o.path, o.repr_flat())
                 for o in self.children.values()]
        return ''.join(items)


class DictTypeNode(TypeNode):
    def __init__(self, obj):
        super(DictTypeNode, self).__init__()

        for name, value in obj.items():
            self.add_child(name, elementtree(value))

    def __repr__(self, indent = 0):
        prefix = '|   ' * indent
        items = ["\n{}|-- {} {}".format(prefix, n, o.__repr__(indent + 1))
                 for n, o in self.children.items()]
        return ''.join(items)

    def repr_flat(self,): 
        items = ["\n{} {}".format(o.path, o.repr_flat())
                 for o in self.children.values()]
        return ''.join(items)


class NdarrayTypeNode(TypeNode):   
    def __init__(self, obj):
        super(NdarrayTypeNode, self).__init__()
        self.shape = obj.shape
        self.type = obj.dtype
        index = tuple([0] * len(obj.shape))
        item = obj[index]
        child = elementtree(item)
        if not isbasictype(child):
            self.add_child('', child) 

    def __repr__(self, indent = 0):
        if len(self.children) == 0:
            item = "{} {}".format(self.shape, self.type)
        else:
            child = list(self.children.values())[0]
            item = "{} {}".format(self.shape, child.__repr__(indent))
        return item

    def repr_flat(self, indent = 0):
        item = "{} {} {}".format(self.path, self.shape, self.type)
        return item


class PrimaryHDUTypeNode(TypeNode):   
    def __init__(self, obj):
        super(PrimaryHDUTypeNode, self).__init__()

    def __repr__(self, indent = 0):
        return ""

    def repr_flat(self,): 
        return ""


class BinTableHDUTypeNode(TypeNode):   
    def __init__(self, obj):
        super(BinTableHDUTypeNode, self).__init__()
        self.name = obj.name
        self.header = obj.header
        try:
            uint = obj.uint
        except Exception:
            pass

    def __repr__(self, indent = 0):
        prefix = '|   ' * indent
        items = ["\n{}|-- {} {}".format(prefix, n, o.__repr__(indent + 1))
                 for n, o in self.children.items()]
        return ''.join(items)

    def repr_flat(self,): 
        items = ["\n{} {}".format(o.path, o.repr_flat())
                 for o in self.children.values()]
        return ''.join(items)


class BasicTypeNode(TypeNode):   
    def __init__(self, obj):
        super(BasicTypeNode, self).__init__()
        # self.shape = obj.shape
        self.type = type(obj)

    def isbasictype(self):
        return True

    def __repr__(self, indent = 0):
        return self.type

    def repr_flat(self,): 
        item = "\n{} {}".format(self.path, self.type)
        return item


def isbasictype(node):
    try:
        return node.isbasictype()
    except Exception:
        return False


def elementtree(obj):
    if isinstance(obj, dict):
        return DictTypeNode(obj)
    elif isinstance(obj, np.recarray):
        return RecarrayTypeNode(obj)
    elif isinstance(obj, np.ndarray):
        return NdarrayTypeNode(obj)
    elif isinstance(obj, pyfits.PrimaryHDU):
        return PrimaryHDUTypeNode(obj)
    elif isinstance(obj, pyfits.BinTableHDU):
        return BinTableHDUTypeNode(obj)
    elif isinstance(obj, pyfits.FITS_rec):
        return RecarrayTypeNode(obj)
    else:  # basic type
        return BasicTypeNode(obj)
