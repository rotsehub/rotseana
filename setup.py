import sys
import os
from setuptools import setup, find_packages
import importlib.util


'''
is the Python package in your project. It's the top-level folder containing the
__init__.py module that should be in the same directory as your setup.py file


To create package and upload:

  # Registration is obsolete
  # python setup.py register
  python setup.py sdist
  twine upload -s dist/path/to/gz

'''

def get_authors(filename='AUTHORS'):
    ''' reads AUTHORS file.

    returns:
        two lists of authors and author's email.
    '''
    authors_file = os.path.join(os.getcwd(), filename)
    authors = []
    authors_email = []
    if os.path.isfile(authors_file):
        with open(authors_file, 'r') as f:
            content = f.read()
        for line in content.split('\n'):
            name, _, email = line.rpartition(' ')
            if name != '':
                authors.append(name)
                authors_email.append(email)
    return ', '.join(authors), ', '.join(authors_email)

def get_version(filename='VERSION'):
    version_file = os.path.join(os.getcwd(), filename)
    version = None
    if os.path.isfile(version_file):
        with open(version_file, 'r') as f:
            content = f.read()
        version = content.strip()
    return version

def scripts(bin_path = 'bin'):
    ''' pull all scripts from package/bin.This is not limited to
    executables, as some scripts are sources.
    '''
    bindir = os.path.join(os.getcwd(), bin_path)
    scripts = []
    if os.path.isdir(bindir):
        for file in os.listdir(bindir):
            file = os.path.join(bindir, file)
            if os.path.isfile(file) and not file.endswith('__init__.py'):
                scripts += [file]
    return scripts

def import_setup_utils():
    # load setup utils
    path = os.path.abspath(os.path.dirname(__file__))
    location = os.path.join(path, "setup_utils.py")
    try:
        setup_utils_spec = \
            importlib.util.spec_from_file_location("setup.utils", location)
        setup_utils = importlib.util.module_from_spec(setup_utils_spec)
        setup_utils_spec.loader.exec_module(setup_utils)
    except Exception as err:
        raise RuntimeError("Failed to find setup_utils.py."
                           " Please copy or link.") from err
    return setup_utils


setup_utils = import_setup_utils()
PACKAGES = find_packages('py')
PACKAGE_DIR = {'':'py'}
DESCRIPTION = ('matchutils is a set of utilities '
               'for converting MATCH structures into FITS')


AUTHORS, AUTHOR_EMAILS = get_authors()
URL = 'N/A'
VERSION = get_version()

# Find previous installation and warn. See #18115.
existing_path = []
if "install" in sys.argv:
    for p in PACKAGES:
        existing_path += setup_utils.existing_package(p)
    if existing_path:
        existing_path = ', '.join(existing_path)

scripts = scripts()

# Find all sub packages
packages = setup_utils.packages(PACKAGE)
required = setup_utils.read_required(metahost=metahost)

setup_info = {
    'name': NAME,
    'version': VERSION,
    'url': URL,
    'author': AUTHORS,
    'author_email': AUTHOR_EMAILS,
    'description': DESCRIPTION,
    'long_description': open("README.rst", "r").read(),
    'license': 'MIT',
    'keywords': ('library sequence logger yield singleton thread synchronize'
                 ' resource pool utilities os ssh xml excel mail'),
    'packages': packages,
    'scripts': scripts,
    'install_requires': required,
    'extras_require': {'dev': [], 'test': []},
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Environment :: Other Environment',
        # 'Framework :: Project Settings and Operation',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Software Development :: Libraries :: Application '
        'Frameworks',
        'Topic :: Software Development :: Libraries :: Python '
        'Modules']}

setup(**setup_info)


if existing_path:
    sys.stderr.write("""

========
WARNING!
========

You have just installed %(name)s over top of an existing
installation, without removing it first. Because of this,
your install may now include extraneous files from a
previous version that have since been removed from
Accord. This is known to cause a variety of problems. You
should manually remove the

%(existing_path)s

directory and re-install %(name)s.

""" % {"existing_path": existing_path,
       "name": NAME, })
