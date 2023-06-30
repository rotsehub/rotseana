====================
find_burst utilities
====================
:Author: Daniel Sela
:Contact: kehoe@physics.smu.edu, gdhungana@mail.smu.edu, danielsela42@gmail.com
:Acknowledgement: Robert Kehoe, Govinda Dhungana, Arnon Sela


**WARNING**
This is not a complete substitute of the analysis of Variable star search with ROTSE data.

.. contents:: **Table of Contents**
    :depth: 2

----------
Objective
----------

At the beginning of the process of finding variable stars, just getting the data from one star can take an hour. After finding the coordinates of a star, one must search through all the nights to find the objid's of the same star. Then, findburst_gd must be run in IDL to extract the stars data from each night. This package attempts to simplify this process.  Some of these programs are similar to the their IDL counterparts, while others are added to shorten this process. Together, these programs can shorten the time it takes to gather the data of a star from an hour to a minute or less.

--------------------------------------
Installing for product/development
--------------------------------------

Installing for development using bash
=====================================

1. Activate a virtual environment
2. Install the packages in REQUIRED
3. Go to the development directory: cd /path/to/dev_dir
4. Get the URL for the repository and run: git clone URL
5. Add the py folders to the python path: export PYTHONPATH=/path/to/rotseana/py:/path/to/rotseutil/py:/path/to/rotsedatamodel/py:$PYTHONPATH

Installing for product using bash
====================================

1. Download repository as ZIP through GitHub
2. Activate a virtual environment
3. In command line run: pip install /path/to/git-download

-------------------
Recommended Process
-------------------

1. Use getcoords to create a single file with the coordinates and filtered observations over all the nights.
2. After finding the coordinates of one star though the result of find_burst, use matchcoords_gd to gather the data over all the nights.
   The inputed file can be the output of getcoords.
3. Alternatively, if you do not wish to use getcoords, you can use findcoords_gd to gather the data over all the nights of a single star.
   Though findcoords_gd will take longer because it has to filter observations and search through the FIT or MATCH file for the coordinates and object id's.
   The output of getcoords can be used as the inputed file.

--------
Programs
--------

find_burst
==========

This program analyzes a match structure of lightcurves from ROTSE-I or ROTSE-III telescopes and extracts objects according to three types of quantity.  Selection by statistical quantities select varying objects in the field of view.  Selection by object ID # in the structure identifies specific objects of interest.  Alternatively, celestial coordinates and a radius can be used to select all objects in a portion of a match structure.  Find_burst provides the option to create a PDF file of all stars satisfying the selection applied.  The file types supported are .datc, .dat, and .fit.

Parameters
----------

-h         help for find_burst
--file, -f  File with match structure
--mindelta  Minimum delta
--minsig    Minimum sigma
--minchisq  Minimum chisq
--objid     Processes a specific objid
--refra     Reference RA (decimal deg.) for coordinate selection
--refdec    Reference Dec (decimal deg.) for coordinate slection
--radius    Radius of circle bounding coordinate selection
--log       Saves the output under <name provided>.pdf

To run
------

.. code::

   Variability selection:
    $ find_burst -f FILE --mindelta VALUE --minsig VALUE --minchisq VALUE  --log NAME

   Object selection:
    $ find_burst -f FILE --objid NUMBER --log NAME

   Coordinate selection:
    $ find_burst -f FILE --refra VALUE --refdec VALUE --radius VALUE --log NAME
    
Example
-------

.. code::

    $ find_burst -f 130904_vsp2218+4034_3b_match.fit --mindelta 0.1 --minsig 3.0 --minchisq 2.0 --log sky1

    $ find_burst -f 130904_vsp2218+4034_3b_match.fit --objid 4447 --log sky2

    $ find_burst -f 130904_vsp2218+4034_3b_match.fit --refra 335.04050 --refdec 41.34516 --radius 0.001 --log fb4
    
For more information run find_burst -h (or --help)

findburst_gd
============

Extracts the date, magnitude, and magnitude error of a star given an object idea from a file and saves it as a text file.

Parameters
----------

--match     Works on match structured files (.dat and .datc). Only use when the file you are retrieving the data from is a match structured file.
--fits      Works on FIT files. Only use when the file you are retrieving the data from is a FIT structured file.
--mindelta  Minimum delta
--minsig    Minimum sigma
--minchisq  Minimum chisq
--objid     Processes a specific objid
--log       Saves the output as a .txt.

To run using a match structured file
------------------------------------

.. code::

    Variability selection:  findburst_gd --match FILE --mindelta VALUE --minsig VALUE --minchisq VALUE  --log name.txt

    Object selection: find_burst
To run using fit structured file
--------------------------------

.. code::

    findburst_gd --fits FILE --mindelta VALUE --minsig VALUE --minchisq VALUE  --objid VALUE --log name.txt

Example
-------

.. code::

    findburst_gd --match 000409_xtetrans_1a_match.dat --mindelta 0.1 --minsig 1.0 --minchisq 2.0  --objid 115 --log name.txt

getcoords
=========

Extracts the coordinates from one or more files and prints them to the terminal. You can concatenate the output into a text file. getcoords filters out bad observations. Negative RA coordinates are skipped as part of the filtering process. If you are going to use getcoords, it is recommended that you run it over in advance over multiple directories. It takes time for it to run on multiple files due to the filter observation that would be executed per object on each file. Furthermore, it is recommended that you collaborate with other users on the output, since only one output needs to be created per directory.

Parameters
----------

-f, --file  Processes the specified file.

To run
------

.. code::

    getcoords -f FILE

Example
-------

.. code::

    getcoords -f 000409_xtetrans_1a_match.dat

matchcoords
===========

Extract object ids of similar coordinates in a file within a specified error. matchcoords works on a text coordinate file that can be produced by getcoords. When typing the coordinates, there should be no spaces, and a capital "J" in the beginning. matchcoords process is similar to findcoords, however, since it is working on previously generated good coordinates file, it is much faster. Therefore, it is recommended to generate coordinate files per directory in advance, and search through the files using matchcoords.

Parameters
----------

-e  error with a float value.

To run
------

.. code::

    matchoords -e ERRORVALUE -f FILE -c "COORDINATES"

Example
-------

.. code-block::

    getcoords -f 000409*_match.dat > all_coords.txt
    matchoords -e 10 -f all_coords.txt -c "J110526.404+501802.085"

matchcoords_gd
==============

Extract the date, magnitude, and magnitude error of a star given the coordinates and an error and output it into a text file. matchcoords_gd works on a text coordinate file that can be produced by getcoords.

Parameters
----------

-e       error with with a float value.
--w-ref  adds the objid and the name of the file from which the data was extracted to the text file.
-c       (--coord) coordinates with a string "". Do not use any spaces and use a capital "J" at the beginning.
-f       (--file) the file(s) that the data will be extracted from.

To run without reference
------------------------

.. code::

    matchcoords_gd -e ERRORVALUE --log NAME -c "COORDINATES" -f FILE

To run with reference
---------------------

.. code::

    matchcoords_gd -e ERRORVALUE --w-ref --log NAME -c "COORDINATES" -f FILE

Example
-------

.. code::

    matchcoords_gd -e 10 -—w-ref --log name_gd  -c “J111734.010+501526.228” -f ../000409_xtetrans_1a_match.dat ../*.fit

findcoords
==========

Extract object ids of similar coordinates in a file within a specified error.

Parameters
----------

-e  error with with a float value.
-c  (--coord) coordinates with a string "". Do not use any spaces and use a capital "J" at the beginning.
-f  (--file) the file(s) that the data will be extracted from.

To run
------

.. code::

    findcoords -e ERRORVALUE -c "COORDINATES" -f FILE

Example
-------

.. code::

    findcoords -e 10 -c "J110526.404+501802.085" -f 000409_xtetrans_1a_match.dat

findcoords_gd
=============

Extract the date, magnitude, and magnitude error of a star given the coordinates and an error. This program saves the output into a text file.

Parameters
----------

-e       error with with a float value.
--w-ref  adds the objid and the name of the file from which the data was extracted to the text file.
-c       (--coord) coordinates with a string "". Do not use any spaces and use a capital "J" at the beginning.
-f       (--file) the file(s) that the data will be extracted from.

To run without reference
------------------------

.. code::

    findcoords_gd -e ERRORVALUE --log NAME -c "COORDINATES" -f FILE

To run with reference
---------------------

.. code::

    findcoords_gd -e ERRORVALUE --w-ref --log NAME -c "COORDINATES" -f FILE

Example
-------

.. code::

    findcoords_gd -e 10  -—w-ref --log name_gd  -c “J111734.010+501526.228” -f ../000409_xtetrans_1a_match.dat ../*.fit
