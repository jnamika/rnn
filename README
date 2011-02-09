Recurrent neural network package for problems of time-series prediction and generation

Copyright (c) 2009-2011, Jun Namikawa <jnamika@gmail.com>
License: ISC license

This package implements a gradient-based learning algorithm for recurrent neural networks.
The package supports
(1) both fully connected and sparsely connected networks,
(2) both discrete-time neural networks and continuous-time neural networks,
(3) training examples of both symbolic data and floating point numbers,
(4) multi-threading, and
(5) analyzing some characteristics (ex: Lyapunov spectrum, Kullback-Leibler divergence).



=== Installation ===

First, type `./autogen.sh' in the current directory to create configure file.
Next, type `./configure' and when it finishes, type `make'. This will create `rnn-learn', `rnn-generate' and other utility programs.

Run them with the argument `-h' to show the usages of them.

If you wish to install the programs, type `make install'. By default, this will install all the files in `/usr/local/bin' or `/usr/local/lib'. You can change the install path with the `--prefix' option of the configure script, for instance `--prefix=$HOME' (use `./configure --help' for other options).



=== Requirements ===

Building this package requires a C compiler supporting C99 and Autotools (GNU Autoconf, Automake and Libtool).

In addition, utility scripts in the `src/python' directory require python version 2.5 or later (but not python-3.x). Gnuplot is also needed to run `rnn-plot-log' script.



=== Example ===

Here is a sample session.

> cd examples
> echo "import gen_target
gen_target.print_sin_curve(500, 20)" | python > target.txt
> rnn-learn -e 5000 target.txt
> rnn-generate -n 1000 rnn.dat


