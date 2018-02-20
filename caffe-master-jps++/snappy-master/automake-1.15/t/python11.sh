#! /bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Test missing python.

# Python is not required for this test.
. test-init.sh

# We don't want to allow user overrides in this test.
unset PYTHON

cat >>configure.ac <<'EOF'
m4_define([_AM_PYTHON_INTERPRETER_LIST], [IShouldNotExist1 IShouldNotExist2])
AM_PATH_PYTHON
# The following be executed only after the first run, once a
# third argument has been added to the previous macro.
echo PYTHON = $PYTHON
test "$PYTHON" = : || exit 1
EOF

: > Makefile.am

$ACLOCAL
$AUTOCONF

./configure >stdout 2>stderr && { cat stdout; cat stderr >&2; exit 1; }
cat stdout
cat stderr >&2
grep 'checking for IShouldNotExist1' stdout
grep 'checking for IShouldNotExist2' stdout
grep 'no suitable Python interpreter found' stderr

sed 's/AM_PATH_PYTHON/AM_PATH_PYTHON(,,:)/' configure.ac >configure.tmp
mv -f configure.tmp configure.ac
$ACLOCAL --force
$AUTOCONF --force
# This one should define PYTHON as ":" and exit successfully.
./configure

# Any user setting should be used.
./configure PYTHON=foo >stdout && { cat stdout; exit 1; }
cat stdout
grep 'PYTHON = foo' stdout

:
