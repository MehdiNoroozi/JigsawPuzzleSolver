#! /bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Test parallel automake execution.

# This tests:
# 3) normal automake output should be identical and ordered in the same way
#    with --add-missing, even with concurrent file requirements, and the
#    installation of aux files should be race-free.

required=perl-threads
. test-init.sh

cat > configure.ac << 'END'
AC_INIT([parallel-am], [1.0])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE
AC_PROG_CC
AM_PATH_LISPDIR
AM_PATH_PYTHON
AC_CONFIG_FILES([Makefile])
END

cat > Makefile.am << 'END'
SUBDIRS =
END

list='1 2 3'
for i in $list; do
  echo "AC_CONFIG_FILES([sub$i/Makefile])" >> configure.ac
  echo "SUBDIRS += sub$i" >> Makefile.am
  mkdir sub$i
  unindent > sub$i/Makefile.am <<END
    python_PYTHON = foo$i.py
    lisp_LISP = foo$i.el
    bin_PROGRAMS = p$i
END
done

rm -f install-sh missing depcomp
mkdir build-aux

$ACLOCAL

# Generate expected output using the non-threaded code.
unset AUTOMAKE_JOBS
AUTOMAKE_run --add-missing
mv stderr expected
mv Makefile.in Makefile.in.exp

AUTOMAKE_JOBS=3
export AUTOMAKE_JOBS

for run in 1 2 3 4 5 6 7; do
  rm -f build-aux/* sub*/Makefile.in
  AUTOMAKE_run --add-missing
  diff stderr expected
  diff Makefile.in Makefile.in.exp
done

:
