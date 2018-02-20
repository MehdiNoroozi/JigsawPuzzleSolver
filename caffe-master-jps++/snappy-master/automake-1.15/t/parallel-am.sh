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

# There are several hypotheses to be tested:  Independently of the number
# of threads used by automake,
# 0) the generated Makefile.in files must be identical without --add-missing,
# 1) the Makefile.in that distributes auxiliary files must be generated
#    after all other ones, so all installed aux files are caught,
# 2) normal automake output should have identical content and be ordered
#    in the same way, when --add-missing is not passed, or when
#    --add-missing is passed but there are no concurrent file requirements
#    (i.e., two Makefile.am files call for the same needed aux file)
# 3) normal automake output should be identical and ordered in the same way
#    with --add-missing, even with concurrent file requirements, and the
#    installation of aux files should be race-free,
# 4) warning and normal error output should be identical, in that duplicate
#    warnings should be omitted in the same way as without threads,
# 5) fatal error and debug messages could be identical.  This is not
#    intended, though.
#
# This test checks (0), (1), and (2).  See sister tests for further coverage.

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

list='1 2 3 4 5 6 7 8 9'
for i in $list; do
  echo "AC_CONFIG_FILES([sub$i/Makefile])" >> configure.ac
  echo "SUBDIRS += sub$i" >> Makefile.am
  mkdir sub$i
  echo > sub$i/Makefile.am
done
# Use an include chain to cause a nontrivial location object to be
# serialized through a thread queue.
echo 'include foo.am' >> sub7/Makefile.am
echo 'include bar.am' > sub7/foo.am
echo 'python_PYTHON = foo.py' > sub7/bar.am
echo 'lisp_LISP = foo.el' >> sub8/Makefile.am
echo 'bin_PROGRAMS = p' >> sub9/Makefile.am

rm -f install-sh missing depcomp
mkdir build-aux

$ACLOCAL

# This test may have to be run several times in order to expose the
# race that, when the last Makefile.in (the toplevel one) is created
# before the other ones have finished, not all auxiliary files may
# be installed yet, thus some may not be distributed.
#
# Further, automake output should be stable.

# Generate expected output using the non-threaded code.
unset AUTOMAKE_JOBS
AUTOMAKE_run --add-missing
mv stderr expected
Makefile_ins=$(find . -name Makefile.in)
for file in $Makefile_ins; do
  mv $file $file.exp
done

AUTOMAKE_JOBS=5
export AUTOMAKE_JOBS

for run in 1 2 3 4 5 6 7; do
  rm -f build-aux/* sub*/Makefile.in
  AUTOMAKE_run --add-missing
  diff stderr expected
  for file in $Makefile_ins; do
    diff $file $file.exp
  done
done

:
