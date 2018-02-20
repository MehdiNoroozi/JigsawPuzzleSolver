#!/bin/sh
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

# Make sure Automake diagnoses conflicting installations.

required='libtoolize'
. test-init.sh

cat >>configure.ac <<'END'
AM_CONDITIONAL([COND1], [true])
AM_CONDITIONAL([COND2], [false])
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AC_OUTPUT
END

cat >Makefile.am <<'END'
if COND1
  lib_LTLIBRARIES = liba.la
  EXTRA_LTLIBRARIES = libc.la libc.la libb.la
else
  lib_LTLIBRARIES = libb.la
endif
if COND2
if COND1
    pkglib_LTLIBRARIES = liba.la
endif
LIBTOOLFLAGS = ouch
endif
END

libtoolize
$ACLOCAL
AUTOMAKE_fails --add-missing
grep libb stderr && exit 1
grep 'Makefile.am:3:.*libc.la.*multiply defined' stderr
grep "Makefile.am:9:.*'pkglib" stderr
grep "Makefile.am:2:.*'lib" stderr
grep 'Makefile.am:11:.*AM_LIBTOOLFLAGS' stderr
