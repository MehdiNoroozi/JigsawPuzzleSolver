#!/bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Make sure it's OK to install a library under different conditions
# in different directories.  PR/285.

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
endif
if COND2
  pkglib_LTLIBRARIES = liba.la
endif
END

libtoolize
$ACLOCAL
$AUTOMAKE --add-missing
# am_liba_la_rpath is defined twice, and used once
test 3 -eq $(grep -c 'am_liba_la_rpath' Makefile.in)

:
