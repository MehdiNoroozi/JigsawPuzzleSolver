#! /bin/sh
# Copyright (C) 1996-2014 Free Software Foundation, Inc.
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

# Test to make sure variable interpolation doesn't break other
# features.  Report from Joel N. Weber, II.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PATH_X
AC_PATH_XTRA
END

cat > Makefile.am << 'END'
noinst_PROGRAMS = x
x_SOURCES = x.c
x_LDADD = $(X_EXTRA_LIBS)
END

$ACLOCAL
$AUTOMAKE

:
