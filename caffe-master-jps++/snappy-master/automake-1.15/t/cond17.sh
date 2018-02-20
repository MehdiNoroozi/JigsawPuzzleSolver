#! /bin/sh
# Copyright (C) 2001-2014 Free Software Foundation, Inc.
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

# Test for being able to define an object to be generated in different ways
# according to a conditional.
# Report from Richard Boulton.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_CXX
AM_CONDITIONAL([COND1], [true])
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = helldl
if COND1
helldl_SOURCES = foo.cc
else
helldl_SOURCES = foo.c
endif
END

$ACLOCAL
$AUTOMAKE -a

:
