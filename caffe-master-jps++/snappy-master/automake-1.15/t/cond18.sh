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

# Regression test for substitution references to conditional variables.
# Report from Richard Boulton.

. test-init.sh

cat >> configure.ac << 'END'
AM_CONDITIONAL([COND1], [true])
AM_CONDITIONAL([COND2], [true])
AC_OUTPUT
END

cat > Makefile.am << 'END'
AUTOMAKE_OPTIONS = no-dependencies
CC = false
OBJEXT = obj

var1 = dlmain

if COND1
var2 = $(var1:=.c) foo.cc
else
var2 = $(var1:=.c)
endif

if COND2
var3 = $(var2:.cc=.c)
else
var3 = $(var2:.cc=.c)
endif

helldl_SOURCES = $(var3)

.PHONY: test
test:
	is $(helldl_SOURCES) $(helldl_OBJECTS) == \
           dlmain.c foo.c dlmain.obj foo.obj

bin_PROGRAMS = helldl
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE -a
./configure
$MAKE test

:
