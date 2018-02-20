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

# Regression test for multiple rules being generated for each target when
# conditionals are present.
# From Richard Boulton.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_CONDITIONAL([BAR], [true])
END

cat > Makefile.am << 'END'
if BAR
BAR_SRCS = bar.c
endif

bin_PROGRAMS = foo
foo_CFLAGS = -DFOO
foo_SOURCES = foo.c
END

$ACLOCAL
$AUTOMAKE

uncondval=$($FGREP 'foo-foo.o: foo.c' Makefile.in)

cat >> Makefile.am << 'END'
foo_SOURCES += $(BAR_SRCS)
END

$AUTOMAKE

condval=$($FGREP 'foo-foo.o: foo.c' Makefile.in)

test "x$uncondval" = "x$condval"

:
