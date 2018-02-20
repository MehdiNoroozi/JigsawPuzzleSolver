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

# Test for bug in conditionals.  From Raja R Harinath.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_CONDITIONAL([USE_A], [test x = y])
AM_CONDITIONAL([USE_B], [test x = z])
AC_OUTPUT
END

cat > Makefile.am << 'END'
if USE_A
out=output_a.c
else
if USE_B
out=output_b.c
else
out=output_c.c
endif
endif

noinst_PROGRAMS=foo
foo_SOURCES=foo.c $(out)
END

$ACLOCAL
$AUTOMAKE -a
grep 'USE_A_FALSE.*USE_B_FALSE.*output_c\...OBJEXT.' Makefile.in

:
