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

# Test to make sure 'make' check works.
# From Ralf Corsepius.

required=GNUmake
. test-init.sh

cat >> configure.ac << 'END'
AM_MAKE_INCLUDE
AC_OUTPUT
END

: > Makefile.am

$ACLOCAL
$AUTOCONF
$AUTOMAKE

export ACLOCAL
export AUTOCONF
export AUTOMAKE

# Do the test twice -- once with make and once with make -w.
# This tests for a bug reported by Rainer Orth (see PR 175).

save="$MAKE"
for flag in '' -w; do
   MAKE="$save $flag" ./configure
   $FGREP 'am__include = include' Makefile
   $sleep
   touch configure.ac
   $MAKE $flag
   $FGREP 'am__include = include' Makefile
   rm -f config.cache
done

exit 0
