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

# Make sure that AR, ARFLAGS, and RANLIB can be substituted from configure.ac.

. test-init.sh

cat >> configure.ac << 'END'
AM_PROG_AR
AC_SUBST([AR], ['echo it works'])
AC_SUBST([ARFLAGS], ['>'])
AC_SUBST([RANLIB], ['echo really works >>'])
AC_OUTPUT
END

cat > Makefile.am << 'END'
lib_LIBRARIES = libfoo.a
libfoo_a_SOURCES =
END

:> ar-lib

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure
$MAKE
grep 'it works' libfoo.a
grep 'really works' libfoo.a

:
