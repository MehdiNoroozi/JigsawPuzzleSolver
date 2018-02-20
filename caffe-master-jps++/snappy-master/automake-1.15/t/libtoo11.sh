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

# Make sure config.lt is removed with Libtool 2.2.x's LT_OUTPUT.
# Report by Charles Wilson.

required='cc libtoolize'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_LIBTOOL
m4_ifdef([LT_OUTPUT], [LT_OUTPUT])
AC_OUTPUT
END

: > Makefile.am

libtoolize
$ACLOCAL
$AUTOMAKE --add-missing
$AUTOCONF
./configure
$MAKE distcheck

:
