#! /bin/sh
# Copyright (C) 2011-2014 Free Software Foundation, Inc.
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

# Test AM_PROG_AR ordering requirements for older Libtools
# Keep this test in sync with sister test 'ar-lib6a.sh'.

required=libtoolize
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_RANLIB
AC_PROG_LIBTOOL
AM_PROG_AR
END

libtoolize
$ACLOCAL
$AUTOCONF 2>stderr || { cat stderr >&2; exit 1; }
cat stderr >&2

$EGREP 'AC_PROG_LIBTOOL.*before.*AM_PROG_AR' stderr

:
