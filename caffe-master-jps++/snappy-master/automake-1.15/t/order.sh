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

# Make sure AC_DEFUN is handled correctly.  This test isn't named very
# well.  Report from Jim Meyering.

. test-init.sh

cat >> configure.ac << 'END'
jm_TEST
FOO
END

cat > acinclude.m4 << 'END'
AC_DEFUN([jm_TEST])
AC_DEFUN([FOO], [AC_REQUIRE([jm_TEST])])
END

$ACLOCAL
