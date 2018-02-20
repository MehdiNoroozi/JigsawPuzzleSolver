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

# Make sure aclocal picks up on all macro definitions.

. test-init.sh

cat > acinclude.m4 << 'END'
AC_DEFUN([AM_FUNC_THREE])
AC_DEFUN([AM_FUNC_TWO], [@!GrepThisString!@AM_FUNC_THREE])
AC_DEFUN([AM_FUNC_ONE], [AC_REQUIRE([AM_FUNC_TWO])])
END

cat >>configure.ac << 'END'
AM_FUNC_ONE
END

$ACLOCAL
$AUTOCONF
grep '^@!GrepThisString!@$' configure

:
