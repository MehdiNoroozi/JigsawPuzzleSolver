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

# Test that AC_SUBST($1) does something sensible.  From Ulrich
# Drepper.

. test-init.sh

cat >> configure.ac << 'END'
dnl This test used to have the following lines, which cannot have
dnl worked sensibly with Autoconf for years, however:
dnl AC_SUBST($1)
dnl AC_SUBST([$]$1)  dnl this is the actual invocation that was used
dnl
AC_DEFUN([FOO],
[AC_SUBST([$1])])
FOO([BAR])
END

: > Makefile.am

$ACLOCAL
$AUTOMAKE
grep '^\$1' Makefile.in && exit 1

:
