#! /bin/sh
# Copyright (C) 2004-2014 Free Software Foundation, Inc.
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

# Make sure aclocal scans configure.ac for macro definitions.
# PR/319.

am_create_testdir=empty
. test-init.sh

# Start macros with AM_ because that causes aclocal to complain if it
# cannot find them.

cat > configure.ac << 'END'
AC_INIT
m4_include([somedef.m4])
AC_DEFUN([AM_SOME_MACRO])
AC_DEFUN([AM_SOME_OTHER_MACRO])
AM_SOME_MACRO
AM_SOME_OTHER_MACRO
AM_MORE_MACRO
END

mkdir m4
echo 'AC_DEFUN([AM_SOME_MACRO])' > m4/some.m4
echo 'AC_DEFUN([AM_SOME_DEF])' > somedef.m4
echo 'AC_DEFUN([AM_MORE_MACRO], [AC_REQUIRE([AM_SOME_DEF])])' > m4/more.m4

$ACLOCAL -I m4
$FGREP AM_SOME_MACRO aclocal.m4 && exit 1
$FGREP AM_MORE_MACRO aclocal.m4 && exit 1
$FGREP 'm4_include([m4/more.m4])' aclocal.m4
test 1 = $(grep m4_include aclocal.m4 | wc -l)

:
