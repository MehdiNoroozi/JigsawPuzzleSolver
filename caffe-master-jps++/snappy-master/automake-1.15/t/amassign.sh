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

# Test to see if AM_ name can be assigned to in configure.ac.
# Report from Steve Robbins.

. test-init.sh

cat >> configure.ac << 'END'
AM_CFLAGS=foo
AC_SUBST(AM_BAR)
AC_SUBST([AM_ZARDOZ])
END

$ACLOCAL
