#! /bin/sh
# Copyright (C) 1997-2014 Free Software Foundation, Inc.
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

# Test to make sure multiple source files with the same object
# file causes error.

. test-init.sh

cat > Makefile.am << 'END'
bin_PROGRAMS = zardoz
zardoz_SOURCES = z.c x/z.c
END

cat >> configure.ac << 'END'
AC_PROG_CC
END

mkdir x

: > z.c
: > x/z.c

$ACLOCAL
AUTOMAKE_fails
$FGREP 'z.$(OBJEXT)' stderr
