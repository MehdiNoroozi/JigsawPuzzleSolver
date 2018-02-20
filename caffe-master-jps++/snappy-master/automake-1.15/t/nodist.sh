#! /bin/sh
# Copyright (C) 1999-2014 Free Software Foundation, Inc.
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

# Test to make sure dist_*_SOURCES and nodist_*_SOURCES work.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
END

cat > Makefile.am << 'END'
bin_PROGRAMS = eyeball

eyeball_SOURCES = a.c
nodist_eyeball_SOURCES = b.c
dist_eyeball_SOURCES = c.c
END

$ACLOCAL
$AUTOMAKE

grep '^am_eyeball_OBJECTS' Makefile.in
grep '^DIST_SOURCES =' Makefile.in
grep '^DIST_SOURCES =.*nodist' Makefile.in && exit 1

:
