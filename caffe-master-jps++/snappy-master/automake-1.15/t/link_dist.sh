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

# Test to make sure the linker for a dist_*_SOURCES can override that for
# *_SOURCES
# Richard Boulton <richard@tartarus.org>

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_CXX
END

cat > Makefile.am << 'END'
bin_PROGRAMS = lavalamp
lavalamp_SOURCES = lava.c
dist_lavalamp_SOURCES = lamp.cxx
END

$ACLOCAL
$AUTOMAKE

# We should only see the C++ linker in the rules of 'Makefile.in'.

# Look for this macro not at the beginning of any line; that will have
# to be good enough for now.
grep '.\$(CXXLINK)' Makefile.in

# We should not see these patterns:
grep '.\$(FLINK)' Makefile.in && exit 1
grep '.\$(LINK)'  Makefile.in && exit 1

exit 0
