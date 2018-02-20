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

# Test that '+=' fails when required.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_RANLIB
END

# If you do this in a real Makefile.am, I will kill you.
cat > Makefile.am << 'END'
lib_LIBRARIES = libq.a
libq_a_SOURCES = q.c
AR += qq
END

$ACLOCAL
AUTOMAKE_fails -Wno-portability
grep "^Makefile\.am:3:.* AR .* with '=' before .*'+='" stderr

:
