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

# Test to make sure AC_REPLACE_FUNCS works across lines.  Report from
# Jim Meyering.

. test-init.sh

cat > Makefile.am << 'END'
bin_PROGRAMS = joe
LDADD = @LIBOBJS@
END

cat >> configure.ac << 'END'
AC_PROG_CC
AC_REPLACE_FUNCS(\
   foo_bar_quux)
END

: > foo_bar_quux.c

$ACLOCAL
$AUTOMAKE
$FGREP foo_bar_quux.c Makefile.in

:
