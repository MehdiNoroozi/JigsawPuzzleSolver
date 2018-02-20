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

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_PROG_LEX
END

cat > Makefile.am << 'END'
bin_PROGRAMS = zot
zot_SOURCES = joe.l
LDADD = @LEXLIB@
END

$ACLOCAL
$AUTOMAKE -a

# Test to make sure that lex source generates correct target.
$FGREP '$(LEX)' Makefile.in

# Test to make sure that lex source generates correct clean rule.
# From Ralf Corsepius.
$FGREP joel Makefile.in && exit 1

:
