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

# Test for bug in conditionals.
# Report from Robert Boehne.

. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_CONDITIONAL([COND1], [true])
END

cat > Makefile.am << 'END'
if COND1
BUILD_helldl = helldl
helldl_SOURCES = dlmain.c
helldl_DEPENDENCIES = libhello.la
else
BUILD_helldl =
bin_SCRIPTS = helldl
helldl$(EXEEXT):
	rm -f $@
	echo '#! /bin/sh' > $@
	echo '-dlopen is unsupported' >> $@
	chmod +x $@
endif

bin_PROGRAMS = $(BUILD_helldl)
END

$ACLOCAL
$AUTOMAKE

$FGREP helldl Makefile.in # For debugging.
test $($FGREP -c 'helldl$(EXEEXT):' Makefile.in) -eq 2

:
