#! /bin/sh
# Copyright (C) 2002-2014 Free Software Foundation, Inc.
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

# Regression test for a bug reported by Andrew Suffield.
# (Automake goes wild and try to rerun itself more than two time
# to fix the Makefiles.)

required='libtoolize'
. test-init.sh

cat > configure.ac << 'END'
AC_INIT([req2], [1.0])
AC_CONFIG_AUX_DIR([autoconf])
AM_INIT_AUTOMAKE
AC_CONFIG_FILES([Makefile])
AC_PROG_CC
AM_PROG_AR
AM_PROG_LIBTOOL
AC_CONFIG_FILES([autoconf/Makefile main/Makefile])
AC_OUTPUT
END

mkdir autoconf
mkdir main

: > autoconf/Makefile.am
echo 'SUBDIRS = autoconf main' >Makefile.am

cat >main/Makefile.am <<'END'
lib_LTLIBRARIES = lib0.la
lib0_la_SOURCES = 0.c
END

: > ar-lib
libtoolize --force --copy
$ACLOCAL
$AUTOCONF

test -f autoconf/ltmain.sh # Sanity check.
rm -f autoconf/ltmain.sh
AUTOMAKE_fails --add-missing --copy
grep '^configure\.ac:7:.* required file.*autoconf/ltmain\.sh' stderr

:
