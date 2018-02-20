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

# Test if a file can be mentioned in LTLIBOBJS and explicitly.
# (Like libobj12.sh, but for Libtool libraries.)

required='libtoolize'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AC_LIBOBJ([foo])
AC_OUTPUT
END

cat > Makefile.am << 'END'
noinst_LTLIBRARIES = libfoo.la libbar.la

libfoo_la_SOURCES =
libfoo_la_LIBADD = @LTLIBOBJS@

libbar_la_SOURCES = foo.c
END

: > foo.c

$ACLOCAL
: > ltmain.sh
$AUTOMAKE --add-missing

# This however should be diagnosed, since foo.c is in @LIBOBJS@.
echo 'libfoo_la_SOURCES += foo.c' >> Makefile.am
AUTOMAKE_fails
grep 'foo\.c.*explicitly mentioned' stderr

:
