#!/bin/sh
# Copyright (C) 2008-2014 Free Software Foundation, Inc.
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

# Test for libtool errors for multiple install locations, esp. with nobase.


required='libtoolize'
. test-init.sh

cat >>configure.ac <<'END'
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AM_CONDITIONAL([COND], [:])
AC_OUTPUT
END

cat >Makefile.am <<'END'
if COND
lib_LTLIBRARIES = liba1.la sub/liba2.la
#else
pkglib_LTLIBRARIES = liba1.la
nobase_lib_LTLIBRARIES = sub/liba2.la
endif
AUTOMAKE_OPTIONS = subdir-objects
END

libtoolize
$ACLOCAL
$AUTOCONF
AUTOMAKE_fails --add-missing

# libtoolize might have installed config.guess and config.sub already,
# and autom4te might warn about bugs in Libtool macro files, so filter
# out warnings about Makefile.am only.  We don't care in this test
# whether automake installs config.guess, config.sub and ar-lib.

cat >expected <<'END'
Makefile.am:5: error: sub/liba2.la multiply defined in condition COND
Makefile.am:5: 'sub/liba2.la' should be installed below 'lib' in condition COND ...
Makefile.am:2: ... and should also be installed in 'lib' in condition COND.
Makefile.am:4: error: liba1.la multiply defined in condition COND
Makefile.am:4: 'liba1.la' should be installed in 'pkglib' in condition COND ...
Makefile.am:2: ... and should also be installed in 'lib' in condition COND.
Makefile.am:2: Libtool libraries can be built for only one destination
END

grep '^Makefile.am' stderr | diff - expected

sed 's/#//' < Makefile.am > t
mv -f t Makefile.am

$AUTOMAKE
grep ' -rpath \$(libdir)/sub' Makefile.in

:
