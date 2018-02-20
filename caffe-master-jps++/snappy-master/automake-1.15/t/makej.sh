#! /bin/sh
# Copyright (C) 2003-2014 Free Software Foundation, Inc.
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

# 'make -j' used to fail with Autoconf < 2.58, because tools like
# autoconf and automake can try to update autom4te's cache in parallel.
#
# Note that failures might not be reproducible systematically as they
# depend on the time at which autoconf and automake update the cache
# via autom4te.

required=GNUmake
. test-init.sh

cat >configure.ac <<END
m4_include([version.m4])
AC_INIT([$me], [THE_VERSION])
AM_INIT_AUTOMAKE
AC_CONFIG_HEADER([config.h])
AC_CONFIG_FILES([Makefile])
AC_OUTPUT
END

echo 'm4_define([THE_VERSION], [2.718])' > version.m4

: > Makefile.am

$ACLOCAL
$AUTOCONF
$AUTOHEADER
$AUTOMAKE --add-missing
./configure --version | grep '2\.718'
./configure
$MAKE

$sleep
echo 'm4_define([THE_VERSION], [3.141])' > version.m4
$MAKE -j
./configure --version | grep '3\.141'

:
