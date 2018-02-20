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

# Check AC_CONFIG_FILES support for files starting with '../'.
# Report from Bruno Haible.

. test-init.sh

mkdir testdir
cd testdir

mv ../configure.ac .
cat >> configure.ac << END
AC_CONFIG_FILES([a/foo.sh:../testdir/a/foo.sh.in])
AC_CONFIG_FILES([a/Makefile])
AC_OUTPUT
END

mkdir a

echo SUBDIRS = a >Makefile.am
: >a/Makefile.am

echo foo >a/foo.sh.in

$ACLOCAL
$AUTOCONF
$AUTOMAKE --add-missing

./configure
$MAKE
test "$(cat a/foo.sh)" = foo

$sleep
echo 'bar' >a/foo.sh.in

cd a
$MAKE foo.sh
test "$(cat foo.sh)" = bar

:
