#! /bin/sh
# Copyright (C) 1997-2014 Free Software Foundation, Inc.
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

# Test to make sure tags and subdirs work correctly.  Bug report by
# François Pinard, and later by Akim Demaille.

required=etags
. test-init.sh

cat >> configure.ac << 'END'
AC_CONFIG_FILES([sub/Makefile])
AC_OUTPUT
END

echo 'SUBDIRS = sub' > Makefile.am
mkdir sub
echo 'noinst_HEADERS = iguana.h' > sub/Makefile.am
: > sub/iguana.h

$ACLOCAL
$AUTOCONF
$AUTOMAKE

./configure
$MAKE tags
test -f sub/TAGS
test -f TAGS
$FGREP sub/TAGS TAGS
$FGREP iguana.h sub/TAGS

$MAKE distclean
test ! -e sub/TAGS
test ! -e TAGS

:
