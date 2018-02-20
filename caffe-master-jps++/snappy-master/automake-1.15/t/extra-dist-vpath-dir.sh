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

# Check to make sure EXTRA_DIST can contain a directory from $buildir.
# From Dean Povey.

. test-init.sh

echo AC_OUTPUT >> configure.ac

cat > Makefile.am << 'END'
EXTRA_DIST=foo

foo:
	mkdir foo
	touch foo/bar
END

$ACLOCAL
$AUTOMAKE
$AUTOCONF
mkdir build
cd build
../configure
$MAKE distdir

:
