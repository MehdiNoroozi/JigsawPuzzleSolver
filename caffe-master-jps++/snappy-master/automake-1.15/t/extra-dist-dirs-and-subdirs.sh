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

# Check to make sure EXTRA_DIST can contain a directory or
# a subdirectory, in $(builddir) or $(srcdir).

. test-init.sh

echo AC_OUTPUT >> configure.ac

cat > Makefile.am << 'END'
EXTRA_DIST=foo/bar baz foo2/bar2 baz2

check: distdir
	test -f $(distdir)/foo/bar/baz
	test -f $(distdir)/baz/foo
	test -f $(distdir)/foo2/bar2/baz2
	test -f $(distdir)/baz2/foo2
END

# Create some files in $(srcdir)
mkdir foo
mkdir foo/bar
touch foo/bar/baz
mkdir baz
touch baz/foo

$ACLOCAL
$AUTOMAKE
$AUTOCONF
mkdir build
cd build
../configure

# Create some files in $(builddir)
mkdir foo2
mkdir foo2/bar2
touch foo2/bar2/baz2
mkdir baz2
touch baz2/foo2

$MAKE check

:
