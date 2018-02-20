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

# Check to make sure that when distdir is invoked in a VPATH
# configuration and has to distribute directory X, it actually merge
# $(srcdir)/X and ./X, with the files from the later overriding the
# files from the former.

. test-init.sh

echo AC_OUTPUT >> configure.ac

cat > Makefile.am << 'END'
EXTRA_DIST=foo/bar baz

check: distdir
	test -f $(distdir)/foo/bar/baz
	test -f $(distdir)/foo/bar/baz2
	test -f $(distdir)/baz/foo
	test -f $(distdir)/baz/foo2
	grep source $(distdir)/foo/bar/baz
	grep build $(distdir)/foo/bar/baz2
	grep source $(distdir)/baz/foo
	grep build $(distdir)/baz/foo2
END

# Create some files in $(srcdir)
mkdir foo
mkdir foo/bar
echo source > foo/bar/baz
echo source > foo/bar/baz2
mkdir baz
echo source > baz/foo
echo source > baz/foo2

$ACLOCAL
$AUTOMAKE
$AUTOCONF
mkdir build
cd build
../configure

# Create some files in $(builddir) that will override part of the
# files if $(srcdir) when the distribution is made.
mkdir foo
mkdir foo/bar
echo build > foo/bar/baz2
mkdir baz
echo build > baz/foo2

$MAKE check

:
