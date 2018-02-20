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

# Test to make sure a sensible default source for libraries is used.

required='cc libtool'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AM_PROG_AR
AC_PROG_LIBTOOL
AC_OUTPUT
END

mkdir zoo.d

cat > Makefile.am << 'END'
AM_LDFLAGS = -module
pkglib_LTLIBRARIES = zoo.d/bar.la old.la
noinst_LTLIBRARIES = foo.la zoo.d/old2.la

$(srcdir)/zoo_d_old2_la.c: $(srcdir)/old_la.c
	cp $(srcdir)/old_la.c $@

AUTOMAKE_OPTIONS = -Wno-unsupported
END

cat > foo.c << 'END'
int foo (void)
{
  return 0;
}
END

cp foo.c zoo.d/bar.c
cp foo.c old_la.c

libtoolize
$ACLOCAL
$AUTOCONF
AUTOMAKE_fails -a
grep '^Makefile\.am:2:.*old_la\.c' stderr
grep '^Makefile\.am:2:.*old\.c' stderr
grep '^Makefile\.am:3:.*zoo_d_old2_la\.c' stderr
grep '^Makefile\.am:3:.*zoo\.d/old2\.c' stderr

$AUTOMAKE -Wno-obsolete

mkdir sub
cd sub

../configure
$MAKE

test -f foo.la
test -f zoo.d/bar.la
test -f old.la
test -f zoo.d/old2.la

$MAKE distcheck

:
