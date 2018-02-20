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

# Test to make sure dependencies are generated correctly for .h files.
# Report from Richard Boulton.
#
# Also check that the sources of the generated parser are distributed.
# PR/47.

required='cc yacc'
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_PROG_YACC
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = foo
foo_SOURCES = foo.y
AM_YFLAGS = -d

check-dist: distdir
	test -f $(distdir)/foo.y
	test -f $(distdir)/foo.c
	test -f $(distdir)/foo.h
END

# The %union will cause Bison to output '#line's in y.tab.h too.
cat > foo.y << 'END'
%union
{
  int i;
  char c;
}
%%
WORD: "up";
%%
END

$ACLOCAL
$AUTOMAKE -a
$AUTOCONF
./configure

$MAKE foo.h

test -f foo.h

rm -f foo.h foo.c
$MAKE check-dist

# We should be able to recover if foo.h is deleted.

rm -f foo.h
$MAKE foo.h
test -f foo.h

# Make sure '#line ... y.tab.h' gets replaced.
$FGREP 'y.tab.h' foo.h && exit 1

# Make distclean must not erase foo.c nor foo.h (by GNU standards) ...
$MAKE foo.c
test -f foo.h
test -f foo.c
$MAKE distclean
test -f foo.h
test -f foo.c
# ... but maintainer-clean should.
./configure # Re-create 'Makefile'.
$MAKE maintainer-clean
test ! -e foo.h
test ! -e foo.c

:
