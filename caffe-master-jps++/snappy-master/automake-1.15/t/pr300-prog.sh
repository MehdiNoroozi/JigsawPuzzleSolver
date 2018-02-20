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

# Make sure that installing subdirectory programs works.
# PR/300

required=cc
. test-init.sh

cat >> configure.ac << 'END'
AC_PROG_CC
AC_OUTPUT
END

cat > Makefile.am << 'END'
bin_PROGRAMS = subdir/wish
subdir_wish_SOURCES = a.c

nobase_bin_PROGRAMS = subdir/want
subdir_want_SOURCES = a.c

test-all: all
	test -f subdir/wish$(EXEEXT)
	test -f subdir/want$(EXEEXT)
test-install: install
	test -f inst/bin/wish$(EXEEXT)
	test -f inst/bin/subdir/want$(EXEEXT)
test-uninstall: uninstall
	test ! -f inst/bin/wish$(EXEEXT)
	test ! -f inst/bin/subdir/want$(EXEEXT)
test-install-strip: install-strip
	test -f inst/bin/wish$(EXEEXT)
	test -f inst/bin/subdir/want$(EXEEXT)
END

cat > a.c << 'END'
#include <stdio.h>
int main ()
{
   printf ("hi liver!\n");
   return 0;
}
END

## A rule in the Makefile should create subdir
# mkdir subdir

$ACLOCAL
$AUTOCONF
$AUTOMAKE --copy --add-missing

./configure --prefix "$(pwd)/inst"

$MAKE test-all
$MAKE test-install
$MAKE test-uninstall
$MAKE test-install-strip

:
