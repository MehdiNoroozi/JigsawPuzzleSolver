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

# Check to make sure we recognize a Makefile.in, even if post-processed
# and renamed.

required=cc
. test-init.sh

cat >configure.ac <<END
AC_INIT([$me], [1.0])
AM_INIT_AUTOMAKE
AC_PROG_CC
AC_CONFIG_FILES([myMakefile])
AC_OUTPUT
END

cat > myMakefile.am << 'END'
bin_PROGRAMS = fred
fred_SOURCES = fred.c
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE -- myMakefile

mv myMakefile.in myMakefile.old
echo '# Post-processed by post-processor 3.14.' > myMakefile.in
cat myMakefile.old >> myMakefile.in

./configure

test -f .deps/fred.Po || test -f _deps/fred.Po || exit 1

:
