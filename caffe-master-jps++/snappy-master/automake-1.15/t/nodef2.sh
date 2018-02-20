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

# Make sure that PACKAGE and VERSION are AC_DEFINEd when requested.

. test-init.sh

# First, check that PACKAGE and VERSION are output by default.

cat > configure.ac << 'END'
AC_INIT([UnIqUe_PaCkAgE], [UnIqUe_VeRsIoN])
AM_INIT_AUTOMAKE
AC_OUTPUT(output)
END

echo 'DEFS = @DEFS@' > output.in

$ACLOCAL
$AUTOCONF
./configure

grep 'DEFS.*-DVERSION=\\"UnIqUe' output

# Then, check that PACKAGE and VERSION are not output if requested.

cat > configure.ac << 'END'
AC_INIT([UnIqUe_PaCkAgE], [UnIqUe_VeRsIoN])
AM_INIT_AUTOMAKE([no-define])
AC_OUTPUT(output Makefile)
END

: > Makefile.am

$ACLOCAL
$AUTOCONF
$AUTOMAKE # Dummy call to make sure Automake grok 'no-define' silently.
./configure

grep 'DEFS.*-DVERSION=\\"UnIqUe' output && exit 1

:
