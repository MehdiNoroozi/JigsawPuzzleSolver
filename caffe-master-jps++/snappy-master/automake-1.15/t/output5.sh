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

# Make sure Automake complains when there is no Makefile specified.

. test-init.sh

cat > configure.ac << 'END'
AC_INIT([foo], [bar], [baz])
AM_INIT_AUTOMAKE
AC_OUTPUT
END

: > Makefile.am

$ACLOCAL
AUTOMAKE_fails
grep 'AC_CONFIG_FILES(.Makefile.)' stderr
