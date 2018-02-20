#! /bin/sh
# Copyright (C) 2006-2014 Free Software Foundation, Inc.
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

# Check for _AM_SUBST_NOTMAKE.

. test-init.sh

cat >> configure.ac << 'END'
AC_SUBST([backslash], "\\")
_AM_SUBST_NOTMAKE([backslash])
AC_OUTPUT
END

cat > Makefile.am << 'END'
test:
	@echo $(backslash) @backslash@$$
END

$ACLOCAL
$AUTOCONF
$AUTOMAKE
./configure

# If _AM_SUBST_NOTMAKE is not honored, the backslash
# variable will not be empty.
$MAKE test | grep '^[$]$'
