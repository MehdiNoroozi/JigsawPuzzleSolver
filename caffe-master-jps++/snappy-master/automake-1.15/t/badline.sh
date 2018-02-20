#! /bin/sh
# Copyright (C) 1998-2014 Free Software Foundation, Inc.
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

# Test to make sure that line number and file name in error message
# referring to variable is always correct.

. test-init.sh

cat > configure.ac << END
AC_INIT([$me], [1.0])
AM_INIT_AUTOMAKE
AC_SUBST(TEXINFOS)
AC_CONFIG_FILES([Makefile])
END

: > Makefile.am

$ACLOCAL || exit 1
AUTOMAKE_fails
grep 'configure.ac:3:.*info_TEXINFOS' stderr
