#! /bin/sh
# Copyright (C) 1996-2014 Free Software Foundation, Inc.
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

# Test to ensure --gnits version checking is correct.

. test-init.sh

cat > configure.ac << END
AC_INIT([$me], [3.5.3.2])
AM_INIT_AUTOMAKE
AC_CONFIG_FILES(Makefile)
END

cat > Makefile.am << 'END'
pkgdata_DATA =
END

# Files required by Gnits.
: > INSTALL
: > NEWS
: > README
: > COPYING
: > AUTHORS
: > ChangeLog
: > THANKS

$ACLOCAL
AUTOMAKE_fails --gnits
grep 'configure.ac:.*3\.5\.3\.2' stderr
