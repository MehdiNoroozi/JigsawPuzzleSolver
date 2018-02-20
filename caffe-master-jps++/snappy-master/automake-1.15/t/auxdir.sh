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

# Test to make sure AC_CONFIG_AUX_DIR works correctly.

. test-init.sh

# The "./." is here so we don't have to mess with subdirs.
cat > configure.ac <<END
AC_INIT([$me], [1.0])
AC_CONFIG_AUX_DIR([./.])
AM_INIT_AUTOMAKE
AC_CONFIG_FILES([Makefile])
END

cat > Makefile.am << 'END'
pkgdata_DATA =
END

cp "$am_scriptdir/mkinstalldirs" .

# The "././" prefix confuses Automake into thinking it is doing a
# subdir build.  Yes, this is hacky.
$ACLOCAL
$AUTOMAKE ././Makefile

grep '/\./\./mkinstalldirs' Makefile.in

:
