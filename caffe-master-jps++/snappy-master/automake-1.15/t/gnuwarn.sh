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

# Check that Automake warns about user variables being overridden.

. test-init.sh

# We need (almost) complete control over automake options.
AUTOMAKE="$am_original_AUTOMAKE -Werror"

cat >> configure.ac << 'END'
AC_PROG_CC
AC_OUTPUT
END

# Needed by --gnu.
: > NEWS
: > README
: > AUTHORS
: > ChangeLog

cat > Makefile.am << 'END'
CFLAGS += -I..
LDFLAGS = -lfoo
CXXFLAGS = -Wall
bin_PROGRAMS = bar
END

$ACLOCAL
# Don't warn in foreign mode
$AUTOMAKE --add-missing --foreign
# Warn in gnu mode
AUTOMAKE_fails --add-missing --gnu
grep '^Makefile\.am:1:.*CFLAGS' stderr
grep '^Makefile\.am:2:.*LDFLAGS' stderr
# No reason to warn about CXXFLAGS since it's not used.
grep CXXFLAGS stderr && exit 1
# Don't warn if -Wno-gnu.
$AUTOMAKE --gnu -Wno-gnu

:
