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

# Test that := definitions produce warnings, but otherwise work.

. test-init.sh

cat > Makefile.am << 'END'
ICONS := $(wildcard *.xbm)
END

$ACLOCAL
AUTOMAKE_fails
grep ':=.*not portable' stderr

$AUTOMAKE -Wno-portability
grep '^ICONS *:= *\$(wildcard \*\.xbm) *$' Makefile.in

:
